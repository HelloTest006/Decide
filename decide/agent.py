"""Jev & Laya compatible System-1 Decision Agent (`predict`, `system_one`, `predict_batch`)."""
import json
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

from .config import DecideConfig, QTYPES
from .model import DecideModel
from .tokenizer import FastDecideTokenizer, render_options_with_keys


class DecideAgent:
    """
    Drop-in Jev & Laya compatible System-1 Decision Agent (`predict`, `system_one`, `predict_batch`).
    Encodes `state` ONCE and answers arbitrary numbers of `choice`, `score`, and `noul` questions in parallel.
    """
    def __init__(
        self,
        model: DecideModel,
        tokenizer: Optional[FastDecideTokenizer] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.cfg: DecideConfig = model.cfg
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer or FastDecideTokenizer(self.cfg.vocab_size)
        self.amp_dtype = (
            torch.bfloat16
            if (self.device.type == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float16
        )

    def prepare_inputs(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if state is None:
            raise ValueError("state must not be None")
        state_str = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        state_ids, state_mask = self.tokenizer.encode_batch([state_str], max_len=self.cfg.max_state_len)

        qids = list(questions.keys())
        Q = len(qids)
        q_texts = []
        all_opt_keys = []
        all_opt_seqs = []
        qtypes = []

        for qid in qids:
            qdef = questions[qid]
            t = qdef["type"]
            if t not in QTYPES:
                raise ValueError(f"Question {qid!r}: unsupported type {t!r}")
            ins = qdef.get("instructions")
            if not ins or (isinstance(ins, str) and not ins.strip()):
                raise ValueError(f"Question {qid!r}: instructions must not be empty")
            qtypes.append(QTYPES[t])
            ins_str = ins if isinstance(ins, str) else json.dumps(ins, ensure_ascii=False)
            q_texts.append(f"{t} question: {ins_str}")
            keys, opt_texts = render_options_with_keys(qdef)
            all_opt_keys.append(keys)
            opt_seqs = [self.tokenizer.encode_text(ot, max_len=self.cfg.max_opt_len) for ot in opt_texts]
            all_opt_seqs.append(opt_seqs)

        q_ids, q_mask = self.tokenizer.encode_batch(q_texts, max_len=96)
        K_max = max(len(seqs) for seqs in all_opt_seqs)
        Lo_max = max(max(len(s) for seqs in all_opt_seqs for s in seqs), 4)

        opt_ids = torch.zeros((Q, K_max, Lo_max), dtype=torch.long)
        opt_tok_mask = torch.zeros((Q, K_max, Lo_max), dtype=torch.long)
        opt_valid_mask = torch.zeros((Q, K_max), dtype=torch.bool)

        for i, seqs in enumerate(all_opt_seqs):
            for k_idx, s in enumerate(seqs):
                opt_ids[i, k_idx, :len(s)] = torch.tensor(s, dtype=torch.long)
                opt_tok_mask[i, k_idx, :len(s)] = 1
                opt_valid_mask[i, k_idx] = True

        return {
            "qids": qids,
            "opt_keys": all_opt_keys,
            "state_ids": state_ids.to(self.device),
            "state_mask": state_mask.to(self.device),
            "q_ids": q_ids.to(self.device),
            "q_mask": q_mask.to(self.device),
            "opt_ids": opt_ids.to(self.device),
            "opt_tok_mask": opt_tok_mask.to(self.device),
            "opt_valid_mask": opt_valid_mask.to(self.device),
            "qtype": torch.tensor(qtypes, dtype=torch.long, device=self.device),
        }

    @torch.no_grad()
    def predict(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        min_confidence: Optional[Union[float, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        if state is None:
            raise ValueError("state must not be None")
        if not questions:
            return {"model": self.cfg.model_name, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}

        self.model.eval()
        prep = self.prepare_inputs(state, questions)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp):
            state_cache = self.model.encode_state(prep["state_ids"], prep["state_mask"])
            out = self.model.forward_questions(
                state_cache,
                prep["q_ids"],
                prep["q_mask"],
                prep["opt_ids"],
                prep["opt_tok_mask"],
                prep["opt_valid_mask"],
                prep["qtype"],
            )

        probs_np = out["probs"].float().cpu().numpy()
        act_probs_np = F.softmax(out["act_logits"].float(), dim=-1).cpu().numpy()
        conf_np = out["confidence"].float().cpu().numpy()
        ans_conf_np = out["answer_confidence"].float().cpu().numpy()

        state_toks = int(prep["state_mask"].sum().item())
        q_toks = int(prep["q_mask"].sum().item() + prep["opt_tok_mask"].sum().item())
        answers = {}

        for r, qid in enumerate(prep["qids"]):
            qdef = questions[qid]
            t = qdef["type"]
            keys = prep["opt_keys"][r]
            k = len(keys)
            p = probs_np[r, :k]
            p = p / max(p.sum(), 1e-12)
            act_p = round(float(act_probs_np[r, 0]), 4)
            conf = round(float(conf_np[r]), 4)
            ans_conf = round(float(ans_conf_np[r]), 4)

            if t == "choice":
                ans = {
                    "type": "choice",
                    "choice": keys[int(p.argmax())],
                    "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)},
                    "confidence": conf,
                    "answer_confidence": ans_conf,
                    "action": {"act_probability": act_p},
                }
            elif t == "score":
                crit_list = qdef["criteria"]
                expected_score = float((np.arange(k) * p).sum())
                ans = {
                    "type": "score",
                    "score": round(expected_score, 4),
                    "legend": {str(i): str(c) for i, c in enumerate(crit_list)},
                    "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
                    "confidence": conf,
                    "answer_confidence": ans_conf,
                    "action": {"act_probability": act_p},
                }
            else:
                ans = {
                    "type": "noul",
                    "noul": round(float(p[1]), 4),
                    "confidence": ans_conf,
                    "answer_confidence": ans_conf,
                    "action": {"act_probability": act_p},
                }

            if min_confidence is not None:
                thresh = (
                    float(min_confidence)
                    if isinstance(min_confidence, (int, float))
                    else float(min_confidence.get("default", 0.7))
                )
                ans["abstention"] = "passed" if ans_conf >= thresh else "abstained"
                ans["abstention_threshold"] = thresh
                if ans_conf < thresh:
                    ans["low_confidence"] = True

            answers[qid] = ans

        return {
            "model": self.cfg.model_name,
            "family": self.cfg.family,
            "answers": answers,
            "usage": {
                "input_tokens": state_toks + q_toks,
                "state_tokens": state_toks,
                "question_tokens": q_toks,
                "output_tokens": 0,
            },
        }

    system_one = predict

    @torch.no_grad()
    def predict_batch(
        self,
        states: List[Any],
        questions: Dict[str, Dict[str, Any]],
        min_confidence: Optional[Union[float, Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        return [self.predict(s, questions, min_confidence=min_confidence) for s in states]
