"""Router for dispatching and confidence-cascading between Decide (Standard) and DecideX (Large)."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch

from .agent import DecideAgent
from .config import DecideConfig
from .model import DecideModel
from .tokenizer import FastDecideTokenizer


class DecideRouter:
    """
    Routes between `Decide` (Standard fast reflex) and `DecideX` (Larger high-accuracy sibling).
    Supports explicit pinning (`model="decide"` or `model="decidex"`) and automatic cascade escalation.
    """
    def __init__(
        self,
        decide_agent: Optional[DecideAgent] = None,
        decidex_agent: Optional[DecideAgent] = None,
        device: Optional[Union[str, torch.device]] = None,
        cascade_threshold: float = 0.75,
    ):
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tok = FastDecideTokenizer()
        weights_dir = Path(__file__).resolve().parent.parent / "weights"
        if decide_agent is None:
            d_mdl = DecideModel(DecideConfig.decide())
            d_ckpt = weights_dir / "decide" / "model.safetensors"
            if d_ckpt.exists():
                from safetensors.torch import load_file
                d_mdl.load_state_dict({k: v.float() for k, v in load_file(str(d_ckpt)).items()}, strict=True)
            decide_agent = DecideAgent(d_mdl.to(dev), tok, dev)
        if decidex_agent is None:
            dx_mdl = DecideModel(DecideConfig.decidex())
            dx_ckpt = weights_dir / "decidex" / "model.safetensors"
            if dx_ckpt.exists():
                from safetensors.torch import load_file
                dx_mdl.load_state_dict({k: v.float() for k, v in load_file(str(dx_ckpt)).items()}, strict=True)
            decidex_agent = DecideAgent(dx_mdl.to(dev), tok, dev)
        self.decide = decide_agent
        self.decidex = decidex_agent
        self.cascade_threshold = cascade_threshold

    def predict(
        self,
        state: Any,
        questions: Dict[str, Dict[str, Any]],
        model: str = "auto",
        min_confidence: Optional[Union[float, Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        if not questions:
            res = self.decide.predict(state, questions, min_confidence=min_confidence)
            res["routing"] = {"model": "decide", "reason": "empty question set"}
            return res

        m = model.lower().strip()
        if m in ("decide", "standard", "fast", "decide-v1"):
            res = self.decide.predict(state, questions, min_confidence=min_confidence)
            res["routing"] = {"model": "decide", "reason": "explicitly pinned to Decide (Standard)"}
            return res
        if m in ("decidex", "large", "x", "decidex-v1"):
            res = self.decidex.predict(state, questions, min_confidence=min_confidence)
            res["routing"] = {"model": "decidex", "reason": "explicitly pinned to DecideX (Large)"}
            return res

        max_opts = max(
            (len(q.get("criteria", [])) if q["type"] in ("choice", "score") else 2)
            for q in questions.values()
        )
        if max_opts > 16:
            res = self.decidex.predict(state, questions, min_confidence=min_confidence)
            res["routing"] = {"model": "decidex", "reason": f"high-cardinality option space (K={max_opts} > 16)"}
            return res

        res = self.decide.predict(state, questions, min_confidence=min_confidence)
        min_ans_conf = min(a["answer_confidence"] for a in res["answers"].values())
        if min_ans_conf < self.cascade_threshold:
            res_x = self.decidex.predict(state, questions, min_confidence=min_confidence)
            res_x["routing"] = {
                "model": "decidex",
                "escalated": True,
                "reason": f"Decide confidence {min_ans_conf:.3f} < {self.cascade_threshold} cascade threshold",
            }
            return res_x

        res["routing"] = {
            "model": "decide",
            "escalated": False,
            "reason": f"fast reflex confident ({min_ans_conf:.3f} >= {self.cascade_threshold})",
        }
        return res

    system_one = predict

    def predict_batch(
        self,
        requests: List[Dict[str, Any]],
        min_confidence: Optional[Union[float, Dict[str, float]]] = None,
    ) -> List[Dict[str, Any]]:
        return [
            self.predict(
                r["state"],
                r["questions"],
                model=r.get("model", "auto"),
                min_confidence=r.get("min_confidence", min_confidence),
            )
            for r in requests
        ]
