"""Fast deterministic tokenizer and option renderer for Decide & DecideX."""
import json
from typing import Any, Dict, List, Tuple
import torch


class FastDecideTokenizer:
    """
    Deterministic low-latency word + FNV-1a subword hash tokenizer for Decide & DecideX.
    Supports arbitrary text, JSON dicts, and conversation lists without option token starvation.
    """
    def __init__(self, vocab_size: int = 32768):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.opt_token_id = 3
        self.mask_token_id = 4
        self.mask_token = "[MASK]"

    def encode_text(self, text: Any, max_len: int = 512, add_cls: bool = True) -> List[int]:
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False)
        clean = (
            text.lower()
            .replace(",", " , ")
            .replace(".", " . ")
            .replace(":", " : ")
            .replace("?", " ? ")
            .replace("!", " ! ")
        )
        tokens = clean.split()
        ids = [self.cls_token_id] if add_cls else []
        budget = max_len - (2 if add_cls else 0)
        for tok in tokens[:budget]:
            h = 2166136261
            for ch in tok:
                h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
            ids.append(5 + (h % (self.vocab_size - 10)))
        if add_cls:
            ids.append(self.sep_token_id)
        return ids[:max_len]

    def encode_batch(self, texts: List[Any], max_len: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode_text(t, max_len=max_len) for t in texts]
        L = max(max(len(s) for s in seqs), 4)
        B = len(seqs)
        ids = torch.zeros((B, L), dtype=torch.long)
        mask = torch.zeros((B, L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return ids, mask


def render_options_with_keys(qdef: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice":
        if isinstance(crit, list):
            if len(crit) != len(set(str(c) for c in crit)):
                raise ValueError("choice criteria list must have unique labels")
            keys = [str(c) for c in crit]
            texts = [str(c) for c in crit]
        elif isinstance(crit, dict):
            keys = [str(k) for k in crit.keys()]
            texts = [f"{k}: {v}" if v else str(k) for k, v in crit.items()]
        else:
            raise ValueError("choice criteria must be list or dict")
        if len(keys) < 1:
            raise ValueError("choice must provide at least 1 option")
        return keys, texts
    elif t == "score":
        if not isinstance(crit, list) or len(crit) < 2:
            raise ValueError("score criteria must be a list of >=2 ordered rubric levels")
        keys = [str(i) for i in range(len(crit))]
        texts = [f"level {i}: {c}" for i, c in enumerate(crit)]
        return keys, texts
    elif t == "noul":
        crit = crit or {}
        f_desc = crit.get("false") or crit.get(False) or "no, false, statement does not hold"
        t_desc = crit.get("true") or crit.get(True) or "yes, true, statement holds"
        return ["false", "true"], [f"false: {f_desc}", f"true: {t_desc}"]
    else:
        raise ValueError(f"Unsupported question type: {t}")
