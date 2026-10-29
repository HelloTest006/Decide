"""
Public HuggingFace Benchmark Fine-Tuning & Evaluation Script (`PolyAI/banking77`, `dair-ai/emotion`,
`fancyzhx/ag_news`, and `pngwn/typed-decisions-v2-system-one`).

Note: This script evaluates supervised in-domain adaptation by training `Decide` and `DecideX`
on each dataset's official `train` split and evaluating on its `test` split.
"""
from typing import Dict, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from decide.model import FastSelfAttentionBlock
from decide.runtime import StructuredStateEnricher

_PUNCT_TRANS_BENCH = str.maketrans({c: " " for c in '.,!?;:"()[]{}<>/\\|@#$%^&*~`+='})


class RealBenchmarkNgramTokenizer:
    """Fast vectorized unigram + bigram + subword stem tokenizer for public benchmark evaluation."""
    def __init__(self, vocab_size: int = 131072):
        self.vocab_size = vocab_size

    def encode_batch(
        self, texts: List[str], max_words: int = 56, add_stems: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(texts)
        mod = self.vocab_size - 10
        seqs = []
        max_l = 4
        for t in texts:
            words = t.lower().translate(_PUNCT_TRANS_BENCH).split()[:max_words]
            n = len(words)
            ids = [1] + [5 + (hash(w) % mod) for w in words]
            if n > 1:
                ids.extend(5 + (hash(words[i] + "_" + words[i + 1]) % mod) for i in range(n - 1))
            if add_stems:
                ids.extend(5 + (hash("s_" + w[:5]) % mod) for w in words if len(w) >= 5)
            ids.append(2)
            seqs.append(ids)
            if len(ids) > max_l:
                max_l = len(ids)
        out_ids = np.zeros((B, max_l), dtype=np.int64)
        out_mask = np.zeros((B, max_l), dtype=np.int64)
        for i, s in enumerate(seqs):
            l = len(s)
            out_ids[i, :l] = s
            out_mask[i, :l] = 1
        return torch.from_numpy(out_ids), torch.from_numpy(out_mask)


def run_decide_family_on_hf_dataset(
    tr_texts: List[str],
    tr_labels: List[int],
    te_texts: List[str],
    te_labels: List[int],
    label_names: List[str],
    device: torch.device,
    epochs: int = 6,
    bs: int = 128,
    decide_cap: int = 36,
) -> Dict[str, float]:
    """Trains and evaluates both `Decide` (Standard) and `DecideX` (Larger Sibling) on a public dataset's train/test split."""
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    btok = RealBenchmarkNgramTokenizer(131072)
    tr_ids_full, tr_mask_full = btok.encode_batch(tr_texts, max_words=56, add_stems=True)
    te_ids_full, te_mask_full = btok.encode_batch(te_texts, max_words=56, add_stems=True)
    tr_ids_full, tr_mask_full = tr_ids_full.to(device), tr_mask_full.to(device)
    te_ids_full, te_mask_full = te_ids_full.to(device), te_mask_full.to(device)

    tr_y = torch.tensor(tr_labels, dtype=torch.long, device=device)
    te_y = torch.tensor(te_labels, dtype=torch.long, device=device)
    opt_strs = [f"option {i} {name.replace('_', ' ')}" for i, name in enumerate(label_names)]
    opt_ids, opt_mask = btok.encode_batch(opt_strs, max_words=16, add_stems=False)
    opt_ids, opt_mask = opt_ids.to(device), opt_mask.to(device)

    results = {}
    for line, cap_len, d_m, nl, use_ms in [
        ("Decide", decide_cap, 384, 1, False),
        ("DecideX", tr_ids_full.size(1), 640, 2, True),
    ]:
        tr_ids = tr_ids_full[:, :cap_len]
        tr_mask = tr_mask_full[:, :cap_len]
        te_ids = te_ids_full[:, :cap_len]
        te_mask = te_mask_full[:, :cap_len]
        torch.manual_seed(42)
        tok_emb = nn.Embedding(131072, d_m, padding_idx=0).to(device)
        blocks = nn.ModuleList(
            [FastSelfAttentionBlock(d_m, max(4, d_m // 64), d_m * 2, 0.05, n_layers=nl) for _ in range(nl)]
        ).to(device)
        state_head = nn.Sequential(nn.LayerNorm(d_m), nn.Dropout(0.15), nn.Linear(d_m, d_m)).to(device)
        opt_head = nn.Sequential(nn.LayerNorm(d_m), nn.Linear(d_m, d_m), nn.LayerNorm(d_m)).to(device)

        params = list(tok_emb.parameters()) + list(blocks.parameters()) + list(state_head.parameters()) + list(opt_head.parameters())
        lr_line = 2.4e-3 if line == "Decide" else 1.7e-3
        opt = torch.optim.AdamW(params, lr=lr_line, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * (len(tr_ids) // bs + 1))

        for _ in range(epochs):
            perm = torch.randperm(len(tr_ids), device=device)
            for i in range(0, len(tr_ids), bs):
                idx = perm[i:i + bs]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                    ids_b, mask_b = tr_ids[idx], tr_mask[idx]
                    raw_e = tok_emb(ids_b)
                    mf = mask_b.unsqueeze(-1).type_as(raw_e)
                    lex_pool = (raw_e * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                    h = raw_e
                    m4d = mask_b[:, None, None, :].bool()
                    for blk in blocks:
                        h = blk(h, m4d)
                    ctx_pool = (h * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                    s_vec = state_head(lex_pool + 0.3 * ctx_pool)

                    oh = tok_emb(opt_ids)
                    omf = opt_mask.unsqueeze(-1).type_as(oh)
                    o_pool = (oh * omf).sum(dim=1) / omf.sum(dim=1).clamp(min=1.0)
                    w_k = opt_head(o_pool)
                    logits = torch.matmul(s_vec, w_k.t()) / math.sqrt(d_m)
                    if use_ms:
                        tok_sim = torch.matmul(raw_e, w_k.t()) / math.sqrt(d_m)
                        tok_sim = tok_sim.masked_fill(~mask_b.unsqueeze(-1).bool(), -1e4)
                        logits = logits + 0.25 * tok_sim.max(dim=1).values
                    loss = F.cross_entropy(logits, tr_y[idx], label_smoothing=0.02)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
            oh = tok_emb(opt_ids)
            omf = opt_mask.unsqueeze(-1).type_as(oh)
            o_pool = (oh * omf).sum(dim=1) / omf.sum(dim=1).clamp(min=1.0)
            w_k = opt_head(o_pool)
            preds = []
            for i in range(0, len(te_ids), 512):
                ids_b, mask_b = te_ids[i:i + 512], te_mask[i:i + 512]
                raw_e = tok_emb(ids_b)
                mf = mask_b.unsqueeze(-1).type_as(raw_e)
                lex_pool = (raw_e * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                h = raw_e
                m4d = mask_b[:, None, None, :].bool()
                for blk in blocks:
                    h = blk(h, m4d)
                ctx_pool = (h * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                s_vec = state_head(lex_pool + 0.3 * ctx_pool)
                logits = torch.matmul(s_vec, w_k.t()) / math.sqrt(d_m)
                if use_ms:
                    tok_sim = torch.matmul(raw_e, w_k.t()) / math.sqrt(d_m)
                    tok_sim = tok_sim.masked_fill(~mask_b.unsqueeze(-1).bool(), -1e4)
                    logits = logits + 0.25 * tok_sim.max(dim=1).values
                preds.append(logits.argmax(dim=-1))
            preds = torch.cat(preds)
            results[line] = float((preds == te_y).float().mean().item())
    return results


if __name__ == "__main__":
    from datasets import load_dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Supervised Public HuggingFace Dataset Fine-Tuning on {dev}...")
    ds_b77 = load_dataset("PolyAI/banking77")
    b77 = run_decide_family_on_hf_dataset(
        list(ds_b77["train"]["text"]), list(ds_b77["train"]["label"]),
        list(ds_b77["test"]["text"]), list(ds_b77["test"]["label"]),
        ds_b77["train"].features["label"].names, dev, epochs=7, bs=128,
    )
    print(f"PolyAI/banking77 (77-way, fine-tuned on train split): Decide={b77['Decide']*100:.2f}% | DecideX={b77['DecideX']*100:.2f}%")
