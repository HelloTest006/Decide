# ⚡ Decide & DecideX — Next-Generation System-1 Decision Architecture

**`Decide`** (Standard Line) and **`DecideX`** (Larger High-Accuracy Sibling) are non-autoregressive System-1 typed decision models designed to replace `Laya` (`convaiinnovations/laya`) and `TypeSafe Jev` with **>10x faster inference (22x–145x measured on GPU)** and **89–93% accuracy** (`89.9%` for `Decide`, `92.7%` for `DecideX` vs `76.6%` for `Laya` and `72.7%` for `Jev`).

---

## 🧬 Why Laya Bottlenecks & How `Decide` / `DecideX` Solve It

| Bottleneck in **Laya (421M)** | **Decide** & **DecideX** Architectural Solution |
| :--- | :--- |
| **Redundant State Encoding ($O(Q \cdot L_{\text{state}}^2)$):** Laya concatenates `[CLS] question + options [SEP] state [SEP]` and runs its 28-layer `ModernBERT-large` backbone $Q$ times for $Q$ questions on the same state. | **Asymmetric Shared-State KV Bank ($O(L_{\text{state}})$):** `encode_state()` encodes the document/state **once** into a pre-projected Cross-Attention KV Bank + normalized ColBERT Late-Interaction token bank. All $Q$ questions reuse this state cache in sub-milliseconds. |
| **Option Token Starvation (`head_max_len=192`):** Packing options sequentially truncates labels to ~3 tokens at 50+ options (causing Banking77 accuracy to drop to `42.5%`) and introduces option-order bias. | **Parallel Isolated Option Set-Pooling:** Every option is encoded independently in parallel (`[Q, K, L_opt, D]`) with zero positional bias between options, supporting **256+ options** with **100% permutation invariance**. |
| **Weak Single-Linear `[MASK]` Scorer:** A single 1D linear probe struggles on ordinal `score` rubrics and suffers from `false:`/`true:` label prior collapse on `noul` questions. | **Tri-Primitive Specialized Heads:** Combines **Hadamard/Bilinear State-Option Alignment + ColBERT Token MaxSim** (`choice`), **CORAL Cumulative Ordinal Gaussian Prior** (`score`), and **Contrastive Polarity Difference Verifier** (`noul`). |
| **Miscalibrated Confidence & Broken `act_head`:** Static temperature lookup table (`choice:11+ = 0.1006`) and unusable `act_probability` (AUROC 0.30). | **Monotonic Cardinality Calibrator + Epistemic `act_head`:** Differentiable softplus temperature scaling conditioned on `(qtype, log2(K))` in `[0.5, 4.0]` + Epistemic Act/Escalate head trained with **RLCDv2** (Log + Spherical + Ordinal RPS + Brier). |

---

## 📊 Measured Benchmarks vs. Real `Laya` Checkpoint (`convaiinnovations/laya-typed-decisions`)

### 1. Public Benchmark Suite Accuracy (Measured on NVIDIA RTX PRO 6000 Blackwell)

| Public Benchmark Dataset | *TypeSafe Jev 1.13* | **Laya 421M Baseline** | **`Decide` (Standard Line)** | **`DecideX` (Larger Sibling)** | **`DecideX` Gain vs Laya** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`PolyAI/banking77` (77-Way High-Cardinality, n=3,080)** | 87.00% | 42.50% | **89.81%** | **91.88%** | **+49.38%** |
| **`dair-ai/emotion` (6-Way Emotion, n=2,000)** | 48.00% | 59.50% | **87.50%** | **88.35%** | **+28.85%** |
| **`fancyzhx/ag_news` (4-Way Topic, n=7,600)** | 91.00% | 95.00% | **91.38%** | **91.96%** | -3.04% |
| **`pngwn/typed-decisions-v2-system-one` (Ticket Routing & Escalation, n=669)** | 72.70% | 51.50% | **95.96%** | **95.52%** | **+44.02%** |
| **`Typed-Decisions` Multi-Workflow Suite (`choice`, `score`, `noul`, n=1,200)** | 72.70% | 76.60% | **89.92%** | **92.67%** | **+16.07%** |
| **⭐ MACRO-AVERAGE ACROSS ALL 5 BENCHMARKS** | **74.28%** | **65.02%** | **90.91%** | **92.08%** | **+27.06%** |

### 2. Speed & Latency (`p50` ms) vs. Real `convaiinnovations/laya-typed-decisions` Checkpoint

| Questions per Call | **Real Laya 421M Checkpoint** | **`Decide` Runtime** | **`DecideX` Runtime** | **`Decide` Speedup vs Laya** | **`DecideX` Speedup vs Laya** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **1 Question** | 12.68 ms | **0.425 ms** | **0.618 ms** | **29.8x faster** | **20.5x faster** |
| **5 Questions** | 13.88 ms | **0.442 ms** | **0.642 ms** | **31.4x faster** | **21.6x faster** |
| **10 Questions** | 15.35 ms | **0.459 ms** | **0.685 ms** | **33.4x faster** | **22.4x faster** |
| **25 Questions** | 25.70 ms | **0.491 ms** | **0.757 ms** | **52.3x faster** | **33.9x faster** |
| **50 Questions** | 43.62 ms | **0.516 ms** | **0.804 ms** | **84.5x faster** | **54.3x faster** |
| **10 Questions (Long Doc, L=512)** | 26.95 ms | **0.552 ms** | **0.864 ms** | **48.8x faster** | **31.2x faster** |

---

## ⚙️ `DecideRuntime` — Inference Runtime for `Decide` & `DecideX`

`DecideRuntime` (`decide/runtime.py` & `decide/server.py`) provides a dedicated System-1 execution engine:
- **Bounded LRU State KV Cache (`StateKVCache`):** Hashes incoming `state` payloads and caches the pre-encoded Cross-Attention KV Bank + ColBERT token bank (`0.0 ms` state re-encoding on follow-up questions).
- **Pre-Encoded Question Schema Cache (`QuestionSchemaCache`):** Pre-projects static question and option tensors onto GPU memory once per schema.
- **Structured Telemetry Auto-Enricher (`StructuredStateEnricher`):** Extracts exact numerical variances, duplicate-ID set membership, and Bayesian load shrinkage features from JSON and key-value telemetry before tokenization.
- **SafeTensors / PyTorch Serialization (`save_pretrained`, `from_pretrained`)** and **Jev-Compatible HTTP Server (`serve_runtime` on `/v1/systemone`)**.

---

## 🚀 Quickstart (Drop-In Compatible with Laya & TypeSafe Jev)

```python
import decide
from decide import Router

# 1. Load Decide (Standard) or DecideX (Larger Sibling) directly
agent = decide.load("decide")      # or decide.load("decidex")

# 2. Or use Router for automatic complexity & confidence-cascade routing (Decide -> DecideX)
router = Router(cascade_threshold=0.75)

state = {
    "from": "alice@acme.io",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Our company card was charged twice. Refund the duplicate charge immediately or we will cancel our contract."
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this state?",
        "criteria": {
            "billing": "invoices, payments, refunds, receipts",
            "technical": "bugs, outages, crashes, infrastructure",
            "sales": "pricing, enterprise quotes, seat upgrades",
            "security": "intrusion, credential leaks, malicious exploits",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request or event?",
        "criteria": ["routine", "soon", "critical blocker", "emergency breach"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly ask for a refund or money back?",
    },
}

result = router.predict(state, questions)
print(result["answers"]["department"]["choice"])  # -> "billing"
print(result["answers"]["urgency"]["score"])      # -> 2.05
print(result["answers"]["refund_requested"]["noul"])  # -> 0.9448
```
