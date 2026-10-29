# Decide & DecideX — Shared-State System-1 Decision Models & Runtime

`Decide` (Standard Line) and `DecideX` (Larger Sibling Line) are non-autoregressive System-1 decision models and an inference runtime (`DecideRuntime`) for answering typed questions (`choice`, `score`, `noul`) over text or JSON states in a single forward pass.

The repository includes:
- **`decide/`**: Model architecture (`model.py`), inference runtime (`runtime.py`), `/v1/systemone` HTTP server (`server.py`), router (`router.py`), and RLCDv2 training recipe (`training.py`).
- **`weights/`**: Trained FP16 checkpoints (`weights/decide/model.safetensors`, `33.35 MB`; `weights/decidex/model.safetensors`, `78.49 MB`).
- **`benchmarks/reproduce_all.py`** & **`tests/test_decide_and_runtime.py`**: Reproducible evaluation, latency profiling, and option permutation-invariance tests.

---

## 1. Architecture & Parameter Breakdown

In [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya), each question concatenates `[CLS] question + options [SEP] state [SEP]` and runs a 28-layer `ModernBERT-large` encoder (`403.3M` parameters), re-encoding the same `state` $Q$ times when $Q$ questions are asked and sharing a fixed `head_max_len` token budget (`192`–`256` tokens) sequentially across all options.

`Decide` and `DecideX` decouple state encoding from option scoring:
1. **Single-Pass Shared State KV Bank (`encode_state`):** Encodes `state` once into a Cross-Attention KV Bank and an L2-normalized ColBERT late-interaction token bank. All $Q$ questions cross-attend to this shared state representation.
2. **Parallel Isolated Option Set-Pooling:** Each option is encoded in its own isolated row `[Q, K, L_opt, D]` without sequential packing or positional encoding across options. Reordering options produces identical option probabilities up to floating-point reduction order (verified across 50 random shuffles of 24 options in `tests/test_decide_and_runtime.py::test_option_permutation_invariance_50_shuffles`).
3. **Tri-Primitive Scoring & Calibration:**
   - `choice`: Multiplicative Hadamard state-option alignment (`opt_raw * s_ctx`) + Gated Cross-Attention + token-level MaxSim.
   - `score`: Cumulative ordinal Gaussian prior (`ordinal_sharpness`) around a learned scalar rank projection.
   - `noul`: Contrastive polarity difference projection (`opt_raw[:, 1] - opt_raw[:, 0]`) conditioned on `q_pooled * s_pooled`.
   - `calibrator`: Monotonic temperature scaling `softplus(base + slope * log2(K)) + 0.5` clamped to `[0.5, 4.0]`.

| Model Line | Checkpoint | Width (`d_model`) | Layers (State / Option / Cross) | Total Params | Embedding Params (`tok_emb` + `pos_emb`) | Dense Active Params | FP16 Weight File |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| **`Decide`** (Standard) | `weights/decide/` | 384 | 2 / 1 / 1 | **17,485,459** (17.49M) | 12,976,128 (12.98M) | **4,509,331** (4.51M) | 33.35 MB |
| **`DecideX`** (Large) | `weights/decidex/` | 640 | 3 / 1 / 2 | **41,148,820** (41.15M) | 21,626,880 (21.63M) | **19,521,940** (19.52M) | 78.49 MB |
| *Laya Reference* | `convaiinnovations/laya` | 1024 | 28 + 2 Head | ~421,000,000 (421M) | ~51,500,000 (51.5M) | ~369,500,000 (369.5M) | ~808 MB |

---

## 2. Accuracy Evaluation & Train/Test Protocol Disclosure

**Important Methodological Distinction (Fine-Tuned vs. Zero-Shot):**
Unlike `Laya` (`ModernBERT-large`, pretrained on 2 trillion tokens), `Decide` and `DecideX` use a lightweight deterministic hash tokenizer (`FastDecideTokenizer`) trained from scratch without language-model pretraining. Consequently:
- **Table 2A** evaluates the **single shipped checkpoints** (`weights/decide` and `weights/decidex`) on the held-out validation split of their training workflow suite (`1,200` decisions) and on an **unseen-schema zero-shot transfer test** (`300` decisions with unseen question IDs, option keys, and prompt phrasing).
- **Table 2B** evaluates **task-adapted fine-tuning** on public HuggingFace benchmarks (`PolyAI/banking77`, `dair-ai/emotion`, `fancyzhx/ag_news`, `pngwn/typed-decisions-v2-system-one`), where `Decide`, `DecideX`, and a **matched `ModernBERT-large + Classification Head` reference** are all trained on each dataset's official `train` split and evaluated on its `test` split. Published `Laya` and `TypeSafe Jev` numbers on those datasets are **zero-shot** and are shown in separate columns for context, not as an apples-to-apples supervised comparison.

### Table 2A: Shipped Checkpoints — Multi-Workflow Validation & Unseen-Schema Zero-Shot Transfer

| Evaluation Split | Regime | **Laya 421M (`laya-typed-decisions`)** | **`Decide` (`weights/decide`)** | **`DecideX` (`weights/decidex`)** |
| :--- | :--- | ---: | ---: | ---: |
| **Multi-Workflow Validation Suite** (`n=1,200` decisions: `choice`, `score`, `noul`) | Held-out test split of training distribution | 76.60% *(official benchmark)* | **89.75%** (ECE `0.0456`) | **92.58%** (ECE `0.0478`) |
| — `choice` primitive accuracy (4 to 24 options) | Held-out test split | 73.30% | **87.54%** | **93.43%** |
| — `score` primitive accuracy (4-level ordinal rubric) | Held-out test split | 72.30% | **91.51%** | **91.51%** |
| — `noul` primitive accuracy (boolean proposition) | Held-out test split | 85.70% | **92.13%** | **91.88%** |
| **Unseen-Schema Cloud/FinOps Transfer** (`n=300` decisions, unseen option keys & instructions) | **Zero-Shot** (no training on schema) | 91.67% (`275/300`) | **75.67%** (`227/300`) | **98.33%** (`295/300`) |

### Table 2B: Public HuggingFace Benchmarks — Supervised Train-Split Adaptation vs. Zero-Shot Baselines

*In the three right-hand columns, `ModernBERT-large + Clf Head`, `Decide`, and `DecideX` are trained on each dataset's official `train` split (`benchmarks/real_benchmarks.py`). In the two left-hand baseline columns, `TypeSafe Jev` and `Laya 421M` are evaluated zero-shot without task-split fine-tuning.*

| Public HuggingFace Dataset | *TypeSafe Jev (Zero-Shot Published)* | *Laya 421M (Zero-Shot Baseline)* | **`ModernBERT-large + Clf Head` (Fine-Tuned Ref)** | **`Decide` (Fine-Tuned on Train Split)** | **`DecideX` (Fine-Tuned on Train Split)** |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **`PolyAI/banking77`** (77 options, `n_train=10,003`, `n_test=3,080`) | 87.00% | 42.50%¹ | 84.35% | **89.81%** | **91.88%** |
| **`dair-ai/emotion`** (6 options, `n_train=16,000`, `n_test=2,000`) | 48.00%² | 59.50% | 67.35% | **87.50%** | **88.35%** |
| **`fancyzhx/ag_news`** (4 options, `n_train=32,000`, `n_test=7,600`) | 91.00% | **95.00%** | 87.70% | **87.46%** *(91.38% full 120k)* | **88.38%** *(91.96% full 120k)* |
| **`pngwn/typed-decisions-v2-system-one`** (`team` + `escalate`, `n_test=669`) | — *(not published on v2)* | 51.50%³ | 95.52% | **95.96%** | **95.52%** |

**Notes on baseline fairness:**
1. **`Laya` on `Banking77` (42.50%):** At Laya's default `head_max_len=192` (`256` on `laya-typed-decisions`), packing 77 options sequentially allocates `~3` tokens per option label, truncating option descriptions. Testing `laya-typed-decisions` zero-shot on `Banking77` (`n=200`) with an expanded budget (`head_max_len=640, max_len=1024`) yielded `35.00%` vs `38.50%` at default (`256`), because the checkpoint was trained at `head_max_len=256` and degrades when option positions shift past its training range. Adding a standard classification head on `ModernBERT-large` embeddings trained on the `Banking77` train split reaches `84.35%`, while `Decide` and `DecideX` (with unigram+bigram+subword stem pooling and MaxSim) reach `89.81%` and `91.88%`.
2. **`Jev` on `DAIR Emotion` (48.00%):** Reported in Laya's benchmark table; Jev's zero-shot prompt assigned zero probability to the ground-truth emotion label on 16% of examples.
3. **`Laya` on `pngwn/typed-decisions-v2-system-one` (51.50%):** Measured live using `convaiinnovations/laya-typed-decisions` zero-shot on the `workflow4` test split (`n=200`), where tickets contain numeric telemetry (`Load reading`, `Signal quality`) whose Bayesian posterior threshold requires task supervision.

---

## 3. Latency & Speedup Breakdown (Cold vs. Cached vs. Forward-Only)

All latencies below were measured on an **NVIDIA RTX PRO 6000 Blackwell** (`torch 2.11.0+cu130`, **batch size = 1 state**, $Q \in \{1, 5, 10, 25, 50\}$ questions per call, **BF16 autocast**, synchronized with `torch.cuda.synchronize()`, median `p50` over 25 iterations after warmup).

To separate caching effects from raw model execution speed, we report **four distinct execution modes**:
1. **Cold Uncached End-to-End (`cache_states=False, cache_questions=False`):** Includes Python string tokenization of `state`, `questions`, and `options`, tensor creation/host-to-device copy, `encode_state()`, `forward_questions()`, and Python output dictionary formatting.
2. **Fresh-State + Cached Question Schema (`cache_states=False, cache_questions=True`):** The `state` is new and uncached on every call (tokenized and encoded from scratch), while the static `questions` schema is pre-encoded on GPU. This matches standard production triage/guardrail services where every request brings a new document against a fixed rubric.
3. **Warm State + Schema Cache (`cache_states=True, cache_questions=True`):** Both `state` KV bank and `questions` schema are cached on GPU (applicable when asking follow-up questions about an already-encoded state).
4. **CUDA-Graph Forward-Only (`torch.cuda.CUDAGraph`):** Pure GPU kernel execution time of `encode_state() + forward_questions()` on pre-allocated device tensors, excluding Python tokenization and dict formatting.

| Questions ($Q$) | **Real Laya 421M End-to-End** | **`Decide` Cold Uncached E2E** *(Speedup)* | **`DecideX` Cold Uncached E2E** *(Speedup)* | **`Decide` Fresh-State (Schema Cached)** *(Speedup)* | **`DecideX` Fresh-State (Schema Cached)** *(Speedup)* | **`Decide` Warm State+Schema Cache** | **`Decide` CUDA-Graph Forward** *(Speedup)* | **`DecideX` CUDA-Graph Forward** *(Speedup)* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1** | 13.10 ms | **2.84 ms** *(4.6x)* | **3.27 ms** *(4.0x)* | **2.02 ms** *(6.5x)* | **2.44 ms** *(5.4x)* | 1.07 ms | **0.425 ms** *(30.8x)* | **0.618 ms** *(21.2x)* |
| **5** | 13.93 ms | **3.13 ms** *(4.4x)* | **3.56 ms** *(3.9x)* | **2.10 ms** *(6.6x)* | **2.53 ms** *(5.5x)* | 1.15 ms | **0.442 ms** *(31.5x)* | **0.642 ms** *(21.7x)* |
| **10** | 15.01 ms | **3.44 ms** *(4.4x)* | **3.86 ms** *(3.9x)* | **2.15 ms** *(7.0x)* | **2.57 ms** *(5.8x)* | 1.19 ms | **0.459 ms** *(32.7x)* | **0.685 ms** *(21.9x)* |
| **25** | 22.50 ms | **4.42 ms** *(5.1x)* | **4.87 ms** *(4.6x)* | **2.35 ms** *(9.6x)* | **2.80 ms** *(8.0x)* | 1.38 ms | **0.491 ms** *(45.8x)* | **0.757 ms** *(29.7x)* |
| **50** | 39.01 ms | **5.90 ms** *(6.6x)* | **6.34 ms** *(6.1x)* | **2.63 ms** *(14.8x)* | **3.08 ms** *(12.6x)* | 1.63 ms | **0.516 ms** *(75.6x)* | **0.804 ms** *(48.5x)* |

---

## 4. Limitations & Where `Decide` / `DecideX` Lose

1. **No Pretrained Language Model Backbone (Zero-Shot Vocabulary Gap):**
   Because `Decide` (`17.5M`) and `DecideX` (`41.1M`) use hash-based token embeddings trained from scratch rather than a pretrained encoder like `ModernBERT-large` (`421M`), **they do not support open-domain zero-shot classification on arbitrary unseen vocabularies**. On `AG News` (`fancyzhx/ag_news`), pretrained `Laya` achieves `95.00%` zero-shot, whereas `Decide` and `DecideX` trained from scratch on `32k` articles reach `87.46%` and `88.38%` (`-6.62` points vs. `Laya`), and `91.38%` / `91.96%` when trained on the full `120k` split (`-3.04` points vs. `Laya`).
2. **Short-Window Truncation in `Decide` (Standard Line):**
   `Decide` is trained with a fast 29-token state window (`fast_window_tokens=29` in `decide/training.py`). On long multi-turn states where an early resolved clause is overridden by a late paragraph, `Decide` misses the late override (`75.33%` on unseen-schema transfer vs. `98.33%` for `DecideX`). Use `DecideRouter(cascade_threshold=0.75)` or `model="decidex"` when states exceed ~30 words or contain multi-turn discourse.
3. **Python Overhead Dominates Uncached Small-Batch Calls:**
   While the GPU forward pass of `Decide` takes `0.425 ms` (30.8x faster than `Laya`), uncached Python tokenization and dictionary construction add `~1.6–2.4 ms` per call, bringing cold uncached end-to-end latency to `2.84 ms` (4.6x faster than `Laya` at $Q=1$, scaling to `14.8x` faster at $Q=50$ when the question schema is cached).

---

## 5. Training Recipe & Reproduction

### Training Recipe (`decide/training.py`)
The shipped checkpoints (`weights/decide/model.safetensors` and `weights/decidex/model.safetensors`) were trained using `train_decide_line()` in `decide/training.py`:
- **Objective (`rlcd_v2_loss`):** Strictly proper scoring rule combining soft-target Cross-Entropy (Log Score) + `0.5 *` Spherical Score + `0.4 *` Brier Score + `1.2 *` Ranked Probability Score (RPS, applied to ordinal `score` questions) + `0.25 *` Cross-Entropy on `act_head` (predicting whether `argmax` matches the target label).
- **Optimizer & Schedule:** `AdamW` (`weight_decay=0.01`, gradient clipping `max_norm=1.0`) with `CosineAnnealingLR` over `8` epochs (`batch_size=48` states $\times 3$ questions $= 144$ decisions/step; `lr=1.6e-3` for `Decide`, `lr=1.3e-3` for `DecideX`).
- **Precision:** Trained in `BF16` mixed precision (`torch.autocast`) and exported to `FP16` `.safetensors`.

### Reproduce All Benchmarks & Tests

```bash
pip install -e .
pip install pytest

# 1. Run automated unit, permutation-invariance, and HTTP server tests
python -m pytest -v

# 2. Reproduce parameter counts, 50-shuffle permutation invariance, accuracy, and cold/warm latency tables
python -m benchmarks.reproduce_all

# 3. Reproduce public HuggingFace dataset fine-tuning (requires `pip install datasets`)
python -m benchmarks.real_benchmarks
```

---

## 6. Quickstart (`DecideRuntime` & HTTP Server)

```python
from decide import DecideRuntime

rt = DecideRuntime()  # automatically loads weights/decide and weights/decidex

state = {
    "from": "alice@acme.io",
    "subject": "Duplicate charge on invoice #4411",
    "body": "Our company card was charged twice. Refund the duplicate charge immediately or we will cancel our contract.",
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
        "criteria": [
            "level 0: routine or not urgent",
            "level 1: moderately urgent, handle soon",
            "level 2: critical production or customer blocker",
            "level 3: emergency security breach or destructive threat",
        ],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly ask for a refund or money back?",
    },
}

result = rt.predict(state, questions, model="auto")
print(result["answers"]["department"]["choice"])      # -> "billing"
print(result["answers"]["refund_requested"]["noul"])  # -> 0.9469
print(result["runtime"])                              # -> latency_ms, state_cache_hit, schema_cache_hit
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
