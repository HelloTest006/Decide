"""
Automated Verification Suite for Decide & DecideX:
- Parameter count verification
- 50-trial random option permutation invariance test
- High-cardinality (100+ options) non-truncating test
- Shipped `.safetensors` weights loading and accuracy check
- Cold vs warm cache semantics in DecideRuntime
- Jev / Laya `/v1/systemone` HTTP server wire compatibility
"""
import json
import random
import threading
import urllib.request
from pathlib import Path

from decide import DecideConfig, DecideRuntime, load, serve_runtime
from benchmarks.reproduce_all import count_model_parameters, verify_permutation_invariance


def test_exact_parameter_counts():
    rt = DecideRuntime(device="cpu")
    p_d = count_model_parameters(rt.decide_model)
    p_x = count_model_parameters(rt.decidex_model)
    # Decide: vocab(32768*384=12,582,912) + pos(1024*384=393,216) + dense(4,548,875) = 17,525,003
    assert 17_000_000 < p_d["total"] < 18_000_000
    assert 4_000_000 < p_d["dense_active"] < 5_000_000
    # DecideX: vocab(32768*640=20,971,520) + pos(1024*640=655,360) + dense(19,515,789) = 41,142,669
    assert 40_000_000 < p_x["total"] < 42_000_000
    assert 19_000_000 < p_x["dense_active"] < 21_000_000


def test_option_permutation_invariance_50_shuffles():
    """Verify option probabilities are invariant to option order across 50 random permutations."""
    rt = DecideRuntime(device="cpu")
    matches, max_drift = verify_permutation_invariance(rt, n_permutations=50)
    assert matches == 50
    assert max_drift <= 1e-4


def test_high_cardinality_100_options_no_truncation():
    agent = load("decidex", device="cpu")
    many_options = {f"option_{i:03d}": f"Candidate routing destination {i}" for i in range(100)}
    res = agent.predict(
        "Route request to destination 42",
        {"route": {"type": "choice", "instructions": "Pick destination", "criteria": many_options}},
    )
    ans = res["answers"]["route"]
    assert ans["type"] == "choice"
    assert len(ans["probabilities"]) == 100
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-2


def test_shipped_weights_predictions():
    """Verify the shipped trained weights in `weights/` produce high-confidence accurate predictions."""
    rt = DecideRuntime(device="cpu")
    state = (
        "Customer alice@acme.io writes: Our company card was charged twice on invoice #4411 "
        "for the annual subscription. Refund the duplicate charge immediately or we will cancel our contract."
    )
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
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the user explicitly ask for a refund or money back?",
        },
    }
    res_d = rt.predict(state, questions, model="decide")
    res_x = rt.predict(state, questions, model="decidex")
    assert res_d["answers"]["department"]["choice"] == "billing"
    assert res_x["answers"]["department"]["choice"] == "billing"
    assert res_d["answers"]["refund_requested"]["noul"] > 0.80
    assert res_x["answers"]["refund_requested"]["noul"] > 0.80


def test_runtime_cold_vs_warm_cache_accounting():
    rt = DecideRuntime(device="cpu", cache_states=True, cache_questions=True)
    state = "Billing alert: duplicate charge on statement #3017."
    q = {"refund": {"type": "noul", "instructions": "Does the user request a refund?"}}
    r1 = rt.predict(state, q, model="decide")
    assert r1["runtime"]["state_cache_hit"] is False
    assert r1["runtime"]["schema_cache_hit"] is False
    assert r1["usage"]["input_tokens"] > 0

    r2 = rt.predict(state, q, model="decide")
    assert r2["runtime"]["state_cache_hit"] is True
    assert r2["runtime"]["schema_cache_hit"] is True
    assert r2["usage"]["input_tokens"] == 0


def test_http_systemone_server():
    rt = DecideRuntime(device="cpu")
    server = serve_runtime(rt, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {
            "model": "decide",
            "state": "Customer charged twice on invoice #4411, requesting immediate refund.",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle this state?",
                    "criteria": {
                        "billing": "invoices, payments, refunds, receipts",
                        "technical": "bugs, outages, crashes, infrastructure",
                    },
                }
            },
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            body = json.loads(resp.read().decode("utf-8"))
            assert body["answers"]["department"]["choice"] == "billing"
    finally:
        server.shutdown()
        server.server_close()
