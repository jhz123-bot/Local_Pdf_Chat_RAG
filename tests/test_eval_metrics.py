from eval.metrics import schema_valid_rate, citation_rate, coverage_at_k, hallucination_proxy


def test_metrics_basic_rates():
    results = [
        {
            "task": "contributions",
            "items": [{"content": "framework better", "evidence": [{"text": "framework"}]}],
            "retrieval": {"mode": "hybrid", "top_k": 8, "reason": "x"},
            "supported": True,
        }
    ]
    gold = {"keywords": ["framework"]}
    assert schema_valid_rate(results) == 1.0
    assert citation_rate(results) == 1.0
    assert coverage_at_k(results, gold, k=3) == 1.0
    assert 0.0 <= hallucination_proxy(results) <= 1.0
