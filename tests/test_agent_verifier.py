from agent.verifier import verify_with_fallback


def test_verify_with_fallback_supported_first_round():
    def retrieve_once(_q, _k):
        return [
            {"id": "1", "content": "evidence a", "source": "a.pdf", "page": 1},
            {"id": "2", "content": "evidence b", "source": "a.pdf", "page": 2},
        ]

    out = verify_with_fallback("q", retrieve_once, first_top_k=4, fallback_top_k=8, max_rounds=2)
    assert out["supported"] is True
    assert out["rounds"] == 1
    assert len(out["evidence"]) >= 2


def test_verify_with_fallback_unsupported_after_two_rounds():
    def retrieve_once(_q, _k):
        return [{"id": "1", "content": "only one evidence", "source": "a.pdf", "page": 1}]

    out = verify_with_fallback("q", retrieve_once, first_top_k=3, fallback_top_k=6, max_rounds=2)
    assert out["supported"] is False
    assert out["rounds"] == 2
    assert len(out["retrieval_log"]) == 2
