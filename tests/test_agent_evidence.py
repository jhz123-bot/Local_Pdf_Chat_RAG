from agent.tools import evidence_check


def test_evidence_check_schema():
    passages = [
        {"id": "a", "content": "first evidence content", "source": "paper1.pdf", "page": 1},
        {"id": "b", "content": "second evidence content", "source": "paper2.pdf", "page": 3},
    ]
    out = evidence_check(passages)
    assert set(out.keys()) == {"supported", "evidence"}
    assert out["supported"] is True
    assert len(out["evidence"]) == 2
    assert {"id", "source", "page", "snippet"}.issubset(out["evidence"][0].keys())
