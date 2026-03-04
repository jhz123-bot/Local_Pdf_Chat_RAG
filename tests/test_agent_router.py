from agent.router_rules import choose_route


def test_choose_route_keyword_prefers_bm25():
    res = choose_route("这篇论文的作者和年份是什么？")
    assert res["route"] in {"bm25", "hybrid"}
    assert "reason" in res
    assert isinstance(res["top_k"], int)


def test_choose_route_semantic_prefers_vector_or_hybrid():
    res = choose_route("请比较方法A和方法B的优缺点")
    assert res["route"] in {"vector", "hybrid"}
    assert res["top_k"] >= 4
