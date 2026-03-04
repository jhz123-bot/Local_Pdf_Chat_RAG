from typing import Dict


def choose_route(question: str, kb_ready: bool = True) -> Dict[str, object]:
    """Route retrieval strategy and top_k with a short reason."""
    q = (question or "").strip()
    q_lower = q.lower()

    keyword_hints = ["定义", "术语", "公式", "等于", "缩写", "作者", "年份"]
    semantic_hints = ["比较", "优缺点", "为什么", "影响", "关系", "原理", "总结"]

    if not kb_ready:
        return {
            "route": "bm25",
            "top_k": 4,
            "reason": "知识库未就绪，优先使用关键词检索路径并降低召回规模",
        }

    has_keyword_hint = any(h in q for h in keyword_hints)
    has_semantic_hint = any(h in q for h in semantic_hints)

    if has_keyword_hint and not has_semantic_hint:
        return {
            "route": "bm25",
            "top_k": 6 if len(q) > 12 else 4,
            "reason": "问题更偏精确关键词匹配，优先BM25",
        }

    if has_semantic_hint and not has_keyword_hint:
        return {
            "route": "vector",
            "top_k": 8,
            "reason": "问题更偏语义理解，优先向量检索",
        }

    # 默认混合路由
    dynamic_top_k = 10 if len(q_lower) > 18 or "?" in q_lower or "？" in q_lower else 7
    return {
        "route": "hybrid",
        "top_k": dynamic_top_k,
        "reason": "问题包含多种意图，采用混合检索提高召回稳定性",
    }
