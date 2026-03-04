from typing import Any, Dict, List
import numpy as np


def retrieve_vector(query: str, embed_model: Any, faiss_index: Any, id_order: List[str], content_map: Dict[str, str], metadata_map: Dict[str, Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
    if faiss_index is None or getattr(faiss_index, "ntotal", 0) <= 0:
        return []

    q_vec = np.array(embed_model.encode([query])).astype("float32")
    _, indices = faiss_index.search(q_vec, k=top_k)

    results = []
    for idx in indices[0]:
        if idx == -1 or idx >= len(id_order):
            continue
        doc_id = id_order[idx]
        content = content_map.get(doc_id)
        if not content:
            continue
        meta = metadata_map.get(doc_id, {})
        results.append(
            {
                "id": doc_id,
                "content": content,
                "metadata": meta,
                "score": 1.0,
                "source": meta.get("source", "本地文档"),
                "page": meta.get("page"),
            }
        )
    return results


def retrieve_bm25(query: str, bm25_manager: Any, top_k: int = 8, metadata_map: Dict[str, Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    if bm25_manager is None or getattr(bm25_manager, "bm25_index", None) is None:
        return []

    metadata_map = metadata_map or {}
    rows = bm25_manager.search(query, top_k=top_k)
    out = []
    for r in rows:
        meta = metadata_map.get(r["id"], {})
        out.append(
            {
                "id": r["id"],
                "content": r["content"],
                "metadata": meta,
                "score": float(r.get("score", 0.0)),
                "source": meta.get("source", "本地文档"),
                "page": meta.get("page"),
            }
        )
    return out


def merge_dedupe(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in primary + secondary:
        doc_id = item.get("id")
        if not doc_id:
            continue
        if doc_id not in merged or item.get("score", 0) > merged[doc_id].get("score", 0):
            merged[doc_id] = item

    # prefer higher score first
    items = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
    return items[:limit]


def optional_rerank(query: str, passages: List[Dict[str, Any]], rerank_fn: Any | None = None, top_k: int = 6) -> List[Dict[str, Any]]:
    if not passages:
        return []
    if rerank_fn is None:
        return passages[:top_k]

    docs = [p["content"] for p in passages]
    ids = [p["id"] for p in passages]
    metas = [p.get("metadata", {}) for p in passages]
    reranked = rerank_fn(query, docs, ids, metas, top_k=top_k)

    out = []
    for doc_id, payload in reranked:
        out.append(
            {
                "id": doc_id,
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
                "score": float(payload.get("score", 0.0)),
                "source": payload.get("metadata", {}).get("source", "本地文档"),
                "page": payload.get("metadata", {}).get("page"),
            }
        )
    return out


def evidence_check(passages: List[Dict[str, Any]], min_items: int = 2) -> Dict[str, Any]:
    evidence = []
    for p in passages:
        text = p.get("content", "")
        snippet = text[:220] + ("..." if len(text) > 220 else "")
        evidence.append(
            {
                "id": p.get("id"),
                "source": p.get("source", p.get("metadata", {}).get("source", "未知来源")),
                "page": p.get("page", p.get("metadata", {}).get("page")),
                "snippet": snippet,
            }
        )

    return {
        "supported": len(evidence) >= min_items,
        "evidence": evidence,
    }
