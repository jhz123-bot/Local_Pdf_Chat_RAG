from __future__ import annotations

import re
from typing import Any, Dict, List


REQUIRED_SECTIONS = {"problem", "method", "contributions", "experiments", "limitations"}


def _has_evidence(item: Dict[str, Any]) -> bool:
    ev = item.get("evidence", [])
    return isinstance(ev, list) and len(ev) > 0


def validate_schema(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    for key in ["task", "items", "retrieval", "supported"]:
        if key not in result:
            return False
    if not isinstance(result["items"], list):
        return False

    task = result.get("task")
    if task == "paper_summary":
        sections = {str(i.get("section")) for i in result["items"]}
        return REQUIRED_SECTIONS.issubset(sections)
    return True


def schema_valid_rate(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    ok = sum(1 for r in results if validate_schema(r))
    return ok / len(results)


def citation_rate(results: List[Dict[str, Any]]) -> float:
    total_items = 0
    cited_items = 0
    for r in results:
        for item in r.get("items", []):
            total_items += 1
            if _has_evidence(item):
                cited_items += 1
    if total_items == 0:
        return 0.0
    return cited_items / total_items


def coverage_at_k(results: List[Dict[str, Any]], gold: Dict[str, Any], k: int = 5) -> float:
    """How often evidence text covers gold keywords in top-k items."""
    if not results:
        return 0.0

    kw = [x.lower() for x in gold.get("keywords", [])]
    if not kw:
        return 0.0

    covered = 0
    for r in results:
        texts = []
        for item in r.get("items", [])[:k]:
            for ev in item.get("evidence", []):
                texts.append(str(ev.get("text", "")).lower())
        blob = "\n".join(texts)
        if any(kword in blob for kword in kw):
            covered += 1
    return covered / len(results)


def hallucination_proxy(results: List[Dict[str, Any]]) -> float:
    """Proxy: ratio of content tokens not supported by evidence snippets (lower is better)."""
    ratios = []
    token_re = re.compile(r"[\w\u4e00-\u9fff]+")

    for r in results:
        for item in r.get("items", []):
            content = str(item.get("content", ""))
            evidence_blob = " ".join(str(e.get("text", "")) for e in item.get("evidence", []))
            content_tokens = set(token_re.findall(content.lower()))
            evidence_tokens = set(token_re.findall(evidence_blob.lower()))
            if not content_tokens:
                continue
            unsupported = [t for t in content_tokens if t not in evidence_tokens]
            ratios.append(len(unsupported) / max(1, len(content_tokens)))

    if not ratios:
        return 1.0
    return sum(ratios) / len(ratios)
