from typing import Any, Dict, List, Callable
from .tools import evidence_check


def verify_with_fallback(
    question: str,
    retrieve_once: Callable[[str, int], List[Dict[str, Any]]],
    first_top_k: int,
    fallback_top_k: int,
    max_rounds: int = 2,
) -> Dict[str, Any]:
    """Run evidence verification and fallback retrieval once if needed."""
    rounds = 0
    passages: List[Dict[str, Any]] = []
    retrieval_log: List[Dict[str, Any]] = []

    while rounds < max_rounds:
        rounds += 1
        top_k = first_top_k if rounds == 1 else fallback_top_k
        passages = retrieve_once(question, top_k)
        check = evidence_check(passages)
        retrieval_log.append({"round": rounds, "top_k": top_k, "supported": check["supported"], "count": len(passages)})
        if check["supported"]:
            return {
                "passages": passages,
                "evidence": check["evidence"],
                "supported": True,
                "rounds": rounds,
                "retrieval_log": retrieval_log,
            }

    check = evidence_check(passages)
    return {
        "passages": passages,
        "evidence": check["evidence"],
        "supported": False,
        "rounds": rounds,
        "retrieval_log": retrieval_log,
    }
