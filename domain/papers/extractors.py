from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.router_rules import choose_route
from agent.tools import (
    evidence_check,
    merge_dedupe,
    optional_rerank,
    retrieve_bm25,
    retrieve_vector,
)

from .prompts import (
    CONTRIBUTIONS_PROMPT,
    EXPERIMENT_SETUP_PROMPT,
    METHOD_PIPELINE_PROMPT,
    PAPER_SUMMARY_PROMPT,
)


TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "contributions": {
        "sections": ["abstract", "introduction", "conclusion", "discussion"],
        "keywords": [
            "contribution", "we propose", "we present", "our main contributions",
            "本文贡献", "本文提出", "主要工作",
        ],
        "prompt": CONTRIBUTIONS_PROMPT,
        "query_hint": "main contributions, claimed novelty, 本文贡献 主要工作",
    },
    "method_pipeline": {
        "sections": ["method", "approach", "proposed method", "framework", "model", "methods", "methodology"],
        "keywords": [
            "framework", "architecture", "pipeline", "module", "stage", "algorithm",
            "方法", "模型", "框架", "流程", "模块", "结构",
        ],
        "prompt": METHOD_PIPELINE_PROMPT,
        "query_hint": "method pipeline modules framework architecture 方法 流程 模块",
    },
    "experiment_setup": {
        "sections": ["experiments", "experimental setup", "evaluation", "results"],
        "keywords": [
            "dataset", "benchmark", "baseline", "evaluation", "metric", "training",
            "数据集", "基线", "指标", "实验设置", "评估", "结果",
        ],
        "prompt": EXPERIMENT_SETUP_PROMPT,
        "query_hint": "experiment setup datasets baselines metrics training 实验设置 数据集 基线 指标",
    },
}

SUMMARY_SEGMENTS = [
    {
        "section": "problem",
        "sections": ["abstract", "introduction"],
        "keywords": ["problem", "motivation", "challenge", "研究问题", "动机", "挑战"],
        "query": "problem motivation challenge 研究问题 动机 挑战",
    },
    {
        "section": "method",
        "sections": ["method", "approach"],
        "keywords": ["method", "framework", "model", "proposed", "方法", "框架", "模型"],
        "query": "method framework model proposed 方法 框架 模型",
    },
    {
        "section": "contributions",
        "sections": ["abstract", "introduction", "conclusion"],
        "keywords": TASK_CONFIGS["contributions"]["keywords"],
        "query": "main contributions novelty 本文贡献 主要工作",
    },
    {
        "section": "experiments",
        "sections": ["experiments", "results"],
        "keywords": TASK_CONFIGS["experiment_setup"]["keywords"],
        "query": "experiments results datasets metrics 实验 结果 数据集 指标",
    },
    {
        "section": "limitations",
        "sections": ["discussion", "conclusion"],
        "keywords": ["limitation", "future work", "局限", "展望", "未来工作"],
        "query": "limitations future work 局限 未来工作",
    },
]


@dataclass
class PaperExtractionService:
    embed_model: Any
    faiss_index: Any
    id_order: List[str]
    content_map: Dict[str, str]
    metadata_map: Dict[str, Dict[str, Any]]
    bm25_manager: Any
    rerank_fn: Optional[Callable[..., Any]]
    llm_callable: Callable[[str], str]

    def _kb_ready(self) -> bool:
        return bool(self.faiss_index is not None and getattr(self.faiss_index, "ntotal", 0) > 0)

    def _score_item(self, item: Dict[str, Any], section_targets: List[str], keyword_targets: List[str]) -> float:
        score = float(item.get("score", 0.0))
        metadata = item.get("metadata", {}) or {}
        section_title = str(metadata.get("section_title", "")).lower()
        if section_title:
            for sec in section_targets:
                if sec in section_title:
                    score += 2.0
                    break

        content_lower = str(item.get("content", "")).lower()
        hits = 0
        for kw in keyword_targets:
            if kw.lower() in content_lower:
                hits += 1
        score += min(1.5, 0.2 * hits)
        return score

    def _retrieve_candidates(self, query: str, route: str, top_k: int) -> List[Dict[str, Any]]:
        vector_rows: List[Dict[str, Any]] = []
        bm25_rows: List[Dict[str, Any]] = []

        if route in {"vector", "hybrid"}:
            vector_rows = retrieve_vector(
                query,
                self.embed_model,
                self.faiss_index,
                self.id_order,
                self.content_map,
                self.metadata_map,
                top_k=top_k,
            )

        if route in {"bm25", "hybrid"}:
            bm25_rows = retrieve_bm25(
                query,
                self.bm25_manager,
                top_k=top_k,
                metadata_map=self.metadata_map,
            )

        merged = merge_dedupe(vector_rows, bm25_rows, limit=max(top_k, 10))
        if self.rerank_fn:
            merged = optional_rerank(query, merged, self.rerank_fn, top_k=min(max(top_k, 6), 10))
        return merged

    def _section_aware_select(
        self,
        passages: List[Dict[str, Any]],
        section_targets: List[str],
        keyword_targets: List[str],
        context_limit: int,
    ) -> List[Dict[str, Any]]:
        rescored = []
        for p in passages:
            score = self._score_item(p, section_targets, keyword_targets)
            item = dict(p)
            item["_boosted_score"] = score
            rescored.append(item)

        rescored.sort(key=lambda x: x.get("_boosted_score", 0), reverse=True)
        return rescored[:context_limit]

    def _build_context(self, passages: List[Dict[str, Any]]) -> str:
        chunks = []
        for idx, p in enumerate(passages, 1):
            meta = p.get("metadata", {}) or {}
            source = meta.get("source", p.get("source", "未知来源"))
            page = meta.get("page", p.get("page"))
            section = meta.get("section_title", "")
            chunks.append(
                f"[Passage {idx}] source={source}; page={page}; section={section}\n{p.get('content', '')}"
            )
        return "\n\n".join(chunks)

    def _parse_json(self, text: str, default_task: str) -> Dict[str, Any]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?", "", raw).strip()
            raw = re.sub(r"```$", "", raw).strip()

        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {"task": default_task, "items": []}

    def _resolve_evidence(self, evidence_field: Any, passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(evidence_field, list):
            candidates = [str(x).strip() for x in evidence_field if str(x).strip()]
        elif isinstance(evidence_field, str):
            candidates = [evidence_field.strip()] if evidence_field.strip() else []
        else:
            candidates = []

        mapped: List[Dict[str, Any]] = []
        for candidate in candidates:
            matched = None
            c_low = candidate.lower()
            for p in passages:
                content = str(p.get("content", ""))
                if c_low and c_low[:40] in content.lower():
                    matched = p
                    break
            if matched is None and passages:
                matched = passages[0]

            if matched:
                meta = matched.get("metadata", {}) or {}
                mapped.append(
                    {
                        "text": candidate or str(matched.get("content", ""))[:180],
                        "source": meta.get("source", matched.get("source", "未知来源")),
                        "page": meta.get("page", matched.get("page")),
                    }
                )
        return mapped

    def _normalize_items(self, task: str, items: List[Dict[str, Any]], passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for idx, item in enumerate(items):
            row = dict(item)
            ev = self._resolve_evidence(row.get("evidence"), passages)
            if task == "method_pipeline":
                row["step"] = int(row.get("step", idx + 1))
            if task == "experiment_setup":
                category = str(row.get("category", "")).lower()
                if category not in {"dataset", "baseline", "metric", "training"}:
                    # best-effort map
                    if "data" in category or "数据" in category:
                        category = "dataset"
                    elif "base" in category or "基线" in category:
                        category = "baseline"
                    elif "metric" in category or "指标" in category:
                        category = "metric"
                    else:
                        category = "training"
                row["category"] = category
            row["evidence"] = ev
            normalized.append(row)

        if task == "method_pipeline":
            normalized = sorted(normalized, key=lambda x: x.get("step", 0))
            for i, row in enumerate(normalized, 1):
                row["step"] = i
        return normalized

    def _finalize(
        self,
        task: str,
        items: List[Dict[str, Any]],
        routing: Dict[str, Any],
        supported: bool,
        answer: str = "",
    ) -> Dict[str, Any]:
        return {
            "task": task,
            "answer": answer,
            "items": items,
            "retrieval": {
                "mode": routing.get("route", "hybrid"),
                "top_k": int(routing.get("top_k", 8)),
                "reason": routing.get("reason", ""),
            },
            "supported": supported,
        }

    def _run_single_task(self, task: str, question: str) -> Dict[str, Any]:
        cfg = TASK_CONFIGS[task]
        query = f"{question}\n{cfg['query_hint']}"
        routing = choose_route(query, kb_ready=self._kb_ready())

        passages = self._retrieve_candidates(query, route=routing["route"], top_k=int(routing["top_k"]))
        passages = self._section_aware_select(passages, cfg["sections"], cfg["keywords"], context_limit=10)

        check = evidence_check(passages)
        if not check["supported"]:
            # fallback once to hybrid
            fallback = self._retrieve_candidates(query, route="hybrid", top_k=min(int(routing["top_k"]) + 3, 14))
            passages = self._section_aware_select(fallback, cfg["sections"], cfg["keywords"], context_limit=10)
            check = evidence_check(passages)

        if not check["supported"]:
            return self._finalize(
                task,
                [],
                routing,
                supported=False,
                answer="文档证据不足，无法确认。",
            )

        selected = passages[:10] if len(passages) <= 10 else passages[:10]
        if len(selected) > 10:
            selected = selected[:10]
        context = self._build_context(selected)
        prompt = f"{cfg['prompt']}\n\nContext Passages:\n{context}"
        model_out = self.llm_callable(prompt)
        parsed = self._parse_json(model_out, task)
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        items = self._normalize_items(task, items, selected)
        return self._finalize(task, items, routing, supported=True)

    def extract_contributions(self, question: str) -> Dict[str, Any]:
        return self._run_single_task("contributions", question)

    def extract_method_pipeline(self, question: str) -> Dict[str, Any]:
        return self._run_single_task("method_pipeline", question)

    def extract_experiment_setup(self, question: str) -> Dict[str, Any]:
        return self._run_single_task("experiment_setup", question)

    def generate_paper_summary(self, question: str) -> Dict[str, Any]:
        routing = choose_route(question, kb_ready=self._kb_ready())

        all_passages: List[Dict[str, Any]] = []
        for seg in SUMMARY_SEGMENTS:
            seg_query = f"{question}\n{seg['query']}"
            seg_rows = self._retrieve_candidates(seg_query, route=routing["route"], top_k=max(4, int(routing["top_k"]) - 1))
            seg_rows = self._section_aware_select(seg_rows, seg["sections"], seg["keywords"], context_limit=3)
            all_passages.extend(seg_rows)

        merged = merge_dedupe(all_passages, [], limit=10)
        check = evidence_check(merged)
        if not check["supported"]:
            fallback_rows = self._retrieve_candidates(question, route="hybrid", top_k=min(int(routing["top_k"]) + 4, 14))
            merged = self._section_aware_select(fallback_rows, ["abstract", "introduction", "method", "experiments", "conclusion", "discussion"], [], context_limit=10)
            check = evidence_check(merged)

        if not check["supported"]:
            items = [
                {"section": "problem", "content": "Not explicitly stated in the provided text.", "evidence": []},
                {"section": "method", "content": "Not explicitly stated in the provided text.", "evidence": []},
                {"section": "contributions", "content": "Not explicitly stated in the provided text.", "evidence": []},
                {"section": "experiments", "content": "Not explicitly stated in the provided text.", "evidence": []},
                {"section": "limitations", "content": "Not explicitly stated in the provided text.", "evidence": []},
            ]
            return self._finalize("paper_summary", items, routing, supported=False, answer="文档证据不足，无法确认。")

        context = self._build_context(merged[:10])
        prompt = f"{PAPER_SUMMARY_PROMPT}\n\nContext Passages:\n{context}"
        model_out = self.llm_callable(prompt)
        parsed = self._parse_json(model_out, "paper_summary")
        items = parsed.get("items", []) if isinstance(parsed, dict) else []
        normalized = []
        for row in items:
            section = row.get("section")
            content = row.get("content")
            evid = self._resolve_evidence(row.get("evidence"), merged[:10])
            normalized.append({"section": section, "content": content, "evidence": evid})

        required = ["problem", "method", "contributions", "experiments", "limitations"]
        existing = {str(item.get("section")): item for item in normalized}
        ordered = []
        for sec in required:
            if sec in existing:
                ordered.append(existing[sec])
            else:
                ordered.append({"section": sec, "content": "Not explicitly stated in the provided text.", "evidence": []})

        return self._finalize("paper_summary", ordered, routing, supported=True)


def build_service(**kwargs: Any) -> PaperExtractionService:
    return PaperExtractionService(**kwargs)
