from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.papers.extractors import build_service
from eval.metrics import citation_rate, coverage_at_k, hallucination_proxy, schema_valid_rate


class DummyEmbed:
    def encode(self, texts):
        return np.array([[0.01, 0.02, 0.03] for _ in texts])


class DummyFaiss:
    def __init__(self, n_total: int):
        self.ntotal = n_total

    def search(self, q_vec, k=8):
        k = min(k, self.ntotal)
        idx = np.array([list(range(k))])
        dist = np.zeros((1, k))
        return dist, idx


class KeywordBM25:
    def __init__(self, docs: List[Dict[str, Any]]):
        self.docs = docs
        self.bm25_index = True

    def search(self, query: str, top_k: int = 8):
        q_terms = [t.lower() for t in query.replace("\n", " ").split() if t.strip()]
        scored = []
        for d in self.docs:
            text = d["content"].lower()
            score = sum(1 for q in q_terms if q in text)
            scored.append({"id": d["id"], "content": d["content"], "score": float(score)})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_dummy_llm(task: str):
    def _call(_prompt: str) -> str:
        if task == "contributions":
            obj = {
                "task": "contributions",
                "items": [
                    {"title": "Main contribution", "content": "提出了一种新方法用于论文任务", "evidence": "we propose a framework"},
                    {"title": "Performance gain", "content": "在多个数据集上提升效果", "evidence": "outperforms baseline"},
                ],
            }
        elif task == "method_pipeline":
            obj = {
                "task": "method_pipeline",
                "items": [
                    {"step": 1, "title": "Input processing", "content": "预处理输入", "evidence": "preprocess input"},
                    {"step": 2, "title": "Encoder", "content": "编码特征", "evidence": "encoder module"},
                ],
            }
        elif task == "experiment_setup":
            obj = {
                "task": "experiment_setup",
                "items": [
                    {"category": "dataset", "title": "ToySet", "content": "使用ToySet数据集", "evidence": "dataset ToySet"},
                    {"category": "metric", "title": "F1", "content": "采用F1指标", "evidence": "metric F1"},
                ],
            }
        else:
            obj = {
                "task": "paper_summary",
                "items": [
                    {"section": "problem", "content": "解决文档理解问题", "evidence": ["problem motivation"]},
                    {"section": "method", "content": "基于模块化框架", "evidence": ["framework method"]},
                    {"section": "contributions", "content": ["新框架", "更好效果"], "evidence": ["main contributions"]},
                    {"section": "experiments", "content": "在ToySet上验证", "evidence": ["dataset ToySet"]},
                    {"section": "limitations", "content": "仍需更多真实数据", "evidence": ["future work"]},
                ],
            }
        return json.dumps(obj, ensure_ascii=False)

    return _call


def run_eval(dataset_dir: Path) -> Dict[str, Any]:
    chunks = load_jsonl(dataset_dir / "chunks.jsonl")
    gold = json.loads((dataset_dir / "gold.json").read_text(encoding="utf-8"))
    paper_meta = json.loads((dataset_dir / "paper_meta.json").read_text(encoding="utf-8"))

    id_order = [c["id"] for c in chunks]
    content_map = {c["id"]: c["content"] for c in chunks}
    metadata_map = {c["id"]: {"source": paper_meta.get("paper_id", "toy_paper"), "page": c.get("page"), "section_title": c.get("section_title", "")} for c in chunks}

    def fake_rerank(query, docs, ids, metas, top_k=5):
        return [
            (doc_id, {"content": d, "metadata": m, "score": 1.0})
            for d, doc_id, m in zip(docs[:top_k], ids[:top_k], metas[:top_k])
        ]

    tasks = [
        ("contributions", "extract contributions"),
        ("method_pipeline", "extract method pipeline"),
        ("experiment_setup", "extract experiment setup"),
        ("paper_summary", "generate paper summary"),
    ]

    outputs = []
    for task, question in tasks:
        service = build_service(
            embed_model=DummyEmbed(),
            faiss_index=DummyFaiss(len(chunks)),
            id_order=id_order,
            content_map=content_map,
            metadata_map=metadata_map,
            bm25_manager=KeywordBM25(chunks),
            rerank_fn=fake_rerank,
            llm_callable=build_dummy_llm(task),
        )
        if task == "contributions":
            result = service.extract_contributions(question)
        elif task == "method_pipeline":
            result = service.extract_method_pipeline(question)
        elif task == "experiment_setup":
            result = service.extract_experiment_setup(question)
        else:
            result = service.generate_paper_summary(question)
        outputs.append(result)

    metrics = {
        "schema_valid_rate": schema_valid_rate(outputs),
        "citation_rate": citation_rate(outputs),
        "coverage@k": coverage_at_k(outputs, gold, k=5),
        "hallucination_proxy": hallucination_proxy(outputs),
    }

    return {
        "dataset": str(dataset_dir),
        "metrics": metrics,
        "results": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline eval for Paper Reading Copilot")
    parser.add_argument("--dataset", type=str, default="eval/datasets/toy_papers", help="Dataset directory")
    parser.add_argument("--output", type=str, default="eval_outputs.json", help="Output json file")
    args = parser.parse_args()

    payload = run_eval(Path(args.dataset))
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Eval done. Output saved to {args.output}")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
