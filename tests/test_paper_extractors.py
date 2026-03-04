import json
import numpy as np

from domain.papers.extractors import build_service


class FakeEmbed:
    def encode(self, texts):
        return np.array([[0.1, 0.2, 0.3] for _ in texts])


class FakeFaiss:
    def __init__(self, n=8):
        self.ntotal = n

    def search(self, q_vec, k=8):
        idx = np.array([list(range(min(k, self.ntotal)))])
        dist = np.zeros_like(idx, dtype=float)
        return dist, idx


class FakeBM25:
    def __init__(self, rows):
        self.rows = rows
        self.bm25_index = True

    def search(self, query, top_k=8):
        return self.rows[:top_k]


def make_service(task_json):
    id_order = [f"doc_{i}" for i in range(8)]
    content_map = {f"doc_{i}": f"This is content chunk {i}. We propose a framework and report dataset + metric results." for i in range(8)}
    metadata_map = {
        f"doc_{i}": {
            "source": "paper.pdf",
            "page": i + 1,
            "section_title": ["Abstract", "Introduction", "Method", "Experiments", "Conclusion"][i % 5],
        }
        for i in range(8)
    }
    bm25_rows = [{"id": f"doc_{i}", "content": content_map[f"doc_{i}"], "score": 1.0} for i in range(8)]

    def fake_rerank(query, docs, ids, metas, top_k=5):
        return [(doc_id, {"content": d, "metadata": m, "score": 1.0}) for d, doc_id, m in zip(docs[:top_k], ids[:top_k], metas[:top_k])]

    return build_service(
        embed_model=FakeEmbed(),
        faiss_index=FakeFaiss(),
        id_order=id_order,
        content_map=content_map,
        metadata_map=metadata_map,
        bm25_manager=FakeBM25(bm25_rows),
        rerank_fn=fake_rerank,
        llm_callable=lambda prompt: json.dumps(task_json, ensure_ascii=False),
    )


def test_extract_contributions_schema():
    service = make_service(
        {
            "task": "contributions",
            "items": [
                {"title": "Novel framework", "content": "Introduces a new framework", "evidence": "we propose a framework"}
            ],
        }
    )
    out = service.extract_contributions("请提取论文贡献")
    assert out["task"] == "contributions"
    assert isinstance(out["items"], list)
    if out["supported"]:
        assert out["items"][0]["evidence"]


def test_extract_method_pipeline_schema():
    service = make_service(
        {
            "task": "method_pipeline",
            "items": [
                {"step": 2, "title": "Encoder", "content": "Encode inputs", "evidence": "framework"},
                {"step": 1, "title": "Preprocess", "content": "Prepare data", "evidence": "content chunk"},
            ],
        }
    )
    out = service.extract_method_pipeline("请提取方法流程")
    steps = [x["step"] for x in out["items"]]
    assert steps == sorted(steps)
    assert all(isinstance(s, int) for s in steps)
    if out["supported"]:
        assert all(item["evidence"] for item in out["items"])


def test_extract_experiment_setup_schema():
    service = make_service(
        {
            "task": "experiment_setup",
            "items": [
                {"category": "dataset", "title": "Dataset A", "content": "Used dataset A", "evidence": "dataset"},
                {"category": "metric", "title": "F1", "content": "Use F1", "evidence": "metric"},
            ],
        }
    )
    out = service.extract_experiment_setup("提取实验设置")
    allowed = {"dataset", "baseline", "metric", "training"}
    assert all(item["category"] in allowed for item in out["items"])
    if out["supported"]:
        assert all(item["evidence"] for item in out["items"])


def test_paper_summary_schema():
    service = make_service(
        {
            "task": "paper_summary",
            "items": [
                {"section": "problem", "content": "problem", "evidence": ["challenge"]},
                {"section": "method", "content": "method", "evidence": ["framework"]},
                {"section": "contributions", "content": ["c1", "c2"], "evidence": ["we propose"]},
                {"section": "experiments", "content": "exp", "evidence": ["dataset"]},
                {"section": "limitations", "content": "limit", "evidence": ["future work"]},
            ],
        }
    )
    out = service.generate_paper_summary("总结论文")
    sections = [x["section"] for x in out["items"]]
    required = {"problem", "method", "contributions", "experiments", "limitations"}
    assert set(sections) == required
