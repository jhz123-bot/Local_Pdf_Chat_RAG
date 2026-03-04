
# 📄 Paper Reading Copilot (Local_Pdf_Chat_RAG)

一个面向“论文阅读助手”的本地化 RAG + Agentic RAG 项目。默认模式仍是普通问答；可选开启 Agent Mode 获取 citation-aware 证据展示。

---

## Quickstart

### 1) 环境准备
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) 配置（可选）
在项目根目录可配置：
```env
SERPAPI_KEY=...
SILICONFLOW_API_KEY=...
```
> 不配置 key 也可启动 UI；仅在调用云端模型时会提示未配置。

### 3) 启动
```bash
python rag_demo.py
```
默认会尝试端口 `17995~17999`。

---

## Demo Script

### 常规问答
1. 上传论文（pdf/txt/docx 等）。
2. 输入问题，点击 **开始提问**。
3. 可选勾选 **Agent Mode (citation-aware)**。

### 论文垂直按钮（PR2）
在问答页点击以下任一按钮：
- `Extract Contributions`
- `Extract Method Pipeline`
- `Extract Experiment Setup`
- `Generate Paper Summary`

输出为结构化 `items`，并在折叠区显示 evidence（source/page）。

---

## Offline Evaluation

本项目提供完全离线评估，不依赖网络和真实 LLM。

```bash
python eval/run_eval.py --dataset eval/datasets/toy_papers --output eval_outputs.json
```

输出：`eval_outputs.json`，包含 metrics + 每个任务结果。

Toy 数据集路径：
- `eval/datasets/toy_papers/chunks.jsonl`
- `eval/datasets/toy_papers/gold.json`
- `eval/datasets/toy_papers/paper_meta.json`

---

## Metrics

`eval/metrics.py` 提供：
- `schema_valid_rate`
- `citation_rate`
- `coverage@k`
- `hallucination_proxy`

---

## Project Structure

```text
.
├── rag_demo.py                 # 主应用（Gradio UI + RAG + Agent Mode 接入）
├── api_router.py               # FastAPI 封装
├── agent/
│   ├── router_rules.py         # 检索路由规则
│   ├── tools.py                # 检索/合并/证据工具
│   ├── verifier.py             # 证据核对 + fallback
│   └── runner.py               # Agentic RAG 运行器
├── domain/papers/
│   ├── prompts.py              # 论文 extractor 稳定 prompts
│   └── extractors.py           # 论文 extractor 逻辑（复用 agent 检索能力）
├── eval/
│   ├── run_eval.py             # 离线评估入口
│   ├── metrics.py              # 评估指标
│   └── datasets/toy_papers/    # 小型离线数据集
└── tests/                      # 离线可跑单元测试
```

---

## Testing

```bash
python -m pytest -q
```

---

## Troubleshooting

### 1) ollama 未启动
- 现象：本地模型调用失败。
- 处理：执行 `ollama serve`，并确认模型已拉取。

### 2) 端口占用
- 现象：启动失败提示端口被占用。
- 处理：释放 `17995~17999` 端口，或修改启动端口列表。

### 3) 缺少 API key
- 现象：siliconflow 调用返回未配置提示。
- 处理：配置 `SILICONFLOW_API_KEY`，或切换到 ollama。

### 4) 版本冲突
- 现象：gradio/hf 依赖报错。
- 处理：使用 `requirements.txt` 的 pinned 版本，建议新虚拟环境安装。

---

## Acknowledgements

- FAISS
- Sentence Transformers
- Gradio
- LangChain ecosystem
- Rank-BM25