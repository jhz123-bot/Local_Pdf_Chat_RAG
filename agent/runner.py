from __future__ import annotations

from typing import Any, Dict, List

from .router_rules import choose_route
from .tools import retrieve_vector, retrieve_bm25, merge_dedupe, optional_rerank
from .verifier import verify_with_fallback


class AgenticRAGRunner:
    """Agent wrapper with optional LangChain tool-calling orchestration."""

    def __init__(
        self,
        embed_model: Any,
        faiss_index: Any,
        id_order: List[str],
        content_map: Dict[str, str],
        metadata_map: Dict[str, Dict[str, Any]],
        bm25_manager: Any,
        rerank_fn: Any | None = None,
        ollama_model: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ):
        self.embed_model = embed_model
        self.faiss_index = faiss_index
        self.id_order = id_order
        self.content_map = content_map
        self.metadata_map = metadata_map
        self.bm25_manager = bm25_manager
        self.rerank_fn = rerank_fn
        self.ollama_model = ollama_model

    def _retrieve(self, question: str, route: str, top_k: int) -> List[Dict[str, Any]]:
        if route == "vector":
            vector_rows = retrieve_vector(question, self.embed_model, self.faiss_index, self.id_order, self.content_map, self.metadata_map, top_k=top_k)
            return optional_rerank(question, vector_rows, self.rerank_fn, top_k=min(6, top_k))

        if route == "bm25":
            bm25_rows = retrieve_bm25(question, self.bm25_manager, top_k=top_k, metadata_map=self.metadata_map)
            return optional_rerank(question, bm25_rows, self.rerank_fn, top_k=min(6, top_k))

        # hybrid default
        vector_rows = retrieve_vector(question, self.embed_model, self.faiss_index, self.id_order, self.content_map, self.metadata_map, top_k=top_k)
        bm25_rows = retrieve_bm25(question, self.bm25_manager, top_k=top_k, metadata_map=self.metadata_map)
        merged = merge_dedupe(vector_rows, bm25_rows, limit=top_k)
        return optional_rerank(question, merged, self.rerank_fn, top_k=min(6, top_k))

    def _try_langchain_orchestrate(self, question: str, route: str, top_k: int) -> None:
        """Optional orchestration path. Best-effort; no hard dependency when Agent Mode is off."""
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.tools import tool
            from langchain_community.chat_models import ChatOllama
        except Exception:
            return

        state: Dict[str, Any] = {"route": route, "top_k": top_k, "passages": []}

        @tool
        def retrieve_tool(q: str) -> str:
            """Retrieve passages according to selected route and save state."""
            state["passages"] = self._retrieve(q, state["route"], state["top_k"])
            return f"retrieved {len(state['passages'])} passages"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是论文阅读助手的编排代理。请调用检索工具1次即可。"),
                ("human", "问题：{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        llm = ChatOllama(model=self.ollama_model, temperature=0)
        agent = create_tool_calling_agent(llm, [retrieve_tool], prompt)
        executor = AgentExecutor(agent=agent, tools=[retrieve_tool], verbose=False, max_iterations=2, handle_parsing_errors=True)
        try:
            executor.invoke({"input": question})
        except Exception:
            # silent fallback to deterministic path
            return

    def run(self, question: str) -> Dict[str, Any]:
        kb_ready = bool(self.faiss_index is not None and getattr(self.faiss_index, "ntotal", 0) > 0)
        routing = choose_route(question, kb_ready=kb_ready)
        route = routing["route"]
        top_k = int(routing["top_k"])

        # optional LangChain tool-calling orchestration (best-effort)
        self._try_langchain_orchestrate(question, route, top_k)

        result = verify_with_fallback(
            question=question,
            retrieve_once=lambda q, k: self._retrieve(q, route=route, top_k=k),
            first_top_k=top_k,
            fallback_top_k=min(top_k + 4, 14),
            max_rounds=2,
        )

        return {
            "passages": result["passages"],
            "routing": {**routing, "rounds": result["rounds"]},
            "evidence": result["evidence"],
            "supported": result["supported"],
            "retrieval": {
                "route": route,
                "top_k": top_k,
                "fallback_top_k": min(top_k + 4, 14),
                "logs": result["retrieval_log"],
            },
        }
