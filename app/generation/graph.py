"""
LangGraph orchestration for the RAG pipeline.

Nodes:
  classify_query -> retrieve -> rerank -> generate_answer -> verify_citations -> END

Each run is auto-traced by LangSmith as long as LANGCHAIN_TRACING_V2=true
and LANGCHAIN_API_KEY are set in the environment.
"""
import re
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import rerank
from app.generation.llm_client import invoke_with_fallback
from app.generation.prompts import SYSTEM_PROMPT, ANSWER_PROMPT_TEMPLATE, format_context


class RAGState(TypedDict):
    question: str
    query_type: Optional[str]        # "single_doc" | "multi_doc"
    retrieved_chunks: List[Dict]
    reranked_chunks: List[Dict]
    answer: Optional[str]
    model_used: Optional[str]
    citations_verified: Optional[bool]
    flagged_citations: List[str]


def classify_query_node(state: RAGState) -> RAGState:
    """Cheap heuristic classifier: looks for comparison/multi-doc signal words."""
    q = state["question"].lower()
    multi_doc_signals = ["compare", "difference between", "versus", "vs", "across", "both documents"]
    query_type = "multi_doc" if any(sig in q for sig in multi_doc_signals) else "single_doc"
    return {**state, "query_type": query_type}


def make_retrieve_node(retriever: HybridRetriever):
    def retrieve_node(state: RAGState) -> RAGState:
        top_k = 20 if state["query_type"] == "multi_doc" else 12
        chunks = retriever.retrieve(state["question"], top_k=top_k)
        return {**state, "retrieved_chunks": chunks}
    return retrieve_node


def rerank_node(state: RAGState) -> RAGState:
    top_n = 6 if state["query_type"] == "multi_doc" else 4
    reranked = rerank(state["question"], state["retrieved_chunks"], top_n=top_n)
    return {**state, "reranked_chunks": reranked}


def generate_answer_node(state: RAGState) -> RAGState:
    context = format_context(state["reranked_chunks"])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=ANSWER_PROMPT_TEMPLATE.format(
            question=state["question"], context=context
        )),
    ]
    response, model_used = invoke_with_fallback(messages)
    return {**state, "answer": response.content, "model_used": model_used}


def verify_citations_node(state: RAGState) -> RAGState:
    """Basic grounding check: every [doc_name, p.X] cited in the answer
    should correspond to a chunk that was actually retrieved."""
    answer = state["answer"] or ""
    cited = set(re.findall(r"[\[【]([^\]】,]+),\s*p\.(\d+)[\]】]", answer))
    available = {
        (c["metadata"]["doc_name"], str(c["metadata"]["page_number"]))
        for c in state["reranked_chunks"]
    }
    flagged = [f"{doc}, p.{page}" for doc, page in cited if (doc, page) not in available]
    return {
        **state,
        "citations_verified": len(flagged) == 0,
        "flagged_citations": flagged,
    }


def build_rag_graph(retriever: HybridRetriever):
    graph = StateGraph(RAGState)
    graph.add_node("classify_query", classify_query_node)
    graph.add_node("retrieve", make_retrieve_node(retriever))
    graph.add_node("rerank", rerank_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("verify_citations", verify_citations_node)

    graph.set_entry_point("classify_query")
    graph.add_edge("classify_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "generate_answer")
    graph.add_edge("generate_answer", "verify_citations")
    graph.add_edge("verify_citations", END)

    return graph.compile()