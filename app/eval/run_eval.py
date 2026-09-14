"""
Runs the full golden dataset through the RAG graph and logs results as a
LangSmith Experiment, using the custom evaluators.

Usage:
    python -m app.eval.run_eval
"""
import os
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate

from app.eval.golden_dataset import get_golden_dataset
from app.eval.evaluators import citation_accuracy_evaluator, citations_grounded_evaluator
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.graph import build_rag_graph
from app.ingestion.pipeline import ingest_directory

# DATASET_NAME = "multidoc-rag-golden-eval-v2"
DATASET_NAME = "multidoc-rag-golden-eval-v3"


def upload_golden_dataset(client: Client):
    existing = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        return existing[0]

    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    for item in get_golden_dataset():
        client.create_example(
            inputs={"question": item["question"]},
            outputs={
                "expected_answer": item["expected_answer"],
                "expected_sources": item["expected_sources"],
            },
            dataset_id=dataset.id,
        )
    return dataset


def make_target_fn(rag_app):
    def target(inputs: dict) -> dict:
        result = rag_app.invoke({
            "question": inputs["question"],
            "query_type": None,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "answer": None,
            "model_used": None,
            "citations_verified": None,
            "flagged_citations": [],
        })
        return {
            "answer": result["answer"],
            "citations_verified": result["citations_verified"],
            "flagged_citations": result["flagged_citations"],
            "model_used": result["model_used"],
        }
    return target


def main():
    client = Client()

    vector_store = VectorStore()
    bm25_index = BM25Index()
    ingest_directory("data/sample_docs", vector_store, bm25_index)

    retriever = HybridRetriever(vector_store, bm25_index)
    rag_app = build_rag_graph(retriever)

    dataset = upload_golden_dataset(client)

    results = evaluate(
        make_target_fn(rag_app),
        data=dataset.name,
        evaluators=[citation_accuracy_evaluator, citations_grounded_evaluator],
        experiment_prefix="multidoc-rag",
    )
    print("Eval run complete. View results in the LangSmith UI under project:",
          os.getenv("LANGCHAIN_PROJECT", "multidoc-rag"))
    return results


if __name__ == "__main__":
    main()