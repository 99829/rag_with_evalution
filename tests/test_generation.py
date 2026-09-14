"""End-to-end generation test: ingest -> retrieve -> rerank -> generate -> verify."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.graph import build_rag_graph
from app.ingestion.pipeline import ingest_directory

DATA_DIR = str(Path(__file__).resolve().parents[1] / "data" / "sample_docs")

print("=== Setting up pipeline ===")
vector_store = VectorStore()
bm25_index = BM25Index()
ingest_directory(DATA_DIR, vector_store, bm25_index)
print(f"Indexed {vector_store.count()} chunks")

retriever = HybridRetriever(vector_store, bm25_index)
rag_app = build_rag_graph(retriever)

questions = [
    "How many days of annual leave do employees get?",
    "Compare the notice period for leave requests versus travel bookings.",
]

for q in questions:
    print(f"\n{'='*60}\nQ: {q}\n{'='*60}")
    result = rag_app.invoke({
        "question": q, "query_type": None, "retrieved_chunks": [],
        "reranked_chunks": [], "answer": None, "model_used": None,
        "citations_verified": None, "flagged_citations": [],
    })
    print(f"Query type: {result['query_type']}")
    print(f"Model used: {result['model_used']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nCitations verified: {result['citations_verified']}")
    if result["flagged_citations"]:
        print(f"⚠️ Flagged (hallucinated) citations: {result['flagged_citations']}")