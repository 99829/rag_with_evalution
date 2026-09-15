"""
FastAPI app exposing:
  POST   /documents   - upload one or more files, ingest them
  GET    /documents    - list ingested documents
  DELETE /documents/{doc_id} - remove a document's chunks
  POST   /query        - ask a question, get a cited answer
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.graph import build_rag_graph
from app.ingestion.pipeline import ingest_files
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Multi-Doc RAG API")

# NOTE: single global in-memory BM25 index for simplicity. For multi-user /
# production use, this would need per-session or persisted indexing.
vector_store = VectorStore()
bm25_index = BM25Index()
retriever = HybridRetriever(vector_store, bm25_index)
rag_app = build_rag_graph(retriever)
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str
    citations_verified: bool
    flagged_citations: List[str]
    model_used: str
    num_sources_used: int


@app.get("/documents")
async def list_documents():
    return {"documents": vector_store.list_documents(), "total_chunks": vector_store.count()}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    vector_store.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Upload documents first.")

    result = rag_app.invoke({
        "question": request.question,
        "query_type": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": None,
        "model_used": None,
        "citations_verified": None,
        "flagged_citations": [],
    })

    return QueryResponse(
        answer=result["answer"],
        citations_verified=result["citations_verified"],
        flagged_citations=result["flagged_citations"],
        model_used=result["model_used"],
        num_sources_used=len(result["reranked_chunks"]),
    )
@app.get("/health")
async def health():
    return {"status": "ok", "documents_indexed": vector_store.count()}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    vector_store.delete_document(doc_id)
    bm25_index.delete(doc_id)
    return {"status": "deleted", "doc_id": doc_id}