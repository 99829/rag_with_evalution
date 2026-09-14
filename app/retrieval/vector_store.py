"""
Chroma-backed vector store wrapper.
Uses a local persistent Chroma instance so the project runs
without any paid vector DB during development.
"""
import os
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "multidoc_rag"


def get_default_embedding_fn():
    """Lazily construct the default embedding function. Kept lazy
    (not module-level) so simply importing this module doesn't
    trigger a model download/load — that only happens when a
    VectorStore is actually instantiated without a custom embedding_fn."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, embedding_fn=None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedding_fn = embedding_fn or get_default_embedding_fn()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
        )

    def add_chunks(self, chunks: List[Dict]) -> int:
        """Add chunk records (from chunker.py) to the vector store."""
        if not chunks:
            return 0
        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{
                "doc_id": c["doc_id"],
                "doc_name": c["doc_name"],
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
            } for c in chunks],
        )
        return len(chunks)

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Dense similarity search. Returns chunks with scores."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def delete_document(self, doc_id: str):
        self.collection.delete(where={"doc_id": doc_id})

    def list_documents(self) -> List[str]:
        data = self.collection.get()
        doc_names = {m["doc_name"] for m in data["metadatas"]}
        return sorted(doc_names)

    def count(self) -> int:
        return self.collection.count()