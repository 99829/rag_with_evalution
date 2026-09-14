"""
End-to-end ingestion pipeline: file(s) -> loaded records -> chunks
-> vector store + BM25 index.
"""
from typing import List
from app.ingestion.loaders import load_document, load_directory
from app.ingestion.chunker import chunk_records
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index


def ingest_files(file_paths: List[str], vector_store: VectorStore, bm25_index: BM25Index) -> int:
    all_records = []
    for path in file_paths:
        all_records.extend(load_document(path))

    chunks = chunk_records(all_records)
    vector_store.add_chunks(chunks)
    bm25_index.add(chunks)
    return len(chunks)


def ingest_directory(dir_path: str, vector_store: VectorStore, bm25_index: BM25Index) -> int:
    records = load_directory(dir_path)
    chunks = chunk_records(records)
    vector_store.add_chunks(chunks)
    bm25_index.add(chunks)
    return len(chunks)