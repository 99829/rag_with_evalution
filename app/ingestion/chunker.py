"""
Chunking strategy: recursive character splitting with overlap,
preserving doc_name + page_number metadata on every chunk so
citations are possible downstream.
"""
import hashlib
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def make_chunk_id(doc_name: str, page_number: int, chunk_index: int) -> str:
    raw = f"{doc_name}-{page_number}-{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def chunk_records(
    records: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict]:
    """
    Takes page-level records (from loaders.py) and splits them into
    retrieval-sized chunks, keeping doc_name/page_number metadata
    and assigning a stable chunk_id + doc_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for record in records:
        doc_name = record["doc_name"]
        page_number = record["page_number"]
        doc_id = hashlib.md5(doc_name.encode()).hexdigest()[:10]

        pieces = splitter.split_text(record["text"])
        for idx, piece in enumerate(pieces):
            if not piece.strip():
                continue
            all_chunks.append({
                "chunk_id": make_chunk_id(doc_name, page_number, idx),
                "doc_id": doc_id,
                "doc_name": doc_name,
                "page_number": page_number,
                "chunk_index": idx,
                "text": piece.strip(),
            })
    return all_chunks