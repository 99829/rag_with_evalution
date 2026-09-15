"""
BM25 sparse retrieval index — catches exact keyword/entity matches
that dense embedding search sometimes misses.
"""
from typing import List, Dict
from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build(self, chunks: List[Dict]):
        """Build (or rebuild) the BM25 index from chunk records."""
        self.chunks = chunks
        tokenized_corpus = [self._tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def add(self, chunks: List[Dict]):
        """Incrementally add chunks by rebuilding (BM25Okapi has no
        native incremental API — fine at this project's scale)."""
        self.build(self.chunks + chunks)

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        if self.bm25 is None or not self.chunks:
            return []
        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        scored_chunks = sorted(
            zip(self.chunks, scores), key=lambda x: x[1], reverse=True
        )[:top_k]
        return [
            {
                "chunk_id": c["chunk_id"],
                "text": c["text"],
                "metadata": {
                    "doc_id": c["doc_id"],
                    "doc_name": c["doc_name"],
                    "page_number": c["page_number"],
                    "chunk_index": c["chunk_index"],
                },
                "bm25_score": float(score),
            }
            for c, score in scored_chunks if score > 0
        ]
    def delete(self, doc_id: str):
        """Remove all chunks belonging to a given doc_id and rebuild the index."""
        remaining_chunks = [c for c in self.chunks if c["doc_id"] != doc_id]
        self.build(remaining_chunks)