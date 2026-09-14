"""
Hybrid retrieval: merges dense (vector) and sparse (BM25) results
using Reciprocal Rank Fusion (RRF), so exact keyword hits and
semantic matches both get a fair chance to surface.
"""
from typing import List, Dict
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_index import BM25Index

RRF_K = 60  # standard RRF damping constant


def reciprocal_rank_fusion(
    dense_results: List[Dict],
    sparse_results: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    scores: Dict[str, float] = {}
    chunk_lookup: Dict[str, Dict] = {}

    for rank, item in enumerate(dense_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_lookup[cid] = item

    for rank, item in enumerate(sparse_results):
        cid = item["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        chunk_lookup[cid] = item

    ranked_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**chunk_lookup[cid], "rrf_score": score}
        for cid, score in ranked_ids
    ]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25_index: BM25Index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def retrieve(self, query: str, dense_k: int = 15, sparse_k: int = 15, top_k: int = 10) -> List[Dict]:
        dense_results = self.vector_store.query(query, top_k=dense_k)
        sparse_results = self.bm25_index.query(query, top_k=sparse_k)
        return reciprocal_rank_fusion(dense_results, sparse_results, top_k=top_k)