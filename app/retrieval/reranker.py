"""
Cross-encoder reranking: takes the top ~20 hybrid candidates and
re-scores them with a model that looks at query+chunk together
(much more accurate than embedding similarity alone).
"""
from typing import List, Dict
from sentence_transformers import CrossEncoder

_model = None


def get_reranker_model():
    global _model
    if _model is None:
        _model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _model


def rerank(query: str, candidates: List[Dict], top_n: int = 5) -> List[Dict]:
    if not candidates:
        return []
    model = get_reranker_model()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)

    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)

    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]