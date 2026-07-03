"""
Orchestrates the full hybrid retrieval pipeline:
  dense_search + bm25_search -> RRF fusion -> rerank -> top-N chunks.

Also supports a "dense_only" mode (skips BM25 + fusion entirely) for the
dashboard's side-by-side hybrid vs. dense-only comparison.
"""
from typing import Literal

from app.config import settings
from app.retrieval.dense import dense_search
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.reranker import rerank
from app.retrieval.sparse_bm25 import bm25_search

RetrievalMode = Literal["hybrid", "dense_only"]


def retrieve(
    query: str,
    mode: RetrievalMode = "hybrid",
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    rerank_top_n: int | None = None,
) -> list[dict]:
    """
    Returns the final reranked list of chunks (each with "id", "text",
    "metadata", "relevance_score", plus fusion-specific fields when mode
    is "hybrid").
    """
    dense_top_k = dense_top_k or settings.dense_top_k
    sparse_top_k = sparse_top_k or settings.sparse_top_k
    rerank_top_n = rerank_top_n or settings.rerank_top_n

    dense_results = dense_search(query, top_k=dense_top_k)

    if mode == "dense_only":
        candidates = dense_results
    else:
        sparse_results = bm25_search(query, top_k=sparse_top_k)
        candidates = reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            dense_weight=settings.rrf_dense_weight,
            sparse_weight=settings.rrf_sparse_weight,
        )

    return rerank(query, candidates, keep_top_n=rerank_top_n)