"""
Near-duplicate detection (Phase 1, step 4).

Before inserting a chunk, check cosine similarity against existing chunk
embeddings already in the store. If similarity > threshold (default
0.95), flag and skip — prevents the retriever from wasting context
window slots on redundant content that appears in multiple docs.
"""
import numpy as np


def _cosine_similarity_matrix(candidate: list[float], existing: list[list[float]]) -> np.ndarray:
    if not existing:
        return np.array([])
    cand = np.array(candidate)
    exist = np.array(existing)
    cand_norm = np.linalg.norm(cand)
    exist_norms = np.linalg.norm(exist, axis=1)
    denom = cand_norm * exist_norms
    denom[denom == 0] = 1e-10  # avoid div-by-zero
    return (exist @ cand) / denom


def is_near_duplicate(
    candidate_embedding: list[float],
    existing_embeddings: list[list[float]],
    threshold: float = 0.95,
) -> bool:
    """Returns True if candidate's max similarity to any existing embedding exceeds threshold."""
    if not existing_embeddings:
        return False
    similarities = _cosine_similarity_matrix(candidate_embedding, existing_embeddings)
    return bool(similarities.size and similarities.max() > threshold)


def filter_duplicates(
    candidate_embeddings: list[list[float]],
    existing_embeddings: list[list[float]] | None = None,
    threshold: float = 0.95,
) -> tuple[list[int], list[int]]:
    """
    Filters a batch of new candidate embeddings against both an existing
    pool and each other (in case the same batch contains internal dupes).

    Returns (keep_indices, duplicate_indices) into `candidate_embeddings`.
    """
    existing_embeddings = list(existing_embeddings or [])
    keep_indices: list[int] = []
    duplicate_indices: list[int] = []

    pool = list(existing_embeddings)
    for i, emb in enumerate(candidate_embeddings):
        if is_near_duplicate(emb, pool, threshold=threshold):
            duplicate_indices.append(i)
        else:
            keep_indices.append(i)
            pool.append(emb)  # so later items in this batch are also checked against it

    return keep_indices, duplicate_indices
