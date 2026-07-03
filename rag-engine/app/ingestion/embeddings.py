"""
Embedding wrapper around OpenAI's text-embedding-3-small.
Batches requests and returns plain lists of floats (no SDK objects
leaking into the rest of the codebase).
"""
from functools import lru_cache

from app.config import settings

_MAX_BATCH = 100  # keep batches modest; OpenAI allows more but this is safe


@lru_cache(maxsize=1)
def _get_client():
    from openai import OpenAI

    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a list of strings, preserving input order. Empty list in -> empty list out."""
    if not texts:
        return []

    client = _get_client()
    model = model or settings.embedding_model

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _MAX_BATCH):
        batch = texts[i : i + _MAX_BATCH]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])

    return all_embeddings


def embed_query(query: str, model: str | None = None) -> list[float]:
    """Convenience wrapper for embedding a single query string."""
    return embed_texts([query], model=model)[0]
