"""
Ingestion package.

Provides utilities for:
- Loading documents
- Chunking text
- Generating embeddings
- Deduplicating chunks
- Running the ingestion pipeline
"""

from .loaders import LoadedDocument, load_document
from .embeddings import embed_query, embed_texts
from .dedup import filter_duplicates, is_near_duplicate
from .pipeline import run_ingestion

__all__ = [
    "LoadedDocument",
    "load_document",
    "embed_query",
    "embed_texts",
    "filter_duplicates",
    "is_near_duplicate",
    "run_ingestion",
]