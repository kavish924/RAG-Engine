"""
Pydantic request/response models for the API layer (Phase 5).
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- /v1/ask ----------

class AskRequest(BaseModel):
    question: str
    retrieval_mode: Literal["hybrid", "dense_only"] = "hybrid"
    top_k: Optional[int] = None


class Citation(BaseModel):
    marker: str            # e.g. "[1]"
    chunk_id: str
    source_document: str
    section_heading: Optional[str] = None
    supported: bool        # result of citation_verifier
    excerpt: str


class ConfidenceBreakdown(BaseModel):
    retrieval_confidence: float
    citation_coverage: float
    answer_completeness: float
    composite: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: ConfidenceBreakdown
    retrieved_chunk_ids: list[str]
    is_fallback: bool = False   # True if system returned "I don't know" gracefully


# ---------- /v1/documents ----------

class DocumentSummary(BaseModel):
    source_file: str
    num_chunks: int
    chunking_strategies_used: list[str]
    last_indexed_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


# ---------- /v1/ingest ----------

class IngestRequest(BaseModel):
    file_paths: list[str]
    chunking_strategy: Literal["fixed_size", "recursive_structure", "semantic"] = "recursive_structure"


class IngestResponse(BaseModel):
    ingested_files: list[str]
    chunks_created: int
    duplicates_skipped: int
