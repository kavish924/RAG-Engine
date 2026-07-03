"""
GET /v1/documents
Lists indexed documents with chunk counts and which chunking strategy(ies)
were used for each — useful for the dashboard and for debugging ingestion.
"""
from fastapi import APIRouter

from app.api.schemas import DocumentListResponse

router = APIRouter()


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    # TODO (Phase 1): query vector_store metadata, group by source_file
    raise NotImplementedError
