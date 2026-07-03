"""
POST /v1/ingest
Accepts new documents, runs them through the ingestion pipeline
(load -> chunk -> embed -> dedup -> store), and reports what was indexed.
"""
from fastapi import APIRouter

from app.api.schemas import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    # TODO (Phase 1): pipeline.run(request.file_paths, request.chunking_strategy)
    raise NotImplementedError
