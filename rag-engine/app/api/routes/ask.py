"""
POST /v1/ask
Accepts a question, runs hybrid retrieval + grounded generation + citation
verification + confidence scoring, and returns a structured answer.
"""
from fastapi import APIRouter

from app.api.schemas import AskRequest, AskResponse

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    # TODO (Phase 2/3):
    #   1. chunks = retriever.retrieve(request.question, mode=request.retrieval_mode)
    #   2. answer, raw_citations = generator.generate(request.question, chunks)
    #   3. verified_citations = citation_verifier.verify(raw_citations, chunks)
    #   4. confidence = confidence_scorer.score(chunks, verified_citations, answer)
    #   5. if confidence.composite < settings.confidence_threshold: return fallback response
    raise NotImplementedError
