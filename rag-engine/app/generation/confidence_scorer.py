"""
Composite confidence scorer (Phase 3, step 3).
Combines:
  - retrieval_confidence: how relevant were the top chunks?
  - citation_coverage: % of claims with verified citations
  - answer_completeness: did the response address all parts of the question?
into a single composite score returned alongside the answer.
"""


def score_confidence(chunks: list[dict], verified_citations: list[dict], answer: str) -> dict:
    """
    Returns: {
      "retrieval_confidence": float,
      "citation_coverage": float,
      "answer_completeness": float,
      "composite": float,
    }
    """
    raise NotImplementedError
