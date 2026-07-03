"""
Citation verification (Phase 3, step 2).
For each citation-claim pair the model produced, checks whether the cited
chunk actually supports the claim, using an LLM-as-judge. Flags
unsupported citations rather than trusting the generator blindly —
this is the quality layer most RAG systems skip.
"""


def verify_citations(raw_citations: list[dict], chunks: list[dict]) -> list[dict]:
    """Returns citations annotated with a `supported: bool` field."""
    raise NotImplementedError
