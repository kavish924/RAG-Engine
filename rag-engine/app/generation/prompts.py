"""
Grounded generation prompt templates (Phase 3, step 1).
Instructs the LLM to answer only from provided context, cite chunks with
bracketed references ([1], [2]), and explicitly say when context is
insufficient rather than guessing.
"""

GROUNDED_SYSTEM_PROMPT = """\
You are a documentation assistant. Answer the user's question using ONLY
the numbered context blocks provided below.

Rules:
- Cite every factual claim with the bracketed number(s) of the chunk(s)
  that support it, e.g. "Rate limits reset hourly [2]."
- If the context does not contain enough information to answer, say so
  explicitly rather than guessing or using outside knowledge.
- Do not fabricate citations. Only cite chunks that were actually provided.

Context:
{context_blocks}
"""


def build_context_blocks(chunks: list[dict]) -> str:
    raise NotImplementedError
