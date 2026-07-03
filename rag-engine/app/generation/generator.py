"""
Calls the LLM with the grounded prompt and retrieved chunks, and parses
out the answer text plus raw (unverified) citation markers.
"""


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Returns: {"answer": str, "raw_citations": list[dict]}
    """
    raise NotImplementedError
