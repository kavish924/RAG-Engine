"""
Baseline chunking strategy: fixed-size windows with overlap.
Simple, fast, no structural awareness — the control group for the
chunking strategy comparison in Phase 4.

Sizes are measured in characters for simplicity/determinism; swap to a
tokenizer (tiktoken) if you want token-accurate windows instead.
"""


def chunk_fixed_size(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks
