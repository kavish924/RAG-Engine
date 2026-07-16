"""
Citation verifier (Phase 3, step 2).

Verifies generated citations against retrieved source chunks.

Optimized approach:
- No LLM call
- No Ollama generation
- Deterministic local verification
- Token overlap between citation excerpt and source chunk
- Preserves the citation response structure expected by the API
"""

import re


# Minimum token overlap required for a citation to be supported.
CITATION_OVERLAP_THRESHOLD = 0.35


def verify_citations(
    raw_citations: list[dict],
    chunks: list[dict],
) -> list[dict]:
    """
    Verify every generated citation against its referenced chunk.

    Expected raw citation format:

        {
            "marker": "[1]",
            "excerpt": "Docker is a containerization platform"
        }

    Returns:

        {
            "marker": "[1]",
            "chunk_id": "...",
            "source_document": "...",
            "section_heading": "...",
            "supported": True,
            "excerpt": "..."
        }

    Citation markers are mapped to retrieved chunks by their
    1-indexed position:

        [1] -> chunks[0]
        [2] -> chunks[1]
        [3] -> chunks[2]
    """

    if not raw_citations:
        return []

    if not chunks:
        return []

    verified_citations = []

    for citation in raw_citations:

        marker = citation.get("marker", "")
        excerpt = citation.get("excerpt", "")

        chunk_index = _extract_chunk_index(marker)

        # Invalid citation marker
        if chunk_index is None:
            verified_citations.append(
                _build_invalid_citation(
                    marker=marker,
                    excerpt=excerpt,
                )
            )
            continue

        # Citation points outside retrieved chunk list
        if chunk_index >= len(chunks):
            verified_citations.append(
                _build_invalid_citation(
                    marker=marker,
                    excerpt=excerpt,
                )
            )
            continue

        chunk = chunks[chunk_index]

        print("\n==============================")
        print("Marker :", marker)
        print("Excerpt:")
        print(excerpt)

        print("\nChunk:")
        print(chunk["text"][:500])

        print("==============================")

        chunk_text = chunk.get("text", "")

        metadata = chunk.get("metadata", {})

        supported = _is_excerpt_supported(
            excerpt=excerpt,
            chunk_text=chunk_text,
        )
        print("\n" + "-" * 50)
        print(f"Marker     : {marker}")
        print(f"Excerpt    : {excerpt}")
        print(f"Supported  : {supported}")
        print(f"Chunk ID   : {chunk.get('id')}")
        print(f"Source Doc : {metadata.get('source_document')}")
        print("-" * 50)

        verified_citations.append(
            {
                "marker": marker,
                "chunk_id": chunk.get("id", ""),
                "source_document": metadata.get(
                    "source_document",
                    "unknown",
                ),
                "section_heading": metadata.get(
                    "section_heading"
                ),
                "supported": supported,
                "excerpt": excerpt,
            }
        )

    return verified_citations


def _extract_chunk_index(marker: str) -> int | None:
    """
    Convert a citation marker into a zero-based chunk index.

    Examples:

        [1] -> 0
        [2] -> 1
        [5] -> 4
    """

    match = re.fullmatch(
        r"\[(\d+)\]",
        marker.strip(),
    )

    if not match:
        return None

    citation_number = int(match.group(1))

    if citation_number <= 0:
        return None

    return citation_number - 1


def _is_excerpt_supported(
    excerpt: str,
    chunk_text: str,
) -> bool:
    """
    Determine whether a citation excerpt is supported by the
    referenced source chunk.

    Uses:

    1. Exact normalized substring matching
    2. Token overlap scoring

    No LLM call is required.
    """

    if not excerpt:
        return False

    if not chunk_text:
        return False

    normalized_excerpt = _normalize_text(excerpt)

    normalized_chunk = _normalize_text(chunk_text)

    if not normalized_excerpt:
        return False

    if not normalized_chunk:
        return False

    # --------------------------------------------------
    # Strategy 1: Exact normalized substring
    # --------------------------------------------------

    if normalized_excerpt in normalized_chunk:
        return True

    # --------------------------------------------------
    # Strategy 2: Token overlap
    # --------------------------------------------------

    excerpt_tokens = set(
        _tokenize(normalized_excerpt)
    )

    chunk_tokens = set(
        _tokenize(normalized_chunk)
    )

    if not excerpt_tokens:
        return False

    overlap = excerpt_tokens & chunk_tokens

    overlap_score = (
        len(overlap)
        / len(excerpt_tokens)
    )

    return (
        overlap_score
        >= CITATION_OVERLAP_THRESHOLD
    )


def _normalize_text(text: str) -> str:
    """
    Normalize text before citation comparison.

    Example:

        "Docker is a Platform."
                ↓
        "docker is a platform"
    """

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    return text.strip()


def _tokenize(text: str) -> list[str]:
    """
    Tokenize normalized text.

    Very common stop words are removed to prevent meaningless
    token overlap from inflating citation support scores.
    """

    stop_words = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "in",
        "on",
        "for",
        "with",
        "and",
        "or",
        "that",
        "this",
        "it",
        "as",
        "by",
        "from",
    }

    tokens = text.split()

    return [
        token
        for token in tokens
        if token not in stop_words
        and len(token) > 1
    ]


def _build_invalid_citation(
    marker: str,
    excerpt: str,
) -> dict:
    """
    Build a structured unsupported citation response for invalid
    or out-of-range citation markers.
    """

    return {
        "marker": marker,
        "chunk_id": "",
        "source_document": "unknown",
        "section_heading": None,
        "supported": False,
        "excerpt": excerpt,
    }