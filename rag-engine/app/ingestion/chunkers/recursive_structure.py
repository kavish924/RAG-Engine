"""
Structure-aware chunking: recursively splits on a priority list of
separators (section breaks -> paragraphs -> sentences -> words), only
falling through to a finer-grained separator when a piece is still too
large. This keeps semantically related text (e.g. a full paragraph)
together whenever it fits, unlike naive fixed-size windows.

Thin wrapper around LangChain's RecursiveCharacterTextSplitter so we get
a well-tested implementation, with a fallback pure-Python version if the
dependency isn't installed.
"""


def chunk_recursive_structure(text: str, max_chunk_size: int = 800, overlap: int = 100) -> list[str]:
    text = text.strip()
    if not text:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [c.strip() for c in splitter.split_text(text) if c.strip()]
    except ImportError:
        return _fallback_recursive_split(text, max_chunk_size, overlap)


def _fallback_recursive_split(text: str, max_chunk_size: int, overlap: int, separators=None) -> list[str]:
    separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split(piece: str, seps: list[str]) -> list[str]:
        if len(piece) <= max_chunk_size or not seps:
            return [piece] if piece.strip() else []

        sep, rest_seps = seps[0], seps[1:]
        parts = piece.split(sep) if sep else list(piece)

        results: list[str] = []
        buffer = ""
        for part in parts:
            candidate = (buffer + sep + part) if buffer else part
            if len(candidate) <= max_chunk_size:
                buffer = candidate
            else:
                if buffer:
                    results.extend(split(buffer, rest_seps) if len(buffer) > max_chunk_size else [buffer])
                buffer = part
        if buffer:
            results.extend(split(buffer, rest_seps) if len(buffer) > max_chunk_size else [buffer])
        return results

    chunks = [c.strip() for c in split(text, separators) if c.strip()]

    # apply simple char-overlap by stitching a tail of the previous chunk
    # onto the front of the next, mirroring LangChain's behavior
    if overlap <= 0 or len(chunks) < 2:
        return chunks

    overlapped = [chunks[0]]
    for chunk in chunks[1:]:
        tail = overlapped[-1][-overlap:]
        overlapped.append((tail + " " + chunk).strip())
    return overlapped
