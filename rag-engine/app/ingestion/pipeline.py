"""
Orchestrates the full ingestion flow:
  load -> chunk (configurable strategy) -> embed -> dedup -> store
    -> both ChromaDB (dense) and BM25 index (sparse), kept in sync.
"""
import hashlib
from typing import Literal

from app.ingestion.chunkers.fixed_size import chunk_fixed_size
from app.ingestion.chunkers.recursive_structure import chunk_recursive_structure
from app.ingestion.chunkers.semantic import chunk_semantic
from app.ingestion.dedup import filter_duplicates
from app.ingestion.embeddings import embed_texts
from app.ingestion.loaders import load_document
from app.storage.bm25_store import BM25Store
from app.storage.vector_store import VectorStore

ChunkingStrategy = Literal["fixed_size", "recursive_structure", "semantic"]

_CHUNKERS = {
    "fixed_size": chunk_fixed_size,
    "recursive_structure": chunk_recursive_structure,
    "semantic": chunk_semantic,
}


def _chunk_id(source_file: str, strategy: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source_file}:{strategy}:{index}:{text[:50]}".encode()).hexdigest()[:12]
    return f"{strategy}-{digest}"


def run_ingestion(
    file_paths: list[str],
    strategy: ChunkingStrategy = "recursive_structure",
    vector_store: VectorStore | None = None,
    bm25_store: BM25Store | None = None,
    dedup_threshold: float = 0.95,
) -> dict:
    """
    Runs load -> chunk -> embed -> dedup -> store for each file.
    Returns a summary dict: {"chunks_created": int, "duplicates_skipped": int, "files_processed": list[str]}
    """
    vector_store = vector_store or VectorStore()
    bm25_store = bm25_store or BM25Store()
    chunk_fn = _CHUNKERS[strategy]

    all_chunk_texts: list[str] = []
    all_chunk_metadatas: list[dict] = []
    files_processed: list[str] = []

    for file_path in file_paths:
        blocks = load_document(file_path)
        files_processed.append(file_path)

        for block in blocks:
            pieces = chunk_fn(block.text)
            for i, piece in enumerate(pieces):
                all_chunk_texts.append(piece)
                all_chunk_metadatas.append({
                    "source_document": block.source_file,
                    "chunk_index": i,
                    "section_heading": block.section_heading,
                    "page_number": block.page_number,
                    "chunking_strategy": strategy,
                    "character_count": len(piece),
                })

    if not all_chunk_texts:
        return {"chunks_created": 0, "duplicates_skipped": 0, "files_processed": files_processed}

    # embed everything up front so dedup can compare against real vectors
    new_embeddings = embed_texts(all_chunk_texts)

    existing_embeddings = vector_store.get_all_embeddings()
    keep_indices, duplicate_indices = filter_duplicates(
        new_embeddings, existing_embeddings, threshold=dedup_threshold
    )

    chunks_to_store = []
    for i in keep_indices:
        chunk_id = _chunk_id(
            all_chunk_metadatas[i]["source_document"], strategy, all_chunk_metadatas[i]["chunk_index"], all_chunk_texts[i]
        )
        chunks_to_store.append({
            "id": chunk_id,
            "text": all_chunk_texts[i],
            "embedding": new_embeddings[i],
            "metadata": all_chunk_metadatas[i],
        })

    vector_store.upsert_chunks(chunks_to_store)
    bm25_store.add_chunks([{"id": c["id"], "text": c["text"], "metadata": c["metadata"]} for c in chunks_to_store])

    return {
        "chunks_created": len(chunks_to_store),
        "duplicates_skipped": len(duplicate_indices),
        "files_processed": files_processed,
    }
