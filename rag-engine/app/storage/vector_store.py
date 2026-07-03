"""
Thin wrapper around ChromaDB: collection setup, upsert, and query,
storing metadata (source document, chunk index, section heading,
chunking strategy, character count) alongside each embedding.

Uses a local persistent client by default (good for the seed script /
local dev); point CHROMA_HOST/PORT at a running server in docker-compose
for the containerized setup by using `use_http=True`.
"""


class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_chunks",
        host: str | None = None,
        port: int | None = None,
        use_http: bool = False,
        persist_directory: str = "./chroma_data",
    ):
        import chromadb

        if use_http:
            self.client = chromadb.HttpClient(host=host, port=port)
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """
        Each chunk dict must have: id, text, embedding, and a metadata
        dict with source_document, chunk_index, section_heading (optional),
        chunking_strategy, character_count.
        """
        if not chunks:
            return

        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=[c["embedding"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[self._sanitize_metadata(c["metadata"]) for c in chunks],
        )

    def query(self, query_embedding: list[float], top_k: int = 10) -> list[dict]:
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return self._format_results(results)

    def get_all_embeddings(self) -> list[list[float]]:
        """Used by the dedup step to check new chunks against everything already stored."""
        data = self.collection.get(include=["embeddings"])
        embeddings = data.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return []
        return [list(e) for e in embeddings]

    def list_documents(self) -> list[dict]:
        """Groups stored chunks by source_document for GET /v1/documents."""
        data = self.collection.get(include=["metadatas"])
        summary: dict[str, dict] = {}
        for meta in data.get("metadatas", []):
            source = meta.get("source_document", "unknown")
            entry = summary.setdefault(source, {"source_file": source, "num_chunks": 0, "chunking_strategies_used": set()})
            entry["num_chunks"] += 1
            entry["chunking_strategies_used"].add(meta.get("chunking_strategy", "unknown"))
        for entry in summary.values():
            entry["chunking_strategies_used"] = sorted(entry["chunking_strategies_used"])
        return list(summary.values())

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:
        """Chroma metadata values must be str/int/float/bool — coerce anything else."""
        clean = {}
        for k, v in metadata.items():
            if v is None:
                continue
            clean[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
        return clean

    @staticmethod
    def _format_results(raw: dict) -> list[dict]:
        formatted = []
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        for i in range(len(ids)):
            formatted.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                # Chroma returns cosine *distance*; convert to a similarity score
                "score": 1 - distances[i] if distances else None,
            })
        return formatted
