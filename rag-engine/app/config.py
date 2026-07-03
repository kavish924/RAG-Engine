"""
Centralized settings, loaded from environment variables (see .env.example).
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "anthropic"          # "anthropic" | "openai"
    generation_model: str = "claude-sonnet-4-6"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "rag_chunks"

    # Retrieval tuning (Phase 2)
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3
    rerank_top_n: int = 5

    # Trust layer (Phase 3)
    confidence_threshold: float = 0.45

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
