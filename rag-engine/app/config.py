"""
Centralized application settings.

All configuration is loaded from environment variables (.env).

This project supports:
- Local or OpenAI embeddings
- Anthropic, OpenAI or Ollama for generation
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):

    # ==========================================================
    # LLM Configuration
    # ==========================================================

    llm_provider: Literal["anthropic", "openai", "ollama"] = "anthropic"

    generation_model: str = "claude-sonnet-4-6"

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # ==========================================================
    # Embedding Configuration
    # ==========================================================

    embedding_provider: Literal["local", "openai"] = "local"

    # Local embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # OpenAI embedding model
    openai_embedding_model: str = "text-embedding-3-small"

    # ==========================================================
    # ChromaDB
    # ==========================================================

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "rag_chunks"

    # ==========================================================
    # Retrieval Settings
    # ==========================================================

    dense_top_k: int = 10
    sparse_top_k: int = 10

    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3

    rerank_top_n: int = 5

    # ==========================================================
    # Confidence Settings
    # ==========================================================

    confidence_threshold: float = 0.45

    # ==========================================================
    # API Settings
    # ==========================================================

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ==========================================================
    # Pydantic Settings
    # ==========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
