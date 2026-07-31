"""
Centralized application settings.

All configuration is loaded from environment variables (.env).

This project supports:
- Local or OpenAI embeddings
- Groq for LLM generation
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ==========================================================
    # LLM Configuration
    # ==========================================================

    llm_provider: Literal["groq"] = "groq"

    groq_api_key: str = Field(
        default="",
        alias="GROQ_API_KEY",
    )

    generation_model: str = Field(
        default="llama-3.1-8b-instant",
        alias="GENERATION_MODEL",
    )

    # ==========================================================
    # Embedding Configuration
    # ==========================================================

    embedding_provider: Literal["local", "openai"] = "local"

    # Local embedding model
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # OpenAI embedding model (used only if embedding_provider="openai")
    openai_embedding_model: str = "text-embedding-3-small"

    # Optional OpenAI API key (only required for OpenAI embeddings)
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
    )

    # ==========================================================
    # ChromaDB Configuration
    # ==========================================================

    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "rag_chunks"

    # ==========================================================
    # Retrieval Configuration
    # ==========================================================

    dense_top_k: int = 10
    sparse_top_k: int = 10

    rrf_dense_weight: float = 0.7
    rrf_sparse_weight: float = 0.3

    rerank_top_n: int = 5

    # ==========================================================
    # Confidence Configuration
    # ==========================================================

    confidence_threshold: float = 0.45

    # ==========================================================
    # FastAPI Configuration
    # ==========================================================

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ==========================================================
    # Pydantic Settings
    # ==========================================================

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()