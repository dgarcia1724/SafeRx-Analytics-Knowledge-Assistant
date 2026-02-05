from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
from typing import Literal


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI API
    openai_api_key: str

    # Cohere API (for reranking) - optional
    cohere_api_key: str = ""

    # ChromaDB
    chroma_persist_dir: str = "./data/chroma_db"

    # BM25 Index persistence
    bm25_index_path: str = "./data/bm25_index.pkl"

    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Chunking Configuration
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval Configuration
    top_k_results: int = 5  # Final results returned
    retrieval_top_k: int = 20  # Candidates before reranking

    # Hybrid Search Configuration
    enable_hybrid_search: bool = True
    bm25_weight: float = 0.3  # Weight for BM25 in fusion (0.0-1.0)
    vector_weight: float = 0.7  # Weight for vector search in fusion
    rrf_k: int = 60  # RRF constant (typically 60)

    # Reranking Configuration
    enable_reranking: bool = True
    reranker_model: Literal["cohere", "cross-encoder"] = "cohere"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Tracing Configuration
    enable_tracing: bool = True
    trace_log_path: str = "./data/traces"
    max_traces_stored: int = 1000

    # CORS
    cors_origins: list[str] = ["http://localhost:3000"]

    # Collection name
    collection_name: str = "drug_safety_docs"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
