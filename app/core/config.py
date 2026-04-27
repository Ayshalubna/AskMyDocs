"""
Application configuration using Pydantic Settings.
All values can be overridden via environment variables.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "RAG Document Q&A"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # API
    API_HOST: str = Field(default="0.0.0.0", env="API_HOST")
    API_PORT: int = Field(default=8000, env="API_PORT")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "*"],
        env="ALLOWED_ORIGINS",
    )

    # Storage
    UPLOAD_DIR: Path = Field(default=Path("./storage/uploads"), env="UPLOAD_DIR")
    VECTOR_STORE_DIR: Path = Field(default=Path("./storage/vectorstore"), env="VECTOR_STORE_DIR")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx", ".md", ".csv", ".html"]

    # Embeddings
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )
    EMBEDDING_DEVICE: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")

    # Vector Store (FAISS)
    FAISS_INDEX_TYPE: str = Field(default="IndexFlatIP", env="FAISS_INDEX_TYPE")
    TOP_K_RESULTS: int = Field(default=5, env="TOP_K_RESULTS")
    SIMILARITY_THRESHOLD: float = Field(default=0.3, env="SIMILARITY_THRESHOLD")
    MMR_LAMBDA: float = Field(default=0.5, env="MMR_LAMBDA")  # Maximal marginal relevance

    # Text Chunking
    CHUNK_SIZE: int = Field(default=512, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=64, env="CHUNK_OVERLAP")
    CHUNK_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]

    # LLM (Ollama)
    OLLAMA_BASE_URL: str = Field(default="http://ollama:11434", env="OLLAMA_BASE_URL")
    OLLAMA_MODEL: str = Field(default="mistral", env="OLLAMA_MODEL")
    LLM_TEMPERATURE: float = Field(default=0.1, env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(default=1024, env="LLM_MAX_TOKENS")
    LLM_TIMEOUT: int = Field(default=120, env="LLM_TIMEOUT")

    # Fallback LLM (HuggingFace - when Ollama unavailable)
    USE_HF_FALLBACK: bool = Field(default=True, env="USE_HF_FALLBACK")
    HF_MODEL: str = Field(default="google/flan-t5-base", env="HF_MODEL")

    # Cache
    ENABLE_CACHE: bool = Field(default=True, env="ENABLE_CACHE")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    MAX_CACHE_SIZE: int = Field(default=1000, env="MAX_CACHE_SIZE")

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=60, env="RATE_LIMIT_WINDOW")

    # Session
    SESSION_TTL_HOURS: int = Field(default=24, env="SESSION_TTL_HOURS")
    MAX_HISTORY_LENGTH: int = Field(default=10, env="MAX_HISTORY_LENGTH")

    @validator("UPLOAD_DIR", "VECTOR_STORE_DIR", pre=True)
    def create_dirs(cls, v):
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
