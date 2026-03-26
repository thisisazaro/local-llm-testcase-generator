from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw.strip() if raw else default


@dataclass(frozen=True)
class Settings:
    app_name: str
    data_dir: Path
    raw_dir: Path
    chroma_dir: Path
    chroma_collection: str

    chunk_max_chars: int
    chunk_overlap: int
    min_chunk_chars: int

    retrieval_top_k: int

    embedding_provider: str
    embedding_model: str
    embedding_dim: int

    llm_provider: str
    llm_model_name: str
    llm_temperature: float
    llm_max_tokens: int
    llm_timeout_seconds: int

    ollama_base_url: str
    openai_compat_base_url: str
    openai_compat_api_key: str

    allow_mock_fallback: bool



def load_settings() -> Settings:
    data_dir = (PROJECT_ROOT / _env_str("DATA_DIR", "data")).resolve()
    raw_dir = data_dir / "raw"
    chroma_dir = data_dir / "chroma"

    raw_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        app_name=_env_str("APP_NAME", "Local LLM Test Case Generator"),
        data_dir=data_dir,
        raw_dir=raw_dir,
        chroma_dir=chroma_dir,
        chroma_collection=_env_str("CHROMA_COLLECTION", "documents"),
        chunk_max_chars=_env_int("CHUNK_MAX_CHARS", 1600),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 220),
        min_chunk_chars=_env_int("MIN_CHUNK_CHARS", 120),
        retrieval_top_k=_env_int("RETRIEVAL_TOP_K", 8),
        embedding_provider=_env_str("EMBEDDING_PROVIDER", "hash"),
        embedding_model=_env_str(
            "EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        embedding_dim=_env_int("EMBEDDING_DIM", 384),
        llm_provider=_env_str("LLM_PROVIDER", "mock"),
        llm_model_name=_env_str("LLM_MODEL_NAME", "qwen2.5:7b-instruct"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 2200),
        llm_timeout_seconds=_env_int("LLM_TIMEOUT_SECONDS", 90),
        ollama_base_url=_env_str("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_compat_base_url=_env_str("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1"),
        openai_compat_api_key=_env_str("OPENAI_COMPAT_API_KEY", ""),
        allow_mock_fallback=_env_bool("ALLOW_MOCK_FALLBACK", True),
    )


SETTINGS = load_settings()
