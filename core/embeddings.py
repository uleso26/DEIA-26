from __future__ import annotations

import os
from functools import lru_cache
from math import sqrt
from typing import Any

from tools.ollama_client import OllamaClient


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_rows(rows: list[list[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in rows:
        magnitude = sqrt(sum(value * value for value in row))
        if magnitude <= 0:
            normalized.append([0.0 for _ in row])
            continue
        normalized.append([float(value / magnitude) for value in row])
    return normalized


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, local_only: bool) -> Any:
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name, local_files_only=local_only)


def _embed_with_sentence_transformers(
    texts: list[str],
    model_name: str,
    *,
    local_only: bool,
) -> tuple[list[list[float]] | None, dict[str, str]]:
    try:
        model = _load_sentence_transformer(model_name, local_only)
        vectors = model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in vectors.tolist()], {
            "embedding_provider": "sentence-transformers",
            "embedding_model": model_name,
        }
    except Exception as exc:
        return None, {
            "embedding_provider": "sentence-transformers",
            "embedding_model": model_name,
            "embedding_error": str(exc),
        }


def _embed_with_ollama(texts: list[str], model_name: str) -> tuple[list[list[float]] | None, dict[str, str]]:
    client = OllamaClient()
    vectors = client.embed(
        texts,
        model_env="OLLAMA_EMBED_MODEL",
        default_model=model_name,
        timeout_seconds=30,
    )
    if vectors:
        return _normalize_rows(vectors), {
            "embedding_provider": "ollama",
            "embedding_model": model_name,
        }
    return None, {
        "embedding_provider": "ollama",
        "embedding_model": model_name,
        "embedding_error": "Ollama embeddings unavailable",
    }


def embed_texts(
    texts: list[str],
    *,
    provider: str | None = None,
    model_name: str | None = None,
) -> tuple[list[list[float]] | None, dict[str, str]]:
    requested_provider = (provider or os.getenv("EMBEDDING_PROVIDER", "auto")).strip().lower()
    requested_model = model_name or os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    fallback_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    allow_download = _env_flag("EMBEDDING_ALLOW_DOWNLOAD", False)
    auto_try_sentence_transformers = _env_flag("EMBEDDING_AUTO_TRY_SENTENCE_TRANSFORMERS", False)

    if requested_provider == "ollama":
        vectors, metadata = _embed_with_ollama(texts, requested_model)
        if vectors:
            return vectors, metadata
        return None, metadata

    if requested_provider == "sentence-transformers":
        vectors, metadata = _embed_with_sentence_transformers(
            texts,
            requested_model,
            local_only=not allow_download,
        )
        if vectors:
            return vectors, metadata
        return _embed_with_ollama(texts, fallback_model)

    vectors, metadata = _embed_with_ollama(texts, fallback_model)
    if vectors:
        return vectors, metadata
    if auto_try_sentence_transformers:
        return _embed_with_sentence_transformers(texts, requested_model, local_only=True)
    return None, metadata
