# Import the libraries helpers and shared models needed in this file
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from core.logging_utils import get_logger
from core.runtime_utils import env_flag


logger = get_logger(__name__)


# Define the Ollama client wrapper used for local model calls
class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        configured_base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        self.base_url: str = configured_base_url.rstrip("/")

    @staticmethod
    @lru_cache(maxsize=4)
    def _probe(base_url: str) -> bool:
        # Probe once so normal queries do not keep paying the local health-check cost
        try:
            import requests
        except Exception as exc:
            logger.warning("Ollama health probe disabled because requests is unavailable: %s", exc)
            return False

        try:
            response = requests.get(f"{base_url}/api/tags", timeout=0.35)
            return response.ok
        except Exception as exc:
            logger.warning("Ollama health probe failed for %s: %s", base_url, exc)
            return False

    def available(self) -> bool:
        return self._probe(self.base_url)

    def enabled(self, mode: str) -> bool:
        raw_value = os.getenv(f"USE_OLLAMA_{mode.upper()}")
        if raw_value is None:
            return self.available()
        normalized = raw_value.strip().lower()
        if normalized == "auto":
            return self.available()
        return env_flag(f"USE_OLLAMA_{mode.upper()}", False)

    def generate(
        self,
        prompt: str,
        system: str,
        model_env: str,
        default_model: str,
        timeout_seconds: int = 20,
    ) -> str | None:
        model_name = os.getenv(model_env, default_model)
        keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "15m")
        max_tokens = os.getenv("OLLAMA_MAX_TOKENS")
        options: dict[str, Any] = {"temperature": 0}
        if max_tokens and max_tokens.isdigit():
            options["num_predict"] = int(max_tokens)

        try:
            import requests
        except Exception as exc:
            logger.warning("Ollama generation unavailable because requests is missing: %s", exc)
            return None

        retry_timeout = max(timeout_seconds * 3, timeout_seconds + 45)
        for current_timeout in [timeout_seconds, retry_timeout]:
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model_name,
                        "system": system,
                        "prompt": prompt,
                        "stream": False,
                        "keep_alive": keep_alive,
                        "options": options,
                    },
                    timeout=current_timeout,
                )
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
                text = str(payload.get("response", "")).strip()
                return text or None
            except requests.Timeout:
                continue
            except Exception as exc:
                logger.warning("Ollama generation failed for model %s: %s", model_name, exc)
                return None
        logger.warning("Ollama generation timed out for model %s after retries.", model_name)
        return None

    def embed(
        self,
        texts: list[str],
        model_env: str,
        default_model: str,
        timeout_seconds: int = 30,
    ) -> list[list[float]] | None:
        if not texts:
            return []
        model_name = os.getenv(model_env, default_model)
        try:
            import requests
        except Exception as exc:
            logger.warning("Ollama embedding unavailable because requests is missing: %s", exc)
            return None

        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model_name, "input": texts},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, list) and embeddings:
                return [[float(value) for value in vector] for vector in embeddings]
            legacy_embedding = payload.get("embedding")
            if isinstance(legacy_embedding, list):
                return [[float(value) for value in legacy_embedding]]
            return None
        except Exception as exc:
            logger.warning("Ollama embedding failed for model %s: %s", model_name, exc)
            return None
