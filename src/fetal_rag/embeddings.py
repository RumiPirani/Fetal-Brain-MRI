"""Embedding backends for Gemini and local smoke tests."""

from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Protocol

import numpy as np

from .config import get_api_key, get_embedding_model


class EmbeddingClient(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        ...

    def embed_query(self, text: str) -> np.ndarray:
        ...


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _embedding_values(item: object) -> list[float]:
    if hasattr(item, "values"):
        return list(getattr(item, "values"))
    if isinstance(item, dict):
        values = item.get("values")
        if values is not None:
            return list(values)
    raise TypeError(f"Cannot read embedding values from {type(item)!r}")


class GeminiEmbeddingClient:
    """Gemini embeddings using the official google-genai SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_name: str | None = None,
        batch_size: int = 16,
        request_delay_seconds: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key or get_api_key())
        self.types = types
        self.model_name = model_name or get_embedding_model()
        self.batch_size = int(os.getenv("GEMINI_EMBEDDING_BATCH_SIZE", batch_size))
        self.request_delay_seconds = float(
            os.getenv(
                "GEMINI_EMBEDDING_DELAY_SECONDS",
                0 if request_delay_seconds is None else request_delay_seconds,
            )
        )
        self.max_retries = int(
            os.getenv(
                "GEMINI_EMBEDDING_MAX_RETRIES",
                6 if max_retries is None else max_retries,
            )
        )

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        text = str(error)
        return (
            "429" in text
            or "RESOURCE_EXHAUSTED" in text
            or "503" in text
            or "UNAVAILABLE" in text
        )

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            for attempt in range(self.max_retries + 1):
                try:
                    response = self.client.models.embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=self.types.EmbedContentConfig(task_type=task_type),
                    )
                    break
                except Exception as error:
                    if attempt >= self.max_retries or not self._is_retryable(error):
                        raise
                    time.sleep(min(60.0, 2.0 * (2**attempt)))
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None:
                embedding = getattr(response, "embedding", None)
                embeddings = [embedding] if embedding is not None else []
            rows.extend(_embedding_values(item) for item in embeddings)
            if self.request_delay_seconds and start + self.batch_size < len(texts):
                time.sleep(self.request_delay_seconds)
        return normalize_vectors(np.array(rows, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class HashEmbeddingClient:
    """Tiny deterministic embedding backend for tests and offline smoke checks."""

    model_name = "local-hash-embedding"

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        return normalize_vectors(vector)[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return normalize_vectors(np.vstack([self._embed_one(text) for text in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed_one(text)


def make_embedding_client(backend: str = "gemini") -> EmbeddingClient:
    if backend == "gemini":
        return GeminiEmbeddingClient()
    if backend == "hash":
        return HashEmbeddingClient()
    raise ValueError(f"Unknown embedding backend: {backend}")
