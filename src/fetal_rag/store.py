"""Small NumPy vector store with JSONL metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import normalize_vectors
from .types import DocumentChunk


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float
    rank: int


class VectorStore:
    """A disk-backed cosine-similarity store designed for small paper corpora."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.chunk_path = self.index_dir / "chunks.jsonl"
        self.vector_path = self.index_dir / "vectors.npy"
        self.meta_path = self.index_dir / "index_meta.json"
        self._chunks: list[DocumentChunk] | None = None
        self._vectors: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        index_dir: str | Path,
        chunks: list[DocumentChunk],
        vectors: np.ndarray,
        *,
        embedding_model: str,
        build_config: dict[str, Any] | None = None,
    ) -> "VectorStore":
        if len(chunks) == 0:
            raise ValueError("Cannot create an empty vector store")
        vectors = normalize_vectors(vectors)
        if vectors.shape[0] != len(chunks):
            raise ValueError("Vector row count must match chunk count")

        store = cls(index_dir)
        store.index_dir.mkdir(parents=True, exist_ok=True)
        store.chunk_path.write_text(
            "".join(json.dumps(chunk.to_json()) + "\n" for chunk in chunks),
            encoding="utf-8",
        )
        np.save(store.vector_path, vectors)
        store.meta_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "embedding_model": embedding_model,
                    "chunk_count": len(chunks),
                    "dimensions": int(vectors.shape[1]),
                    "build_config": build_config or {},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        store._chunks = chunks
        store._vectors = vectors
        return store

    @property
    def chunks(self) -> list[DocumentChunk]:
        if self._chunks is None:
            self._chunks = [
                DocumentChunk.from_json(json.loads(line))
                for line in self.chunk_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return self._chunks

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = normalize_vectors(np.load(self.vector_path))
        return self._vectors

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 6,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query = normalize_vectors(np.asarray(query_vector, dtype=np.float32))[0]
        scores = self.vectors @ query
        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results: list[SearchResult] = []
        for rank, index in enumerate(ranked_indices, start=1):
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            results.append(SearchResult(self.chunks[int(index)], score, rank))
        return results
