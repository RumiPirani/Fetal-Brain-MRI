"""Disk-backed sklearn TF-IDF store for lexical retrieval."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .store import SearchResult
from .types import DocumentChunk


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+-]{1,}")
STOPWORDS = {
    "about",
    "after",
    "also",
    "among",
    "and",
    "are",
    "based",
    "been",
    "between",
    "both",
    "but",
    "can",
    "during",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "into",
    "may",
    "more",
    "not",
    "our",
    "patients",
    "study",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "using",
    "was",
    "were",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS and len(token) <= 48
    ]


class TfidfStore:
    """Disk-backed sklearn TF-IDF index with no service or model dependency."""

    def __init__(self, index_dir: str | Path) -> None:
        self.index_dir = Path(index_dir)
        self.chunk_path = self.index_dir / "chunks.jsonl"
        self.vectorizer_path = self.index_dir / "tfidf_vectorizer.joblib"
        self.matrix_path = self.index_dir / "tfidf_matrix.npz"
        self.meta_path = self.index_dir / "tfidf_meta.json"
        self._chunks: list[DocumentChunk] | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix: sparse.csr_matrix | None = None

    @classmethod
    def create(
        cls,
        index_dir: str | Path,
        chunks: list[DocumentChunk],
        *,
        min_df: int = 1,
        max_df_ratio: float = 0.95,
        max_features: int | None = None,
        max_terms_per_chunk: int | None = None,
        build_config: dict[str, Any] | None = None,
    ) -> "TfidfStore":
        if not chunks:
            raise ValueError("Cannot create an empty TF-IDF store")

        vectorizer = TfidfVectorizer(
            analyzer=tokenize,
            min_df=min_df,
            max_df=max_df_ratio,
            max_features=max_features,
            norm="l2",
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(chunk.embedding_text() for chunk in chunks)
        if matrix.shape[1] == 0:
            raise ValueError("No TF-IDF vocabulary terms survived filtering")
        matrix = matrix.tocsr()

        store = cls(index_dir)
        store.index_dir.mkdir(parents=True, exist_ok=True)
        store.chunk_path.write_text(
            "".join(json.dumps(chunk.to_json()) + "\n" for chunk in chunks),
            encoding="utf-8",
        )
        joblib.dump(vectorizer, store.vectorizer_path)
        sparse.save_npz(store.matrix_path, matrix)

        legacy_vector_path = store.index_dir / "tfidf_vectors.jsonl"
        if legacy_vector_path.exists():
            legacy_vector_path.unlink()

        store.meta_path.write_text(
            json.dumps(
                {
                    "format_version": 2,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "retrieval_backend": "tfidf",
                    "tfidf_implementation": "sklearn",
                    "vectorizer_class": (
                        "sklearn.feature_extraction.text.TfidfVectorizer"
                    ),
                    "sklearn_version": sklearn.__version__,
                    "vectorizer_file": store.vectorizer_path.name,
                    "matrix_file": store.matrix_path.name,
                    "chunk_count": len(chunks),
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "min_df": min_df,
                    "max_df_ratio": max_df_ratio,
                    "max_features": max_features,
                    "sublinear_tf": True,
                    "norm": "l2",
                    "deprecated_max_terms_per_chunk": max_terms_per_chunk,
                    "build_config": build_config or {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        store._chunks = chunks
        store._vectorizer = vectorizer
        store._matrix = matrix
        return store

    @property
    def metadata(self) -> dict[str, Any]:
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    @property
    def vectorizer(self) -> TfidfVectorizer:
        if self._vectorizer is None:
            self._vectorizer = joblib.load(self.vectorizer_path)
        return self._vectorizer

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
    def matrix(self) -> sparse.csr_matrix:
        if self._matrix is None:
            self._matrix = sparse.load_npz(self.matrix_path).tocsr()
        return self._matrix

    @property
    def vectors(self) -> sparse.csr_matrix:
        """Return the trained chunk matrix for compatibility with older callers."""

        return self.matrix

    def search(
        self,
        query: str,
        *,
        top_k: int = 6,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []

        scores = (self.matrix @ query_vector.T).toarray().ravel()
        scored: list[tuple[int, float]] = []
        for index, score in enumerate(scores):
            score_value = float(score)
            if min_score is None or score_value >= min_score:
                scored.append((index, score_value))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return [
            SearchResult(self.chunks[index], float(score), rank)
            for rank, (index, score) in enumerate(scored[:top_k], start=1)
        ]
