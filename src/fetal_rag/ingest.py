"""Build retrieval indexes from a directory of paper PDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chunking import chunk_pages
from .embeddings import EmbeddingClient, make_embedding_client
from .pdf import iter_pdf_pages
from .references import load_manifest, match_reference_for_pdf
from .store import VectorStore
from .tfidf import TfidfStore
from .types import DocumentChunk


def collect_pdf_chunks(
    pdf_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    chunk_words: int = 450,
    overlap_words: int = 80,
) -> list[DocumentChunk]:
    pdf_root = Path(pdf_dir)
    pdf_paths = sorted(pdf_root.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_root}")

    references = load_manifest(manifest_path)
    chunks: list[DocumentChunk] = []
    for pdf_path in pdf_paths:
        reference = match_reference_for_pdf(pdf_path, references)
        source_id = str(reference.get("source_id") or pdf_path.stem)
        citation = str(reference.get("citation") or pdf_path.stem)
        chunks.extend(
            chunk_pages(
                iter_pdf_pages(pdf_path),
                source_id=source_id,
                source_title=citation,
                source_path=str(pdf_path),
                chunk_words=chunk_words,
                overlap_words=overlap_words,
            )
        )
    return chunks


def build_index(
    pdf_dir: str | Path,
    index_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    embedding_backend: str = "gemini",
    embedder: EmbeddingClient | None = None,
    chunk_words: int = 450,
    overlap_words: int = 80,
) -> VectorStore:
    chunks = collect_pdf_chunks(
        pdf_dir,
        manifest_path=manifest_path,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )
    client = embedder or make_embedding_client(embedding_backend)
    vectors = client.embed_documents([chunk.embedding_text() for chunk in chunks])
    build_config: dict[str, Any] = {
        "pdf_dir": str(pdf_dir),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "embedding_backend": embedding_backend,
    }
    return VectorStore.create(
        index_dir,
        chunks,
        vectors,
        embedding_model=client.model_name,
        build_config=build_config,
    )


def build_tfidf_index(
    pdf_dir: str | Path,
    index_dir: str | Path,
    *,
    manifest_path: str | Path | None = None,
    chunk_words: int = 450,
    overlap_words: int = 80,
    min_df: int = 1,
    max_df_ratio: float = 0.95,
    max_features: int | None = None,
    max_terms_per_chunk: int | None = None,
) -> TfidfStore:
    chunks = collect_pdf_chunks(
        pdf_dir,
        manifest_path=manifest_path,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )
    return TfidfStore.create(
        index_dir,
        chunks,
        min_df=min_df,
        max_df_ratio=max_df_ratio,
        max_features=max_features,
        max_terms_per_chunk=max_terms_per_chunk,
        build_config={
            "pdf_dir": str(pdf_dir),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
            "max_features": max_features,
        },
    )
