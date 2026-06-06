"""Command line interface for the fetal MRI RAG helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .fetch import fetch_pmc_pdfs, fetch_reference_pdfs, write_fetch_report
from .ingest import build_index, build_tfidf_index
from .rag import RagEngine, TfidfRagEngine
from .references import write_manifest_from_spec
from .store import VectorStore


def _cmd_extract_spec(args: argparse.Namespace) -> None:
    write_manifest_from_spec(args.spec, args.out)
    print(f"Wrote reference manifest to {args.out}")


def _cmd_build(args: argparse.Namespace) -> None:
    if args.retrieval_backend == "tfidf":
        store = build_tfidf_index(
            args.pdf_dir,
            args.index_dir,
            manifest_path=args.manifest,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
            min_df=args.min_df,
            max_df_ratio=args.max_df_ratio,
            max_features=args.max_features,
            max_terms_per_chunk=args.max_terms_per_chunk,
        )
        print(
            f"Built sklearn TF-IDF index at {store.index_dir} with "
            f"{store.metadata['chunk_count']} chunks and "
            f"{store.metadata['vocabulary_size']} terms."
        )
    else:
        store = build_index(
            args.pdf_dir,
            args.index_dir,
            manifest_path=args.manifest,
            embedding_backend=args.retrieval_backend,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
        print(
            f"Built vector index at {store.index_dir} with "
            f"{store.metadata['chunk_count']} chunks using "
            f"{store.metadata['embedding_model']}."
        )


def _cmd_fetch_pmc(args: argparse.Namespace) -> None:
    results = fetch_pmc_pdfs(
        args.manifest,
        args.pdf_dir,
        overwrite=args.overwrite,
        timeout=args.timeout,
        limit=args.limit,
    )
    write_fetch_report(results, args.report)
    downloaded = sum(1 for result in results if result.status == "downloaded")
    failed = sum(1 for result in results if result.status == "failed")
    skipped = sum(1 for result in results if result.status == "skipped")
    print(
        f"PMC fetch complete: {downloaded} downloaded, {skipped} skipped, "
        f"{failed} failed. Report: {args.report}"
    )


def _cmd_fetch_references(args: argparse.Namespace) -> None:
    results = fetch_reference_pdfs(
        args.manifest,
        args.pdf_dir,
        overwrite=args.overwrite,
        timeout=args.timeout,
        limit=args.limit,
    )
    write_fetch_report(results, args.report)
    counts = {
        status: sum(1 for result in results if result.status == status)
        for status in ("downloaded", "skipped", "duplicate", "failed")
    }
    print(
        "Reference fetch complete: "
        f"{counts['downloaded']} downloaded, {counts['skipped']} skipped, "
        f"{counts['duplicate']} duplicates, {counts['failed']} failed. "
        f"Report: {args.report}"
    )


def _cmd_ask(args: argparse.Namespace) -> None:
    calculator_data = None
    if args.calculator_data:
        calculator_data = Path(args.calculator_data).read_text(encoding="utf-8")
    engine = _load_engine(args.index_dir, args.retrieval_backend)
    answer = engine.answer(
        args.question,
        calculator_data=calculator_data,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    print(answer.answer)
    if args.show_context:
        print("\nRetrieved context:")
        for result in answer.contexts:
            print(
                f"- C{result.rank}: {result.chunk.citation_label()} "
                f"score={result.score:.3f}"
            )


def _cmd_retrieve(args: argparse.Namespace) -> None:
    engine = _load_engine(args.index_dir, args.retrieval_backend)
    results = engine.retrieve(args.question, top_k=args.top_k, min_score=args.min_score)
    print(
        json.dumps(
            [
                {
                    "rank": result.rank,
                    "score": result.score,
                    "citation": result.chunk.citation_label(),
                    "source_title": result.chunk.source_title,
                    "text": result.chunk.text[: args.max_chars],
                }
                for result in results
            ],
            indent=2,
        )
    )


def _cmd_health(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    if (index_dir / "tfidf_meta.json").exists():
        from .tfidf import TfidfStore

        meta = TfidfStore(index_dir).metadata
        print(
            f"sklearn TF-IDF index OK: {meta['chunk_count']} chunks, "
            f"{meta['vocabulary_size']} terms"
        )
    else:
        store = VectorStore(index_dir)
        meta = store.metadata
        print(
            f"Vector index OK: {meta['chunk_count']} chunks, "
            f"{meta['dimensions']} dimensions, model={meta['embedding_model']}"
        )


def _load_engine(index_dir: str | Path, retrieval_backend: str):
    if retrieval_backend == "tfidf":
        return TfidfRagEngine.from_index(index_dir)
    return RagEngine.from_index(index_dir, embedding_backend=retrieval_backend)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetal brain MRI PDF RAG helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract-spec", help="Build a manifest from SPEC.md")
    extract.add_argument("--spec", default="../FetalBrainMRI/SPEC.md")
    extract.add_argument("--out", default="data/reference_manifest.json")
    extract.set_defaults(func=_cmd_extract_spec)

    build = subparsers.add_parser("build", help="Build a retrieval index from PDFs")
    build.add_argument("--pdf-dir", default="papers/corpus")
    build.add_argument("--index-dir", default="vector_db/fetal_mri")
    build.add_argument("--manifest", default="data/reference_manifest.json")
    build.add_argument(
        "--retrieval-backend",
        choices=["tfidf", "gemini", "hash"],
        default="tfidf",
    )
    build.add_argument(
        "--embedding-backend",
        choices=["tfidf", "gemini", "hash"],
        help="Deprecated alias for --retrieval-backend.",
    )
    build.add_argument("--chunk-words", type=int, default=450)
    build.add_argument("--overlap-words", type=int, default=80)
    build.add_argument("--min-df", type=int, default=1)
    build.add_argument("--max-df-ratio", type=float, default=0.95)
    build.add_argument("--max-features", type=int)
    build.add_argument(
        "--max-terms-per-chunk",
        type=int,
        help="Deprecated; sklearn TF-IDF uses --max-features for vocabulary limits.",
    )
    build.set_defaults(func=_cmd_build)

    fetch = subparsers.add_parser(
        "fetch-pmc",
        help="Download open-access PMC PDFs from a manifest",
    )
    fetch.add_argument("--manifest", default="data/reference_manifest.json")
    fetch.add_argument("--pdf-dir", default="papers")
    fetch.add_argument("--report", default="data/pmc_fetch_report.json")
    fetch.add_argument("--timeout", type=int, default=60)
    fetch.add_argument("--limit", type=int)
    fetch.add_argument("--overwrite", action="store_true")
    fetch.set_defaults(func=_cmd_fetch_pmc)

    fetch_refs = subparsers.add_parser(
        "fetch-references",
        help="Best-effort download of open PDFs from all manifest references",
    )
    fetch_refs.add_argument("--manifest", default="data/reference_manifest.json")
    fetch_refs.add_argument("--pdf-dir", default="papers")
    fetch_refs.add_argument("--report", default="data/reference_fetch_report.json")
    fetch_refs.add_argument("--timeout", type=int, default=60)
    fetch_refs.add_argument("--limit", type=int)
    fetch_refs.add_argument("--overwrite", action="store_true")
    fetch_refs.set_defaults(func=_cmd_fetch_references)

    ask = subparsers.add_parser("ask", help="Ask Gemini using retrieved paper chunks")
    ask.add_argument("question")
    ask.add_argument("--index-dir", default="vector_db/fetal_mri")
    ask.add_argument(
        "--retrieval-backend",
        choices=["tfidf", "gemini", "hash"],
        default="tfidf",
    )
    ask.add_argument("--embedding-backend", choices=["tfidf", "gemini", "hash"])
    ask.add_argument("--calculator-data")
    ask.add_argument("--top-k", type=int, default=6)
    ask.add_argument("--min-score", type=float)
    ask.add_argument("--show-context", action="store_true")
    ask.set_defaults(func=_cmd_ask)

    retrieve = subparsers.add_parser("retrieve", help="Return retrieved chunks only")
    retrieve.add_argument("question")
    retrieve.add_argument("--index-dir", default="vector_db/fetal_mri")
    retrieve.add_argument(
        "--retrieval-backend",
        choices=["tfidf", "gemini", "hash"],
        default="tfidf",
    )
    retrieve.add_argument("--embedding-backend", choices=["tfidf", "gemini", "hash"])
    retrieve.add_argument("--top-k", type=int, default=6)
    retrieve.add_argument("--min-score", type=float)
    retrieve.add_argument("--max-chars", type=int, default=600)
    retrieve.set_defaults(func=_cmd_retrieve)

    health = subparsers.add_parser("health", help="Validate index files load")
    health.add_argument("--index-dir", default="vector_db/fetal_mri")
    health.set_defaults(func=_cmd_health)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "embedding_backend", None):
        args.retrieval_backend = args.embedding_backend
    args.func(args)


if __name__ == "__main__":
    main()
