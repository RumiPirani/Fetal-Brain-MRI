"""Minimal standalone web app for building and querying the RAG index."""

from __future__ import annotations

import html
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse

from .ingest import build_index, build_tfidf_index
from .rag import RagEngine, TfidfRagEngine
from .store import VectorStore
from .tfidf import TfidfStore


APP_TITLE = "Fetal Brain MRI Literature RAG"
INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "vector_db/fetal_mri"))
UPLOAD_DIR = Path(os.getenv("RAG_UPLOAD_DIR", "papers"))
MANIFEST_PATH = Path(os.getenv("RAG_MANIFEST_PATH", "data/reference_manifest.json"))
RETRIEVAL_BACKEND = os.getenv("RAG_RETRIEVAL_BACKEND", "tfidf")

app = FastAPI(title=APP_TITLE)


def _page(body: str) -> HTMLResponse:
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f8fb;
      color: #17202a;
    }}
    body {{ margin: 0; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 28px 18px 48px; }}
    h1 {{ font-size: 28px; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ font-size: 18px; margin: 28px 0 10px; letter-spacing: 0; }}
    p {{ line-height: 1.55; margin: 0 0 14px; }}
    form {{
      background: #ffffff;
      border: 1px solid #d9dee8;
      border-radius: 8px;
      padding: 16px;
      margin: 14px 0;
    }}
    label {{ display: block; font-weight: 650; margin: 12px 0 6px; }}
    input[type="file"], textarea, input[type="number"] {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #b9c2d0;
      border-radius: 6px;
      padding: 10px;
      font: inherit;
      background: #fff;
    }}
    textarea {{ min-height: 110px; resize: vertical; }}
    button {{
      margin-top: 14px;
      border: 0;
      border-radius: 6px;
      background: #2457c5;
      color: #fff;
      padding: 10px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }}
    button:hover {{ background: #1d49a8; }}
    .status, .answer {{
      background: #ffffff;
      border: 1px solid #d9dee8;
      border-radius: 8px;
      padding: 16px;
      white-space: pre-wrap;
    }}
    .muted {{ color: #566275; font-size: 14px; }}
    .citations li {{ margin: 8px 0; }}
  </style>
</head>
<body>
  <main>
    <h1>{APP_TITLE}</h1>
    <p class="muted">Standalone PDF ingestion, local TF-IDF retrieval, and Gemini-grounded Q&A for authorized paper PDFs.</p>
    {body}
  </main>
</body>
</html>"""
    )


def _index_status() -> str:
    try:
        if (INDEX_DIR / "tfidf_meta.json").exists():
            meta = TfidfStore(INDEX_DIR).metadata
            return (
                f"sklearn TF-IDF index ready: {meta['chunk_count']} chunks, "
                f"{meta['vocabulary_size']} terms."
            )
        meta = VectorStore(INDEX_DIR).metadata
    except FileNotFoundError:
        return "No vector index found yet."
    return (
        f"Vector index ready: {meta['chunk_count']} chunks, "
        f"{meta['dimensions']} dimensions, model {meta['embedding_model']}."
    )


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    status = html.escape(_index_status())
    return _page(
        f"""
<section class="status">{status}</section>

<h2>Build Index</h2>
<form action="/build" enctype="multipart/form-data" method="post">
  <label for="files">Paper PDFs</label>
  <input id="files" name="files" type="file" accept="application/pdf" multiple required>
  <label for="chunk_words">Chunk words</label>
  <input id="chunk_words" name="chunk_words" type="number" min="250" max="1600" value="850">
  <label for="overlap_words">Overlap words</label>
  <input id="overlap_words" name="overlap_words" type="number" min="0" max="400" value="120">
  <button type="submit">Build retrieval index</button>
</form>

<h2>Ask Literature</h2>
<form action="/ask" method="post">
  <label for="question">Question</label>
  <textarea id="question" name="question" required></textarea>
  <label for="calculator_data">Calculator findings or report text</label>
  <textarea id="calculator_data" name="calculator_data"></textarea>
  <label for="top_k">Retrieved chunks</label>
  <input id="top_k" name="top_k" type="number" min="1" max="12" value="6">
  <button type="submit">Ask Gemini</button>
</form>
"""
    )


@app.get("/healthz", response_class=PlainTextResponse)
def healthz() -> PlainTextResponse:
    return PlainTextResponse(_index_status())


@app.post("/build", response_class=HTMLResponse)
async def build(
    files: list[UploadFile] = File(...),
    chunk_words: int = Form(850),
    overlap_words: int = Form(120),
) -> HTMLResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved_files: list[str] = []
    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            continue
        destination = UPLOAD_DIR / Path(upload.filename).name
        destination.write_bytes(await upload.read())
        saved_files.append(destination.name)

    if not saved_files:
        return _page('<section class="status">No PDF files were uploaded.</section>')

    manifest = MANIFEST_PATH if MANIFEST_PATH.exists() else None
    if RETRIEVAL_BACKEND == "tfidf":
        store = build_tfidf_index(
            UPLOAD_DIR,
            INDEX_DIR,
            manifest_path=manifest,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
    else:
        store = build_index(
            UPLOAD_DIR,
            INDEX_DIR,
            manifest_path=manifest,
            embedding_backend=RETRIEVAL_BACKEND,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
    status = html.escape(
        f"Saved {len(saved_files)} PDF(s). Built {store.metadata['chunk_count']} chunks "
        f"at {store.index_dir}."
    )
    names = "".join(f"<li>{html.escape(name)}</li>" for name in saved_files)
    return _page(
        f'<section class="status">{status}</section><ul>{names}</ul><p><a href="/">Back</a></p>'
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(
    question: str = Form(...),
    calculator_data: str = Form(""),
    top_k: int = Form(6),
) -> HTMLResponse:
    engine = (
        TfidfRagEngine.from_index(INDEX_DIR)
        if (INDEX_DIR / "tfidf_meta.json").exists()
        else RagEngine.from_index(INDEX_DIR, embedding_backend=RETRIEVAL_BACKEND)
    )
    answer = engine.answer(
        question,
        calculator_data=calculator_data or None,
        top_k=top_k,
    )
    citations = "".join(
        "<li>"
        f"{html.escape(result.chunk.citation_label())} "
        f"score={result.score:.3f} "
        f"{html.escape(result.chunk.source_title or result.chunk.source_id)}"
        "</li>"
        for result in answer.contexts
    )
    return _page(
        f"""
<section class="answer">{html.escape(answer.answer)}</section>
<h2>Citations</h2>
<ul class="citations">{citations}</ul>
<p><a href="/">Back</a></p>
"""
    )
