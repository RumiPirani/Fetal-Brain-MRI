"""PDF text extraction."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path


def iter_pdf_pages(pdf_path: str | Path) -> Iterator[tuple[int, str]]:
    """Yield one-based page numbers and extracted text for a PDF."""

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    path = Path(pdf_path)
    reader = PdfReader(str(path))
    for page_index, page in enumerate(reader.pages, start=1):
        yield page_index, page.extract_text() or ""
