"""PDF text normalization and page-aware chunking."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .types import DocumentChunk


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
PDF_TEXT_REPLACEMENTS = {
    "\u00a0": " ",
    "\u00ad": "",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}


@dataclass(frozen=True)
class _TextUnit:
    text: str
    page_start: int
    page_end: int
    word_count: int


def normalize_text(text: str) -> str:
    """Normalize extraction artifacts while preserving readable prose."""

    text = text.replace("\r", "\n")
    for before, after in PDF_TEXT_REPLACEMENTS.items():
        text = text.replace(before, after)
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\n", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def _chunk_id(source_id: str, chunk_index: int, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{source_id}:{chunk_index:04d}:{digest}"


def _sentence_candidates(text: str) -> list[str]:
    candidates = [
        candidate.strip(" ;")
        for candidate in SENTENCE_BOUNDARY_RE.split(text)
        if candidate.strip(" ;")
    ]
    return candidates or [text]


def _split_long_candidate(
    text: str,
    *,
    page_number: int,
    max_words: int,
) -> list[_TextUnit]:
    words = text.split()
    if len(words) <= max_words:
        return [
            _TextUnit(
                text=" ".join(words),
                page_start=page_number,
                page_end=page_number,
                word_count=len(words),
            )
        ]

    units: list[_TextUnit] = []
    for start in range(0, len(words), max_words):
        unit_words = words[start : start + max_words]
        if not unit_words:
            continue
        units.append(
            _TextUnit(
                text=" ".join(unit_words),
                page_start=page_number,
                page_end=page_number,
                word_count=len(unit_words),
            )
        )
    return units


def _page_units(
    pages: Iterable[tuple[int, str]],
    *,
    max_unit_words: int,
) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    for page_number, raw_text in pages:
        text = normalize_text(raw_text)
        if not text:
            continue
        for candidate in _sentence_candidates(text):
            units.extend(
                _split_long_candidate(
                    candidate,
                    page_number=page_number,
                    max_words=max_unit_words,
                )
            )
    return units


def _overlap_units(units: list[_TextUnit], overlap_words: int) -> list[_TextUnit]:
    if overlap_words == 0:
        return []

    selected: list[_TextUnit] = []
    total_words = 0
    for unit in reversed(units):
        if selected and total_words + unit.word_count > overlap_words:
            break
        if not selected and unit.word_count > overlap_words:
            break
        selected.append(unit)
        total_words += unit.word_count
    return list(reversed(selected))


def _chunk_from_units(
    units: list[_TextUnit],
    *,
    source_id: str,
    source_title: str,
    source_path: str,
    chunk_index: int,
    overlap_words: int,
) -> DocumentChunk:
    text = " ".join(unit.text for unit in units)
    page_start = min(unit.page_start for unit in units)
    page_end = max(unit.page_end for unit in units)
    word_count = sum(unit.word_count for unit in units)
    return DocumentChunk(
        id=_chunk_id(source_id, chunk_index, text),
        text=text,
        source_id=source_id,
        source_title=source_title,
        source_path=source_path,
        page_start=page_start,
        page_end=page_end,
        metadata={
            "chunk_index": chunk_index,
            "word_count": word_count,
            "character_count": len(text),
            "unit_count": len(units),
            "overlap_words": overlap_words,
            "chunk_strategy": "sentence_window_v1",
        },
    )


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    *,
    source_id: str,
    source_title: str = "",
    source_path: str = "",
    chunk_words: int = 450,
    overlap_words: int = 80,
    min_words: int = 80,
) -> Iterator[DocumentChunk]:
    """Yield sentence-aware overlapping chunks with source and page provenance."""

    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words must be non-negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    max_unit_words = max(40, chunk_words // 2)
    units = _page_units(pages, max_unit_words=max_unit_words)
    if not units:
        return

    current: list[_TextUnit] = []
    current_words = 0
    chunk_index = 0

    for unit in units:
        if current and current_words + unit.word_count > chunk_words:
            yield _chunk_from_units(
                current,
                source_id=source_id,
                source_title=source_title,
                source_path=source_path,
                chunk_index=chunk_index,
                overlap_words=overlap_words,
            )
            chunk_index += 1
            current = _overlap_units(current, overlap_words)
            current_words = sum(overlap.word_count for overlap in current)
            if current_words + unit.word_count > chunk_words:
                current = []
                current_words = 0

        current.append(unit)
        current_words += unit.word_count

    if not current:
        return

    if current_words < min_words and chunk_index > 0:
        return

    yield _chunk_from_units(
        current,
        source_id=source_id,
        source_title=source_title,
        source_path=source_path,
        chunk_index=chunk_index,
        overlap_words=overlap_words,
    )
