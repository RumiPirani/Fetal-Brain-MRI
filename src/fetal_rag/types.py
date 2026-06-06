"""Shared data structures for indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    id: str
    text: str
    source_id: str
    source_title: str = ""
    source_path: str = ""
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation_label(self) -> str:
        pages = ""
        if self.page_start is not None and self.page_end is not None:
            pages = (
                f" p.{self.page_start}"
                if self.page_start == self.page_end
                else f" pp.{self.page_start}-{self.page_end}"
            )
        return f"{self.source_id}{pages}"

    def embedding_text(self) -> str:
        title = f"{self.source_title}\n" if self.source_title else ""
        return f"{self.source_id}\n{title}{self.text}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source_id": self.source_id,
            "source_title": self.source_title,
            "source_path": self.source_path,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DocumentChunk":
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            source_id=str(data["source_id"]),
            source_title=str(data.get("source_title", "")),
            source_path=str(data.get("source_path", "")),
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            metadata=dict(data.get("metadata", {})),
        )
