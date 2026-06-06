"""Prompt templates for grounded fetal MRI report support."""

from __future__ import annotations

from .store import SearchResult


SYSTEM_INSTRUCTIONS = """You support fetal brain MRI reporting for a radiologist.
Use only the calculator data and retrieved literature excerpts provided below.
Do not introduce outside medical claims, prevalence estimates, thresholds, or diagnoses.
If the excerpts do not support an answer, say the evidence is insufficient.
Every factual medical claim must include citations like [C1] or [C2].
Do not include patient identifiers. Phrase output as clinical decision support for physician review."""


def build_grounded_prompt(
    *,
    question: str,
    contexts: list[SearchResult],
    calculator_data: str | None = None,
) -> str:
    context_blocks = []
    for index, result in enumerate(contexts, start=1):
        chunk = result.chunk
        context_blocks.append(
            "\n".join(
                [
                    f"[C{index}] {chunk.citation_label()}",
                    f"Title: {chunk.source_title or chunk.source_id}",
                    f"Retrieval score: {result.score:.3f}",
                    chunk.text,
                ]
            )
        )

    data_block = calculator_data.strip() if calculator_data else "No calculator data."
    return "\n\n".join(
        [
            SYSTEM_INSTRUCTIONS,
            "Calculator data:",
            data_block,
            "Retrieved literature:",
            "\n\n".join(context_blocks),
            "Question:",
            question.strip(),
            "Answer:",
        ]
    )
