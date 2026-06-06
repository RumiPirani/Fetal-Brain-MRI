"""Retrieval and Gemini answer generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import get_api_key, get_generation_model
from .embeddings import EmbeddingClient, make_embedding_client
from .prompts import build_grounded_prompt
from .store import SearchResult, VectorStore
from .tfidf import TfidfStore


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    contexts: list[SearchResult]
    model_name: str


class GeminiGenerator:
    def __init__(self, *, api_key: str | None = None, model_name: str | None = None) -> None:
        from google import genai
        from google.genai import types

        self.client = genai.Client(api_key=api_key or get_api_key())
        self.types = types
        self.model_name = model_name or get_generation_model()

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.8,
            ),
        )
        text = getattr(response, "text", None)
        if text:
            return str(text).strip()
        return str(response).strip()


class RagEngine:
    """Load a vector index, retrieve relevant chunks, and answer with Gemini."""

    def __init__(
        self,
        *,
        store: VectorStore,
        embedder: EmbeddingClient,
        generator: GeminiGenerator | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.generator = generator

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
        *,
        embedding_backend: str = "gemini",
    ) -> "RagEngine":
        return cls(
            store=VectorStore(index_dir),
            embedder=make_embedding_client(embedding_backend),
        )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 6,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        query_vector = self.embedder.embed_query(question)
        return self.store.search(query_vector, top_k=top_k, min_score=min_score)

    def answer(
        self,
        question: str,
        *,
        calculator_data: str | None = None,
        top_k: int = 6,
        min_score: float | None = None,
    ) -> RagAnswer:
        contexts = self.retrieve(question, top_k=top_k, min_score=min_score)
        if not contexts:
            return RagAnswer(
                answer="The indexed literature did not retrieve enough evidence to answer.",
                contexts=[],
                model_name=(
                    self.generator.model_name
                    if self.generator is not None
                    else get_generation_model()
                ),
            )
        generator = self.generator or GeminiGenerator()
        prompt = build_grounded_prompt(
            question=question,
            contexts=contexts,
            calculator_data=calculator_data,
        )
        return RagAnswer(
            answer=generator.generate(prompt),
            contexts=contexts,
            model_name=generator.model_name,
        )


class TfidfRagEngine:
    """Use a local sklearn TF-IDF index for retrieval and Gemini synthesis."""

    def __init__(
        self,
        *,
        store: TfidfStore,
        generator: GeminiGenerator | None = None,
    ) -> None:
        self.store = store
        self.generator = generator

    @classmethod
    def from_index(cls, index_dir: str | Path) -> "TfidfRagEngine":
        return cls(store=TfidfStore(index_dir))

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 6,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        return self.store.search(question, top_k=top_k, min_score=min_score)

    def answer(
        self,
        question: str,
        *,
        calculator_data: str | None = None,
        top_k: int = 6,
        min_score: float | None = None,
    ) -> RagAnswer:
        contexts = self.retrieve(question, top_k=top_k, min_score=min_score)
        if not contexts:
            return RagAnswer(
                answer="The indexed literature did not retrieve enough evidence to answer.",
                contexts=[],
                model_name=(
                    self.generator.model_name
                    if self.generator is not None
                    else get_generation_model()
                ),
            )
        generator = self.generator or GeminiGenerator()
        prompt = build_grounded_prompt(
            question=question,
            contexts=contexts,
            calculator_data=calculator_data,
        )
        return RagAnswer(
            answer=generator.generate(prompt),
            contexts=contexts,
            model_name=generator.model_name,
        )
