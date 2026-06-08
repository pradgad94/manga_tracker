"""
Provider-agnostic interfaces for text generation and embeddings.

Two separate protocols rather than one "provider that does everything" because
not every vendor offers both — the previous Claude+Voyage setup needed two. Gemini
happens to support both natively, but application code (services/ai/*) still
depends only on these Protocols — never on `google.genai` directly — so either
side can be swapped independently by writing a new adapter that satisfies the
same shape, with no changes to call sites.

`GeminiProvider` (gemini_provider.py) is the only implementation of either
protocol, and it talks to Gemini exclusively through the official `google-genai`
SDK.
"""
from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class TextGenerationProvider(Protocol):
    """A chat/completion-style LLM that can produce free text or validated structured output."""

    async def generate_text(
        self,
        *,
        system: str,
        user_prompt: str,
        max_tokens: int = 4096,
        cache_system_prompt: bool = False,
    ) -> str:
        """Return the model's free-text response to a single-turn prompt."""
        ...

    async def generate_structured(
        self,
        *,
        system: str,
        user_prompt: str,
        output_model: type[T],
        max_tokens: int = 4096,
        cache_system_prompt: bool = False,
    ) -> T:
        """Return a response validated against `output_model` (Pydantic structured outputs)."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """A provider that maps text to fixed-dimension vectors for pgvector similarity search."""

    @property
    def dimensions(self) -> int:
        """The vector width this provider produces — must match the pgvector column width."""
        ...

    @property
    def model_name(self) -> str:
        """Identifier stored alongside generated embeddings, for traceability across model changes."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed texts that will be *stored and searched against* (e.g. manga synopses)."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single piece of text that will *search* stored embeddings (e.g. a user query)."""
        ...
