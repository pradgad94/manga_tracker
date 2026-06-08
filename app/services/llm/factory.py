"""Process-wide singletons for the LLM and embedding providers, used as FastAPI deps."""
from __future__ import annotations

from functools import lru_cache

from app.core.config import get_settings
from app.services.llm.base import EmbeddingProvider, TextGenerationProvider
from app.services.llm.gemini_provider import GeminiProvider


@lru_cache
def _get_gemini_provider() -> GeminiProvider:
    """One Gemini client backs both protocols below — see gemini_provider.py."""
    return GeminiProvider(get_settings())


def get_text_provider() -> TextGenerationProvider:
    return _get_gemini_provider()


def get_embedding_provider() -> EmbeddingProvider:
    return _get_gemini_provider()
