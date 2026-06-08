"""
Provider-agnostic LLM error hierarchy.

The rest of the application catches *these* exceptions, never SDK-specific ones
(e.g. `google.genai.errors.ClientError`). That keeps the provider genuinely
swappable — a future OpenAI/local-model provider just needs to translate its own
SDK errors into this hierarchy at the boundary, exactly like GeminiProvider does.
"""
from __future__ import annotations


class LLMError(Exception):
    """Base class for all LLM-provider failures."""


class LLMRateLimitedError(LLMError):
    """The provider is rate-limiting us; safe to retry after a delay."""

    def __init__(self, message: str, retry_after_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMTransientError(LLMError):
    """A transient/server-side failure (5xx, connection drop); safe to retry."""


class LLMAuthenticationError(LLMError):
    """The provider rejected our credentials — not retryable without operator action."""


class LLMRefusalError(LLMError):
    """The model declined to respond (safety refusal); not retryable as-is."""

    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class LLMOutputError(LLMError):
    """The provider returned a response we couldn't parse/validate as expected."""
