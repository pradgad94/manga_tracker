"""
Gemini implementation of `TextGenerationProvider` *and* `EmbeddingProvider`,
built against the official `google-genai` Python SDK exclusively (`genai.Client`,
never raw HTTP or an OpenAI-compatible shim pointed at Google's endpoint).

Unlike Claude, Gemini has a first-party embeddings endpoint
(`client.models.embed_content`), so one provider — and one API key — now covers
both text generation and embeddings. That's a deliberate simplification over the
previous Claude+Voyage split: fewer credentials to manage, one vendor's rate
limits and outages to reason about, and one place to translate SDK errors into
the app's provider-agnostic `LLMError` hierarchy.

SDK version: every type/field/error-shape claim below was checked directly
against the installed `google-genai` 2.8.0 source (`genai.Client`,
`client.aio.models.{generate_content,embed_content}`, `types.{ThinkingConfig,
GenerateContentConfig, EmbedContentConfig, GenerateContentResponse, Candidate,
FinishReason, BlockedReason}`, `errors.{APIError, ClientError, ServerError}`) —
not inferred from docs or another SDK's shape. `requirements.txt`/`pyproject.toml`
pin the floor at `google-genai>=1.51`: that's the first release exposing
`ThinkingConfig.thinking_level` (added in 1.51.0, replacing the older
`thinking_budget` knob for Gemini-3-generation models), which `_thinking_level`
below depends on directly — anything older would raise on construction. The
1.x → 2.x jump only changed the (unused-here) Interactions API surface, so
`generate_content`/`embed_content`/`types.*`/`errors.*` are stable across the
whole `>=1.51` range this provider supports.

Notes on the choices below:

- Async calls go through `client.aio.models.*` — the SDK's asyncio-native mirror
  of the synchronous `client.models.*` surface.
- Structured outputs use `response_mime_type="application/json"` plus
  `response_schema=<Pydantic model class>`; the SDK validates the JSON against
  the schema and exposes the typed instance via `response.parsed`.
- Free-text generation also goes through non-streaming `generate_content`: Gemini
  streams structured JSON as incremental fragments rather than typed deltas, so
  there's no equivalent to Anthropic's `stream.get_final_message()` that yields a
  single validated object — non-streaming is the simpler, equally-reliable choice
  here, backed by a generous client-side timeout (`http_options`).
- "Thinking" is controlled via `ThinkingConfig(thinking_level=...)`, the
  Gemini-3-generation analogue of Claude's adaptive-thinking + effort knobs.
- `cache_system_prompt` is accepted (it's part of the shared `TextGenerationProvider`
  Protocol) but deliberately a no-op here: Gemini caches repeated prompt prefixes
  *implicitly* and automatically — there's no `cache_control: {"type": "ephemeral"}`
  equivalent to set. The reading-history system prompt this app reuses across
  taste-profile/recommendation/search calls gets cached for free.
- SDK errors (`google.genai.errors.ClientError` / `ServerError`, both `APIError`
  subclasses exposing `.code` / `.status` / `.message`) are translated into the
  shared `LLMError` hierarchy by HTTP status code — never by string-matching.
- "Refusals" have no single dedicated field like Anthropic's `stop_reason ==
  "refusal"`; Gemini surfaces them as a blocked-prompt (`prompt_feedback.block_reason`)
  or a safety/policy `finish_reason` (`SAFETY`, `PROHIBITED_CONTENT`, `RECITATION`,
  `BLOCKLIST`, `SPII`) on the candidate — both are mapped to `LLMRefusalError`.
"""
from __future__ import annotations

from typing import Any, TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.core.config import Settings
from app.core.logging import get_logger
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMOutputError,
    LLMRateLimitedError,
    LLMRefusalError,
    LLMTransientError,
)

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Candidate finish reasons that mean "the model declined to produce this content"
# rather than "generation completed/was truncated normally".
_REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII"})

# Generous: structured-output prompts here embed a user's whole reading history
# and can legitimately take a while at higher thinking levels.
_REQUEST_TIMEOUT_MS = 180_000

# Voyage's `input_type` has a direct Gemini analogue in `task_type`: catalog
# content is embedded for retrieval-as-a-document, queries for retrieval-as-a-query.
_TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"
_TASK_TYPE_QUERY = "RETRIEVAL_QUERY"


class GeminiProvider:
    """Talks to Gemini exclusively through the official `google-genai` SDK."""

    def __init__(self, settings: Settings) -> None:
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )
        self._model = settings.gemini_model
        self._thinking_level = settings.gemini_thinking_level
        self._embedding_model = settings.embedding_model
        self._embedding_dimensions = settings.embedding_dimensions

    # --- TextGenerationProvider ---------------------------------------------

    async def generate_text(
        self,
        *,
        system: str,
        user_prompt: str,
        max_tokens: int = 4096,
        cache_system_prompt: bool = False,
    ) -> str:
        response = await self._generate(system, user_prompt, max_tokens, structured_as=None)

        text = response.text
        if not text:
            raise LLMOutputError("Gemini returned no text content")
        return text

    async def generate_structured(
        self,
        *,
        system: str,
        user_prompt: str,
        output_model: type[T],
        max_tokens: int = 4096,
        cache_system_prompt: bool = False,
    ) -> T:
        response = await self._generate(system, user_prompt, max_tokens, structured_as=output_model)

        parsed = response.parsed
        if not isinstance(parsed, output_model):
            raise LLMOutputError(f"Gemini's response did not validate against {output_model.__name__}")
        return parsed

    async def _generate(
        self, system: str, user_prompt: str, max_tokens: int, *, structured_as: type[BaseModel] | None
    ) -> types.GenerateContentResponse:
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self._thinking_level),
            **(
                {"response_mime_type": "application/json", "response_schema": structured_as}
                if structured_as is not None
                else {}
            ),
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=config,
            )
        except genai_errors.APIError as exc:
            raise _translate_error(exc, "Gemini") from exc

        _check_refusal(response)
        _log_usage("generate_structured" if structured_as else "generate_text", self._model, response)
        return response

    # --- EmbeddingProvider ---------------------------------------------------

    @property
    def dimensions(self) -> int:
        return self._embedding_dimensions

    @property
    def model_name(self) -> str:
        return self._embedding_model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts, task_type=_TASK_TYPE_DOCUMENT)

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self._embed([text], task_type=_TASK_TYPE_QUERY)
        return embeddings[0]

    async def _embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        if not texts:
            return []

        try:
            response = await self._client.aio.models.embed_content(
                model=self._embedding_model,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._embedding_dimensions,
                ),
            )
        except genai_errors.APIError as exc:
            raise _translate_error(exc, "Gemini embeddings") from exc

        if not response.embeddings or len(response.embeddings) != len(texts):
            raise LLMOutputError("Gemini returned an unexpected number of embeddings")

        logger.debug(
            "gemini_embed",
            model=self._embedding_model,
            task_type=task_type,
            count=len(texts),
            dimensions=self._embedding_dimensions,
        )
        vectors: list[list[float]] = []
        for embedding in response.embeddings:
            if embedding.values is None:
                raise LLMOutputError("Gemini returned an embedding with no vector values")
            vectors.append(embedding.values)
        return vectors


def _translate_error(exc: genai_errors.APIError, source: str) -> LLMError:
    code = exc.code
    message = exc.message or str(exc)

    if code == 429:
        return LLMRateLimitedError(f"{source} rate limit exceeded: {message}", retry_after_seconds=_retry_after_seconds(exc))
    if code in (401, 403):
        return LLMAuthenticationError(f"{source} API key was rejected: {message}")
    if code is not None and code >= 500:
        return LLMTransientError(f"{source} server error ({code}): {message}")
    return LLMOutputError(f"{source} rejected the request ({code}): {message}")


def _retry_after_seconds(exc: genai_errors.APIError) -> int | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = headers.get("retry-after") if headers is not None else None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _check_refusal(response: types.GenerateContentResponse) -> None:
    feedback = response.prompt_feedback
    block_reason = getattr(feedback, "block_reason", None)
    if block_reason is not None:
        category = _enum_name(block_reason)
        raise LLMRefusalError(f"Gemini blocked the prompt before generating a response ({category})", category=category)

    candidates = response.candidates or []
    if not candidates:
        return

    finish_reason = getattr(candidates[0], "finish_reason", None)
    if finish_reason is None:
        return

    category = _enum_name(finish_reason)
    if category in _REFUSAL_FINISH_REASONS:
        raise LLMRefusalError(f"Gemini declined to generate this content ({category})", category=category)


def _enum_name(value: Any) -> str:
    return getattr(value, "name", None) or str(value)


def _log_usage(operation: str, model: str, response: types.GenerateContentResponse) -> None:
    usage = response.usage_metadata
    candidates = response.candidates or []
    logger.debug(
        "gemini_call",
        operation=operation,
        model=model,
        prompt_tokens=getattr(usage, "prompt_token_count", None),
        output_tokens=getattr(usage, "candidates_token_count", None),
        thinking_tokens=getattr(usage, "thoughts_token_count", None),
        total_tokens=getattr(usage, "total_token_count", None),
        finish_reason=_enum_name(candidates[0].finish_reason) if candidates and candidates[0].finish_reason else None,
    )
