"""FastAPI application factory and entry point (`uvicorn app.main:app`)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.cache import get_cache_client
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.services.llm.exceptions import (
    LLMAuthenticationError,
    LLMError,
    LLMOutputError,
    LLMRateLimitedError,
    LLMRefusalError,
)
from app.services.mal.exceptions import MALAuthenticationError, MALError

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    settings = get_settings()
    logger.info(
        "app_startup",
        environment=settings.environment,
        model=settings.gemini_model,
        cache_enabled=settings.redis_enabled,
    )
    yield
    await get_cache_client().aclose()
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
    )

    # Permissive by default for local frontend development; tighten via an
    # explicit allow-list before deploying somewhere the API is internet-facing.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(LLMRateLimitedError)
    async def _llm_rate_limited(_: Request, exc: LLMRateLimitedError) -> JSONResponse:
        headers = {"Retry-After": str(exc.retry_after_seconds)} if exc.retry_after_seconds else {}
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "The AI provider is rate-limiting requests. Please try again shortly."},
            headers=headers,
        )

    @app.exception_handler(LLMAuthenticationError)
    async def _llm_auth(_: Request, exc: LLMAuthenticationError) -> JSONResponse:
        logger.error("llm_authentication_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "The AI provider rejected our credentials. Check GEMINI_API_KEY."},
        )

    @app.exception_handler(LLMRefusalError)
    async def _llm_refusal(_: Request, exc: LLMRefusalError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "The AI declined to process this request.", "category": exc.category},
        )

    @app.exception_handler(LLMOutputError)
    async def _llm_output(_: Request, exc: LLMOutputError) -> JSONResponse:
        logger.warning("llm_output_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "The AI provider returned an unusable response. Please try again."},
        )

    @app.exception_handler(LLMError)
    async def _llm_generic(_: Request, exc: LLMError) -> JSONResponse:
        logger.warning("llm_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "The AI provider is temporarily unavailable. Please try again shortly."},
        )

    @app.exception_handler(MALAuthenticationError)
    async def _mal_auth(_: Request, exc: MALAuthenticationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": f"MyAnimeList authentication problem: {exc}. Reconnect via POST /sync/mal/connect."},
        )

    @app.exception_handler(MALError)
    async def _mal_generic(_: Request, exc: MALError) -> JSONResponse:
        logger.warning("mal_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": "MyAnimeList is temporarily unavailable. Please try again shortly."},
        )


app = create_app()
