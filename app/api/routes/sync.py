from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.core.logging import get_logger
from app.db.session import session_scope
from app.models.mal_account import MALAccount
from app.schemas.sync import MALConnectRequest, SyncResult, SyncStatus
from app.services.ai.community_reviews import CommunityReviewService, backfill_community_review_digests
from app.services.ai.manga_index import backfill_missing_embeddings
from app.services.factory import get_community_review_service, get_embedding_provider, get_mal_sync_service
from app.services.llm.base import EmbeddingProvider
from app.services.mal.sync import MALSyncService

router = APIRouter(prefix="/sync", tags=["sync"])
logger = get_logger(__name__)


async def _get_account(db: DbSession, current_user: CurrentUser) -> MALAccount | None:
    result = await db.execute(select(MALAccount).where(MALAccount.user_id == current_user.id))
    return result.scalar_one_or_none()


@router.get("/mal/status", response_model=SyncStatus)
async def sync_status(db: DbSession, current_user: CurrentUser) -> SyncStatus:
    account = await _get_account(db, current_user)
    if account is None:
        return SyncStatus(mal_username=None, last_synced_at=None, last_sync_status=None, last_sync_error=None)

    return SyncStatus(
        mal_username=account.mal_username,
        last_synced_at=account.last_synced_at,
        last_sync_status=account.last_sync_status,
        last_sync_error=account.last_sync_error,
    )


@router.post("/mal/connect", response_model=SyncStatus, status_code=status.HTTP_201_CREATED)
async def connect_mal_account(payload: MALConnectRequest, db: DbSession, current_user: CurrentUser) -> SyncStatus:
    """
    Stores MAL OAuth tokens obtained via `scripts/mal_auth.py` (the one-time PKCE
    flow) on the current user's account, creating the link if it doesn't exist yet.
    """
    account = await _get_account(db, current_user)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=payload.expires_in)

    if account is None:
        account = MALAccount(user_id=current_user.id, mal_username=payload.mal_username)
        db.add(account)

    account.mal_username = payload.mal_username
    account.access_token = payload.access_token
    account.refresh_token = payload.refresh_token
    account.token_expires_at = expires_at
    account.last_sync_status = None
    account.last_sync_error = None

    await db.commit()
    await db.refresh(account)

    return SyncStatus(
        mal_username=account.mal_username,
        last_synced_at=account.last_synced_at,
        last_sync_status=account.last_sync_status,
        last_sync_error=account.last_sync_error,
    )


async def _run_sync_in_background(user_id: uuid.UUID) -> None:
    sync_service = get_mal_sync_service()
    embeddings = get_embedding_provider()
    community_reviews = get_community_review_service()

    async with session_scope() as db:
        result = await db.execute(select(MALAccount).where(MALAccount.user_id == user_id))
        account = result.scalar_one_or_none()
        if account is None:
            return
        try:
            await sync_service.sync_user(db, account)
            await backfill_missing_embeddings(db, embeddings)
        except Exception:
            logger.warning("mal_sync_background_failed", user_id=str(user_id), exc_info=True)
            return

        # Best-effort enrichment — failures here shouldn't mark the sync itself as
        # failed (the catalog/library data is already safely persisted by this point).
        try:
            await backfill_community_review_digests(db, community_reviews)
        except Exception:
            logger.warning("community_review_backfill_failed", user_id=str(user_id), exc_info=True)


@router.post("/mal/run", status_code=status.HTTP_202_ACCEPTED)
async def trigger_sync(background_tasks: BackgroundTasks, db: DbSession, current_user: CurrentUser) -> dict:
    account = await _get_account(db, current_user)
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No MAL account connected — run scripts/mal_auth.py and POST /sync/mal/connect first",
        )

    background_tasks.add_task(_run_sync_in_background, current_user.id)
    return {"detail": "Sync started in the background — poll GET /sync/mal/status for progress"}


@router.post("/mal/run-now", response_model=SyncResult)
async def trigger_sync_synchronous(
    db: DbSession,
    current_user: CurrentUser,
    sync_service: MALSyncService = Depends(get_mal_sync_service),
    embeddings: EmbeddingProvider = Depends(get_embedding_provider),
    community_reviews: CommunityReviewService = Depends(get_community_review_service),
) -> SyncResult:
    """Run the sync inline and wait for the result — handy for the first sync or debugging."""
    account = await _get_account(db, current_user)
    if account is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No MAL account connected — run scripts/mal_auth.py and POST /sync/mal/connect first",
        )

    result = await sync_service.sync_user(db, account)
    await backfill_missing_embeddings(db, embeddings)

    # Best-effort enrichment, kept out of the returned `SyncResult` — a slow or
    # failing Jikan/LLM call here shouldn't make this debugging endpoint flaky.
    try:
        await backfill_community_review_digests(db, community_reviews)
    except Exception:
        logger.warning("community_review_backfill_failed", user_id=str(current_user.id), exc_info=True)

    return result
