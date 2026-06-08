from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession
from app.core.logging import get_logger
from app.db.session import session_scope
from app.models.activity import ReadingActivity, ReadingActivityType
from app.models.manga import Manga
from app.models.review import Review
from app.schemas.common import Page
from app.schemas.review import ReviewCreate, ReviewRead, ReviewUpdate
from app.services.ai.habit_analysis import HabitAnalysisService
from app.services.factory import get_habit_analysis_service

router = APIRouter(prefix="/reviews", tags=["reviews"])
logger = get_logger(__name__)


async def _get_owned_review(db: DbSession, current_user: CurrentUser, review_id: uuid.UUID) -> Review:
    result = await db.execute(
        select(Review)
        .where(Review.id == review_id, Review.user_id == current_user.id)
        .options(selectinload(Review.manga))
    )
    review = result.scalar_one_or_none()
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review not found")
    return review


async def _analyze_review_in_background(review_id: uuid.UUID) -> None:
    """Runs after the response is sent — keeps review creation snappy despite the LLM call."""
    service = get_habit_analysis_service()
    async with session_scope() as db:
        result = await db.execute(select(Review).where(Review.id == review_id).options(selectinload(Review.manga)))
        review = result.scalar_one_or_none()
        if review is None:
            return
        try:
            await service.analyze_review(db, review)
        except Exception:
            logger.warning("review_analysis_failed", review_id=str(review_id), exc_info=True)


@router.get("", response_model=Page[ReviewRead])
async def list_reviews(
    db: DbSession,
    current_user: CurrentUser,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Page[ReviewRead]:
    base = select(Review).where(Review.user_id == current_user.id).options(selectinload(Review.manga))

    total = len((await db.execute(base.with_only_columns(Review.id))).all())
    result = await db.execute(base.order_by(Review.created_at.desc()).limit(limit).offset(offset))
    items = list(result.scalars().all())

    return Page(items=[ReviewRead.model_validate(r) for r in items], total=total, limit=limit, offset=offset)


@router.post("", response_model=ReviewRead, status_code=status.HTTP_201_CREATED)
async def create_review(
    payload: ReviewCreate,
    background_tasks: BackgroundTasks,
    db: DbSession,
    current_user: CurrentUser,
) -> Review:
    manga = await db.get(Manga, payload.manga_id)
    if manga is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manga not found")

    review = Review(user_id=current_user.id, manga_id=manga.id, body=payload.body, score=payload.score)
    db.add(review)
    db.add(
        ReadingActivity(
            user_id=current_user.id,
            manga_id=manga.id,
            activity_type=ReadingActivityType.REVIEW_ADDED,
            payload={"score": payload.score},
        )
    )
    await db.commit()
    await db.refresh(review)

    background_tasks.add_task(_analyze_review_in_background, review.id)

    return await _get_owned_review(db, current_user, review.id)


@router.patch("/{review_id}", response_model=ReviewRead)
async def update_review(
    review_id: uuid.UUID,
    payload: ReviewUpdate,
    background_tasks: BackgroundTasks,
    db: DbSession,
    current_user: CurrentUser,
) -> Review:
    review = await _get_owned_review(db, current_user, review_id)
    changes = payload.model_dump(exclude_unset=True)

    body_changed = "body" in changes and changes["body"] != review.body
    for field, value in changes.items():
        setattr(review, field, value)

    if body_changed:
        review.llm_analysis = None
        review.llm_analyzed_at = None

    await db.commit()

    if body_changed:
        background_tasks.add_task(_analyze_review_in_background, review.id)

    return await _get_owned_review(db, current_user, review.id)


@router.post("/{review_id}/analyze", response_model=ReviewRead)
async def analyze_review_now(
    review_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
    service: HabitAnalysisService = Depends(get_habit_analysis_service),
) -> Review:
    """Synchronously (re)run LLM analysis on a review — useful right after editing it."""
    review = await _get_owned_review(db, current_user, review_id)
    await service.analyze_review(db, review)
    return await _get_owned_review(db, current_user, review.id)


@router.delete("/{review_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_review(review_id: uuid.UUID, db: DbSession, current_user: CurrentUser) -> None:
    review = await _get_owned_review(db, current_user, review_id)
    await db.delete(review)
    await db.commit()
