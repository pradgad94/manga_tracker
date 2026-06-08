"""Shared FastAPI dependencies: DB sessions, the authenticated user, service singletons."""
from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.cache import CacheClient, get_cache_client
from app.core.security import TokenError, TokenType, decode_token
from app.db.session import get_db_session
from app.models.user import User

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

DbSession = Annotated[AsyncSession, Depends(get_db_session)]
Cache = Annotated[CacheClient, Depends(get_cache_client)]


async def get_current_user(
    db: DbSession,
    token: Annotated[str | None, Depends(_oauth2_scheme)],
) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token is None:
        raise unauthorized

    try:
        subject = decode_token(token, expected_type=TokenType.ACCESS)
        user_id = uuid.UUID(subject)
    except (TokenError, ValueError) as exc:
        raise unauthorized from exc

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise unauthorized

    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()
