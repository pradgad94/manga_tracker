from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.user import User


class MALAccount(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    """
    Links one local user to one MyAnimeList account and stores the OAuth2 tokens
    needed to call the MAL API on their behalf.

    Scope note: this app is built for single-user MAL sync — exactly one row is
    expected to exist (for the operator's own account), but modelling it as a
    table keyed by user_id keeps the door open for multi-user support later
    without a schema change.
    """

    __tablename__ = "mal_accounts"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    mal_username: Mapped[str] = mapped_column(String(120), nullable=False)
    mal_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    access_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_sync_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    last_sync_error: Mapped[str | None] = mapped_column(Text, nullable=True)

    user: Mapped["User"] = relationship(back_populates="mal_account")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<MALAccount user_id={self.user_id} mal_username={self.mal_username!r}>"
