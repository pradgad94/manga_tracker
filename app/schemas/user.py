from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import EmailStr

from app.schemas.common import ORMModel


class UserRead(ORMModel):
    id: uuid.UUID
    email: EmailStr
    display_name: str | None
    is_active: bool
    created_at: datetime


class UserUpdate(ORMModel):
    display_name: str | None = None
