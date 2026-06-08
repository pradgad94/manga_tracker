from __future__ import annotations


class MALError(Exception):
    """Base class for MyAnimeList integration failures."""


class MALAuthenticationError(MALError):
    """The stored MAL token is missing, expired, or was rejected and couldn't be refreshed."""


class MALAPIError(MALError):
    """MAL's API returned an error response."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code
