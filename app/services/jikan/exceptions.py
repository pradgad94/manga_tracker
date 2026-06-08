from __future__ import annotations


class JikanError(Exception):
    """
    Jikan integration failure (network error, non-2xx response, unexpected shape).

    Deliberately a single flat exception rather than a typed hierarchy like
    `MALError`/`LLMError`: community reviews are best-effort enrichment data, not
    something the rest of the app depends on to function. Every call site catches
    this, logs, and moves on — never surfaces it to the end user.
    """
