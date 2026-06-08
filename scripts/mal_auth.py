"""
One-time helper that walks through MyAnimeList's OAuth2 PKCE flow for the
operator's own account and prints the resulting tokens.

MAL's OAuth implementation only supports the "plain" PKCE challenge method,
i.e. code_challenge == code_verifier — there's no SHA256 step like most
providers use.

Usage:
    python scripts/mal_auth.py

Then:
  1. Open the printed URL, log in to MAL, and authorize the app.
  2. You'll be redirected to your configured redirect URI with `?code=...`.
     Paste the *full* redirected URL back into the prompt.
  3. The script exchanges the code for tokens and prints a ready-to-send
     request body for `POST /api/v1/sync/mal/connect` (send it with your own
     access token, e.g. via `curl` or the interactive API docs at /docs) — that
     endpoint stores the tokens on your account so the sync job can use them.

Requires MAL_CLIENT_ID (and MAL_CLIENT_SECRET, if your MAL app has one) in .env.
Register an app at https://myanimelist.net/apiconfig — set its redirect URI to
something simple like http://localhost:8080/callback.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import string
import sys
import urllib.parse

import httpx

sys.path.insert(0, ".")  # allow running as `python scripts/mal_auth.py` from the project root

from app.core.config import get_settings  # noqa: E402

_AUTHORIZE_URL = "https://myanimelist.net/v1/oauth2/authorize"
_TOKEN_URL = "https://myanimelist.net/v1/oauth2/token"
_REDIRECT_URI = "http://localhost:8080/callback"
_VERIFIER_ALPHABET = string.ascii_letters + string.digits + "-._~"


def _generate_code_verifier(length: int = 128) -> str:
    return "".join(secrets.choice(_VERIFIER_ALPHABET) for _ in range(length))


def _build_authorize_url(client_id: str, code_verifier: str, state: str) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "code_challenge": code_verifier,  # MAL uses the "plain" PKCE method
        "code_challenge_method": "plain",
        "state": state,
        "redirect_uri": _REDIRECT_URI,
    }
    return f"{_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"


def _extract_code(redirected_url: str, expected_state: str) -> str:
    parsed = urllib.parse.urlparse(redirected_url.strip())
    query = urllib.parse.parse_qs(parsed.query)

    if query.get("state", [None])[0] != expected_state:
        raise ValueError("State mismatch — the redirect URL doesn't match this session. Start over.")

    code = query.get("code", [None])[0]
    if not code:
        raise ValueError("No `code` parameter found in the redirected URL.")
    return code


async def _exchange_code(client_id: str, client_secret: str, code: str, code_verifier: str) -> dict:
    data = {
        "client_id": client_id,
        "grant_type": "authorization_code",
        "code": code,
        "code_verifier": code_verifier,
        "redirect_uri": _REDIRECT_URI,
    }
    if client_secret:
        data["client_secret"] = client_secret

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(_TOKEN_URL, data=data)

    response.raise_for_status()
    return response.json()


async def main() -> None:
    settings = get_settings()
    if not settings.mal_client_id:
        raise SystemExit("Set MAL_CLIENT_ID in .env first (https://myanimelist.net/apiconfig).")

    code_verifier = _generate_code_verifier()
    state = secrets.token_urlsafe(16)

    print("1. Open this URL, log in, and authorize the app:\n")
    print(f"   {_build_authorize_url(settings.mal_client_id, code_verifier, state)}\n")
    print(f"   (Make sure your MAL app's redirect URI is set to {_REDIRECT_URI})\n")

    redirected_url = input("2. Paste the full redirected URL here: ").strip()
    code = _extract_code(redirected_url, expected_state=state)

    token_payload = await _exchange_code(settings.mal_client_id, settings.mal_client_secret, code, code_verifier)

    print("\nSuccess! Send the following to POST /api/v1/sync/mal/connect")
    print("(authenticated as yourself, e.g. via the /docs Swagger UI):\n")
    print(
        json.dumps(
            {
                "mal_username": settings.mal_username or "<your-mal-username>",
                "access_token": token_payload["access_token"],
                "refresh_token": token_payload["refresh_token"],
                "expires_in": token_payload["expires_in"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
