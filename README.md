# Manga Tracker API

A backend for tracking your own manga reading — synced from MyAnimeList — with
Gemini-powered analysis of your habits and reviews, an evolving "taste profile",
AI recommendations grounded in what MAL's community actually thinks, natural-
language catalog search, and on-demand "roast my manga" for laughs.

Built with FastAPI, PostgreSQL (+ pgvector), SQLAlchemy 2.0 (async), Alembic,
Redis, and the official `google-genai` SDK.

## Contents

- [Architecture & key decisions](#architecture--key-decisions)
- [Project layout](#project-layout)
- [Running it](#running-it)
- [Configuration](#configuration)
- [Database & migrations](#database--migrations)
- [Connecting your MyAnimeList account](#connecting-your-myanimelist-account)
- [API overview](#api-overview)

## Architecture & key decisions

This is deliberately a **single-user** backend — it tracks one person's MAL
account and library, not a multi-tenant catalog service. That simplifies the
data model considerably (one `MALAccount` per `User`, one current `TasteProfile`
per user) without ruling out multiple local accounts down the line.

### Provider-agnostic LLM layer, one Gemini provider for everything

`app/services/llm/base.py` defines two small `Protocol` interfaces —
`TextGenerationProvider` (`generate_text`, `generate_structured`) and
`EmbeddingProvider` (`embed_documents`, `embed_query`) — that the rest of the
app codes against, keeping application/business logic decoupled from any
specific vendor's SDK. Unlike Claude (which has no first-party embeddings
endpoint and needs a second vendor for that), Gemini natively supports both
generation and embeddings behind one API key — so a single `GeminiProvider`
(`app/services/llm/gemini_provider.py`) implements *both* Protocols via one
`genai.Client`:

- Uses `genai.Client` / `client.aio.models.*` from the official `google-genai`
  package exclusively. Declared as `google-genai>=1.51` (see `requirements.txt`
  / `pyproject.toml`) — that floor isn't arbitrary: `1.51.0` is the first release
  whose `ThinkingConfig` exposes `thinking_level` (the Gemini-3-generation
  thinking knob this provider sets directly), and everything else the provider
  touches (`generate_content`/`embed_content`/`types.*`/`errors.*`) is stable
  across the whole `>=1.51` range — the 1.x→2.x jump only changed the
  (unused-here) Interactions API. Verified directly against the installed
  `google-genai` 2.8.0 source — see the version note atop `gemini_provider.py`
- `model=settings.gemini_model` (defaults to `gemini-3.5-flash`) with
  `thinking_config=ThinkingConfig(thinking_level=settings.gemini_thinking_level)`
  (defaults to `"high"`)
- Structured outputs via `response_mime_type="application/json"` +
  `response_schema=SomeBaseModel`, parsed back into the Pydantic model — used
  for every analysis/recommendation/search/digest/roast payload
  (`ReviewAnalysis`, `TasteProfileAnalysis`, `RecommendationBatch`,
  `SearchExplanationBatch`, `ReadingHabitAnalysis`, `CommunityReviewDigest`,
  `MangaRoast`, …)
- Embeddings via `client.aio.models.embed_content` with task-type hints
  (`RETRIEVAL_DOCUMENT` for catalog text, `RETRIEVAL_QUERY` for search queries)
  — `embedding_model` / `embedding_dimensions` are both configurable and wired
  through to the pgvector column and HNSW index in the initial migration
- Typed exception handling mapped by HTTP status (429 → rate-limited, 401/403 →
  auth, 5xx → transient, refusals via `prompt_feedback.block_reason` /
  `finish_reason`) into a small `LLMError` hierarchy
  (`app/services/llm/exceptions.py`) that the API layer translates to clean
  HTTP responses (429 / 502 / 422) in `app/main.py`
- Relies on Gemini's *implicit* prompt caching (automatic for repeated
  prefixes) rather than Claude's explicit `cache_control` blocks — so
  `cache_system_prompt` exists on the Protocol for compatibility but is a
  deliberate no-op here; see the docstring in `gemini_provider.py`

If you ever want to add another provider, implement the same `Protocol`(s) and
swap it in `app/services/llm/factory.py` — nothing else needs to change. Both
`get_text_provider()` and `get_embedding_provider()` currently resolve to the
same memoized `GeminiProvider` singleton.

### Taste vs. habits — kept deliberately separate

- **`TasteProfile`** (`app/services/ai/taste_profile.py`) — *what* you like:
  genres, themes, demographics, rating tendencies, recent shifts. Versioned and
  immutable (`is_current` flag flips on a new generation), with an embedding so
  it can drive vector retrieval for recommendations.
- **`ReadingHabitAnalysis`** (`app/services/ai/habit_analysis.py`) — *how* you
  read: pace, completion/drop behavior, series-length preferences, suggestions.
  Generated on demand, not persisted as a versioned entity.

Conflating the two would have produced a single muddy "preferences" blob that
answers neither "what should I read next" nor "how is my reading behavior
changing" particularly well.

### Recommendations & search: cheap retrieval, one bounded reasoning call

Both pipelines follow the same shape, balancing cost/latency against quality:

1. **pgvector cosine search** narrows the full catalog to a small shortlist
   (recommendations: against the user's taste-profile embedding, excluding
   already-tracked manga; search: against the query embedding).
2. **One structured-output Gemini call** reasons over that shortlist — picking,
   ranking, and explaining — constrained to reference candidates *only* by
   `mal_id` from the list it was given. `_resolve_candidates` /
   `_explain_candidates` then re-attach real `Manga` rows, so the model can
   never hallucinate a catalog entry that doesn't exist.

This avoids both extremes: pure vector search (no personalized reasoning about
*why* something fits) and "let the LLM search the whole catalog" (expensive,
and prone to inventing titles).

### Community-review digests: grounding recommendations in what MAL readers actually say

MAL's *official* API (which `MALClient` uses for syncing) has no reviews
endpoint — user-written reviews only exist on the website. `app/services/jikan/`
talks to the public [Jikan API](https://docs.api.jikan.moe/), an unofficial but
widely-relied-upon read-only scraper of MAL's site data, specifically
`/manga/{id}/reviews` — the only practical source for this. `JikanClient` is
unauthenticated and self-throttles (`_REQUEST_INTERVAL_SECONDS`) to stay under
its public rate limit.

`CommunityReviewService` (`app/services/ai/community_reviews.py`) samples the
most-engaged-with reviews for a manga, and distills them with **one
structured-output Gemini call** into a spoiler-free `CommunityReviewDigest`
(consensus, praised/criticized aspects, themes, "best for" — see
`schemas/review.py`), persisted on `Manga.community_review_digest`.
`Manga.community_review_digest_generated_at` is the separate "have we checked
yet?" marker — set even when MAL has no reviews for a title, so niche/new manga
aren't re-queried on every sync.

`backfill_community_review_digests` mirrors `manga_index.backfill_missing_embeddings`'s
shape (find what's missing, process in small bounded batches, commit
incrementally) and is wired into the post-sync pipeline in `api/routes/sync.py`
as best-effort enrichment — a slow/failing Jikan or LLM call logs a warning and
never fails the sync itself. `RecommendationService._community_take` then folds
each candidate's digest into the prompt as a "community take" signal alongside
genre/synopsis, so picks can be grounded in what readers actually report
experiencing (pacing, tone, who it tends to land with), not just embedding
similarity.

### "Roast my manga": on-demand, personalized, just for laughs

`GET /ai/roast/{manga_id}` (`app/services/ai/roast.py`) is a different kind of
AI feature — not analysis, just entertainment. It looks up the title in *your*
library (404s with a friendly nudge if you haven't added it yet), pulls in your
own progress/status/score/notes/review alongside the catalog data and community
digest, and asks Gemini for a structured, spoiler-free `MangaRoast`: a short
funny roast, a "signature burn" one-liner, a backhanded compliment, and a
tongue-in-cheek verdict. The system prompt leans hard on personalization — the
funniest material is the *gap* between the catalog description and how you
actually responded to it (three volumes into a "feel-good slice of life" and
you scored it a 4?).

Deliberately **not cached or persisted** — regenerating it for a different (or
just funnier) take is the point, and Gemini's implicit prompt caching already
keeps repeat calls over the same manga/library context cheap on the input side
while still letting the output vary.

### Prompt-cache-friendly context rendering

`app/services/ai/context.py` renders the user's library/review/activity history
into deterministic text — fixed ordering, no timestamps or UUIDs — so that
repeated analysis calls within a session can benefit from Gemini's *implicit*
prompt caching (which keys off exact-prefix matches) instead of silently
missing the cache on every call due to incidental text differences.

### Optional Redis caching

`app/core/cache.py` wraps `redis.asyncio` behind a small `CacheClient` that:

- No-ops transparently when `REDIS_ENABLED=false`, *and* swallows any Redis
  connection error at the call site — caching is a pure optimization and must
  never be a new way for the API to fail.
- Is wired into the three most expensive *repeatable* AI reads
  (`GET /ai/recommendations`, `POST /ai/search`, `GET /ai/habits`) via
  `cache.get_or_compute(...)`. `GET /ai/roast/{manga_id}` is deliberately left
  uncached — see "Roast my manga" above; a stable cached joke would work against
  the feature's whole point.
- Keys recommendations by `(user_id, taste_profile_version)` — so a fresh
  taste-profile generation naturally busts the old recommendation cache without
  any explicit invalidation, and otherwise reuses results until `CACHE_TTL_SECONDS`
  expires.
- Keys search by the normalized query string, so repeated phrasings of the same
  search skip both the embedding call and the explanation call.

### Single-user MAL sync

`app/services/mal/` implements MAL's OAuth2 **PKCE "plain"** flow (MAL is one of
the few providers that doesn't support the `S256` challenge method —
`code_challenge == code_verifier`). `scripts/mal_auth.py` is a one-time helper
you run locally to obtain tokens for *your own* account; `MALSyncService`
(`app/services/mal/sync.py`) then cursor-paginates your list, upserts `Manga` /
`LibraryEntry` rows, and emits `ReadingActivity` log entries for detected
changes (status changes, progress updates, score changes, completions). Sync
can be triggered in the background (`POST /sync/mal/run`) or run inline for
first-time setup/debugging (`POST /sync/mal/run-now`).

## Project layout

```
app/
  core/        settings, security (JWT/password hashing), structlog setup, Redis cache
  db/          SQLAlchemy base/session/engine
  models/      User, Manga, LibraryEntry, Review, ReadingActivity, TasteProfile, MALAccount
  schemas/     Pydantic v2 request/response & structured-LLM-output models
  services/
    llm/       provider Protocols + GeminiProvider (text + embeddings) + factory
    mal/       MAL OAuth client + sync service
    jikan/     unauthenticated client for MAL community reviews (api.jikan.moe)
    ai/        taste profile, habit analysis, recommendations, search, community-review
               digests, roast, embedding backfill, shared context rendering
    factory.py process-wide singleton service factories
  api/
    deps.py    shared FastAPI dependencies (DB session, current user, cache client)
    routes/    auth, manga, library, reviews, sync, ai
    router.py  combines all routers
  main.py      app factory, lifespan, CORS, typed exception → HTTP mapping
alembic/       async-aware migrations (initial schema creates the `vector` extension,
               all tables, and an HNSW cosine index on manga.embedding; a follow-up
               migration adds the community-review-digest columns on `manga`)
scripts/
  mal_auth.py  one-time MAL OAuth PKCE helper
```

## Running it

### Option A — Docker Compose (recommended)

Brings up Postgres (with pgvector pre-installed), Redis, and the API, and runs
migrations automatically on container start (`docker-entrypoint.sh`):

```bash
cp .env.example .env   # then fill in SECRET_KEY, GEMINI_API_KEY, MAL_*
docker compose up --build
```

The app reads `.env` via `env_file`, but `docker-compose.yml` overrides
`DATABASE_URL` / `REDIS_URL` to point at the in-network `db` / `redis` service
names — you don't need to edit those two for Docker.

API docs: http://localhost:8000/docs · Health check: http://localhost:8000/health

### Option B — Local Python

Requires a local PostgreSQL with the `vector` extension available (e.g. the
`pgvector/pgvector:pg16` image) and Redis (or set `REDIS_ENABLED=false`).

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env   # fill in secrets and point DATABASE_URL/REDIS_URL at your services
alembic upgrade head
uvicorn app.main:app --reload
```

## Configuration

All settings live in `app/core/config.py` (`Settings`, loaded from `.env` via
`pydantic-settings`); see `.env.example` for the full annotated list. Highlights:

| Variable | Purpose |
| --- | --- |
| `SECRET_KEY` | JWT signing key — generate with `python -c "import secrets; print(secrets.token_urlsafe(64))"` |
| `DATABASE_URL` | `postgresql+asyncpg://...` — must point at a Postgres with the `vector` extension |
| `REDIS_ENABLED` / `REDIS_URL` / `CACHE_TTL_SECONDS` | Optional response caching for AI endpoints — set `REDIS_ENABLED=false` to disable entirely |
| `GEMINI_API_KEY` / `GEMINI_MODEL` / `GEMINI_THINKING_LEVEL` | Google Gemini — used for *both* text generation and embeddings (get a key at https://aistudio.google.com/apikey); defaults to `gemini-3.5-flash` at `thinking_level=high` |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | Gemini embeddings (`gemini-embedding-001` at 1536 dims by default) — the dimension must match the pgvector column in the initial migration if changed |
| `MAL_CLIENT_ID` / `MAL_CLIENT_SECRET` / `MAL_USERNAME` | Your MyAnimeList API client (register at https://myanimelist.net/apiconfig) and account |

Community-review fetching (`app/services/jikan/`) talks to the public,
unauthenticated [Jikan API](https://docs.api.jikan.moe/) — no extra credentials
needed.

## Database & migrations

Standard Alembic, but async-aware (`alembic/env.py` builds an async engine and
reads the URL from `Settings`, not `alembic.ini`):

```bash
alembic upgrade head                          # apply migrations
alembic revision --autogenerate -m "message"  # generate a new migration from model changes
```

The initial migration (`0001_initial_schema`) creates the `vector` extension,
all seven tables, and an HNSW cosine-distance index on `manga.embedding`.
`0002_community_review_digest` adds `manga.community_review_digest` (JSONB) and
`manga.community_review_digest_generated_at` — the persisted digest and its
"have we checked yet?" marker, respectively (see "Community-review digests" above).

## Connecting your MyAnimeList account

This is a one-time, per-account flow (the backend syncs only the account you
configure — there's no multi-user MAL OAuth dance):

1. Register an app at https://myanimelist.net/apiconfig (redirect URI can be
   something simple like `http://localhost:8080/callback`); put the client
   id/secret in `.env`.
2. Register a local user and log in (`POST /auth/register`, `POST /auth/login`)
   to get an access token.
3. Run `python scripts/mal_auth.py` — it walks you through MAL's OAuth2 PKCE
   ("plain" challenge) flow and prints a ready-to-send body.
4. `POST /sync/mal/connect` with that body (and your own bearer token) to store
   the MAL tokens on your account.
5. `POST /sync/mal/run-now` for an initial synchronous sync (or `POST
   /sync/mal/run` to run it in the background and poll `GET /sync/mal/status`).

Each sync upserts `Manga`/`LibraryEntry` rows, logs detected changes as
`ReadingActivity` entries, and backfills missing manga embeddings so new titles
are immediately searchable/recommendable.

## API overview

All routes are mounted under `API_V1_PREFIX` (default `/api/v1`); full
interactive docs at `/docs`.

**Auth** (`/auth`) — `POST /register`, `POST /login`, `POST /refresh`, `GET /me`

**Catalog** (`/manga`) — `GET /` (paginated, filterable), `GET /{manga_id}`

**Library** (`/library`) — `GET /`, `POST /`, `PATCH /{entry_id}`,
`DELETE /{entry_id}` — reading status, progress, personal score; changes are
logged to `ReadingActivity`

**Reviews** (`/reviews`) — `GET /`, `POST /`, `PATCH /{review_id}`,
`DELETE /{review_id}`, `POST /{review_id}/analyze` (kicks off background
sentiment/theme analysis via Gemini, persisted as structured `llm_analysis`)

**MAL sync** (`/sync`) — `GET /mal/status`, `POST /mal/connect`,
`POST /mal/run` (background), `POST /mal/run-now` (synchronous) — both run
variants also backfill missing manga embeddings and a small batch of community-
review digests as best-effort enrichment after the core sync completes

**AI** (`/ai`):
- `GET /taste-profile`, `GET /taste-profile/history`, `POST /taste-profile` —
  read the current versioned taste profile, list history, or generate a new
  version from your current library/reviews/activity
- `GET /habits` — on-demand reading-behavior analysis (pace, completion,
  rating tendencies, suggestions)
- `GET /recommendations` — AI picks from your synced & embedded catalog,
  reasoned against your current taste profile *and*, where available, what
  MAL's community generally says about each candidate
- `POST /search` — natural-language catalog search ("something like X but
  shorter and funnier") with per-result AI explanations
- `GET /roast/{manga_id}` — on-demand, funny/affectionate AI roast of a manga
  in your library, personalized to your own progress/score/review (404s if
  you haven't added the title to your library yet); uncached, regenerate it
  whenever you want a fresh take
