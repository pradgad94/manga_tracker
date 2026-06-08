#!/usr/bin/env bash
# Applies pending Alembic migrations before handing off to the app's CMD —
# keeps the schema in lockstep with the image without a separate migration step.
set -euo pipefail

echo "[entrypoint] applying database migrations..."
alembic upgrade head

echo "[entrypoint] starting: $*"
exec "$@"
