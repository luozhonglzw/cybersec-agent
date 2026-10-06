# CyberSec Agent - minimal runtime image (Phase 9.3-G)
#
# Design (frozen):
#   - ONE HTTP/FastAPI container, no multi-stage build, no apt packages,
#     no curl, no shell wrapper, no new Python entrypoint.
#   - Base = verified official Astral uv image carrying Python 3.12 (slim).
#   - Runtime deps only (`--no-dev`); the dev group (httpx / pytest) stays out.
#   - Synthetic demo data is GENERATED at build time from the two existing
#     seed scripts (deterministic: fixed seed + fixed base time). Host data/
#     is excluded via .dockerignore and never copied in.
#   - Mutable audit state is NOT baked in; it lives on the /data volume at
#     runtime (see compose.yaml, AUDIT_DB_PATH=/data/audit.db).

FROM ghcr.io/astral-sh/uv:0.12.23-python3.12-trixie-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 1. Dependency metadata only, so the dependency layer caches independently
#    of application source changes.
COPY pyproject.toml uv.lock .python-version ./

# 2. Install third-party runtime dependencies without the project itself
#    (app/ is not present yet, so the project cannot be built at this point).
RUN uv sync --locked --no-dev --no-install-project

# 3. Application package plus exactly the two build-time seed scripts.
COPY app ./app
COPY scripts/seed_logs.py scripts/seed_threat_intel.py ./scripts/
COPY README.md ./

# 4. Install the project itself (hatchling, packages = ["app"]).
RUN uv sync --locked --no-dev

# 5. Bake deterministic synthetic demo data (generated, not committed).
#    Requires the project to be installed (seed scripts import app.schemas).
RUN uv run --no-sync python scripts/seed_logs.py \
 && uv run --no-sync python scripts/seed_threat_intel.py

EXPOSE 8000

# Existing ASGI object; bind 0.0.0.0 so the published port is reachable.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
