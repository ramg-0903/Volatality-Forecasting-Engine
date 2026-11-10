# Application image: serves the Flask API and runs the Prefect flows.
#
# One image, several entrypoints (build-plan section 12.1 targets a single EC2
# instance, so the API and the flows share one artifact):
#   API           : python -m volatility_mlops.api.app        (default CMD)
#   daily flow    : python -m volatility_mlops.flows.daily
#   weekly retrain: python -m volatility_mlops.flows.weekly
#
# Dependencies are installed from the committed, hash-pinned lockfile
# (requirements.txt) for reproducible builds; the package itself is then
# installed with --no-deps so the lock is the single source of versions.
#
# NOTE: the default command runs Flask's built-in server. A production WSGI
# server (gunicorn) is a one-line swap and is planned for the AWS deploy
# (Day 26); it is intentionally not added here to avoid a new dependency.
FROM python:3.12-slim AS runtime

# - PYTHONDONTWRITEBYTECODE/PYTHONUNBUFFERED: cleaner logs, no stray .pyc
# - PIP_NO_CACHE_DIR: smaller image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_PORT=8000

WORKDIR /app

# Install locked runtime dependencies first so this layer caches across code
# changes. --require-hashes enforces the hashes in the lockfile.
COPY requirements.txt ./
RUN pip install --require-hashes --no-deps -r requirements.txt

# Install the package itself without touching the locked dependency set.
# README.md is copied too: pyproject's `readme = "README.md"` makes hatchling read it
# during metadata generation, so the build fails without it.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps .

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# Default entrypoint: the serving API. Override the command to run a flow.
CMD ["python", "-m", "volatility_mlops.api.app"]
