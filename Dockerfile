# ─────────────────────────────────────────────────────────────────────────────
# base — shared system setup, used by all stages
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.14-slim AS base

# Don't write .pyc files; flush stdout/stderr immediately so logs appear in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Allows CI to override the fallback version at build time
ARG SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FRANKENBOTE=0.0.1-dev0
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FRANKENBOTE=${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FRANKENBOTE}

# Create a non-root user for runtime. Running as root inside containers is
# a common security mistake; we avoid it from day one.
RUN useradd --create-home --shell /bin/bash frankenbote

# Set working directory.
WORKDIR /app

# Copy only the dependency declaration first — this lets Docker cache the
# (slow) pip install layer and re-use it whenever pyproject.toml hasn't changed.
COPY pyproject.toml ./

# Copy the package source so pip can install it.
COPY src/ ./src/

# Copy templates so they're available in the package for rendering and publishing.
COPY templates/ ./templates/

# Copy assets so they're available in the package for rendering and publishing.
COPY assets/ ./assets/


# ─────────────────────────────────────────────────────────────────────────────
# production — lean image, no dev/test tooling
# Regular (non-editable) install: the package is truly baked in.
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS production

RUN pip install --no-cache-dir .

# Switch to the non-root user.
USER frankenbote

# Default command — overridable via docker compose run.
ENTRYPOINT ["frankenbote"]
CMD ["--help"]


# ─────────────────────────────────────────────────────────────────────────────
# dev — editable install so volume-mounted src/ changes are picked up immediately
# ─────────────────────────────────────────────────────────────────────────────
FROM base AS dev

RUN pip install --no-cache-dir -e ".[dev]"

# Switch to the non-root user.
USER frankenbote

# Default command — overridable via docker compose run.
ENTRYPOINT ["frankenbote"]
CMD ["--help"]


# ─────────────────────────────────────────────────────────────────────────────
# test — inherits dev, bakes in the test suite, runs pytest by default
# ─────────────────────────────────────────────────────────────────────────────
FROM dev AS test

# Switch back to root briefly to copy test files (frankenbote user has no write
# access to /app which is owned by root at this point).
USER root
COPY tests/ ./tests/
RUN mkdir /app/.pytest_cache && chown frankenbote /app/.pytest_cache && chown frankenbote /app
USER frankenbote

ENTRYPOINT ["pytest"]
CMD ["-q"]