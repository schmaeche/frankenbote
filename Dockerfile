# Use the official slim Python 3.14 image — small, secure, well-maintained.
FROM python:3.14-slim

# Don't write .pyc files; flush stdout/stderr immediately so logs appear in real time.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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

# Copy templates so they're available in docker packagee for rendering and publishing.
COPY templates/ ./templates/

# Copy assets so they're available in docker packagee for rendering and publishing.
COPY assets/ ./assets/

# Install the project and its dependencies.
RUN pip install --no-cache-dir -e .

# Switch to the non-root user.
USER frankenbote

# Default command — overridable via docker compose run.
ENTRYPOINT ["frankenbote"]
CMD ["--help"]