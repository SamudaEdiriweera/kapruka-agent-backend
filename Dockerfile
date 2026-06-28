# ----- Build stage -----------
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual env
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --frozen --no-dev

# ---- Runtime stage -----
FROM python:3.13-slim

WORKDIR /app

# Copy virtual env + app from builder
COPY --from=builder /app /app

# Use the venv's python
ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run sets PORT env var (default 8080)
ENV PORT=8080
EXPOSE 8080

# Run with uvicorn, binding to Cloud Run's port
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]