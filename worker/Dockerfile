FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install runtime deps (core + the heavy `engine` extra: paddle) from the
# lockfile; --no-dev drops the test-only group.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra engine

COPY app ./app

EXPOSE 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
