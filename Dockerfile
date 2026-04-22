FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[platform]"

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY platform/ platform/
COPY alembic.ini ./
COPY alembic/ alembic/

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "platform.backend.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
CMD ["dramatiq", "platform.worker.actors", "--processes", "1", "--threads", "4"]
