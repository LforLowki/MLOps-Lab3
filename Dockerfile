# Base image with Python 3.11 (PyTorch CPU compatible)
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- Builder ----
FROM base AS builder

RUN pip install --no-cache-dir uv

COPY pyproject.toml .
COPY uv.lock* .

# Install ONLY API dependencies (no torch!)
RUN uv pip install --system --no-cache ".[notorch]"

COPY lab1 ./lab1
COPY README.md .

# ---- Runtime ----
FROM base AS runtime

COPY --from=builder /usr/local /usr/local

COPY models ./models
COPY lab1 ./lab1

EXPOSE 8080

CMD ["uvicorn", "lab1.api.api:app", "--host", "0.0.0.0", "--port", "8080"]

