# Base image with Python 3.13
FROM python:3.13-slim AS base

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

COPY lab1 ./lab1
COPY templates ./templates
COPY README.md .

# Install all deps EXCEPT torch/torchvision (they're huge)
RUN uv pip install --system --no-cache . \
    && pip install --no-cache-dir torch==2.2.2+cpu torchvision==0.17.2+cpu \
       -f https://download.pytorch.org/whl/cpu

# ---- Runtime ----
FROM base AS runtime

COPY --from=builder /usr/local /usr/local

COPY models ./models
COPY lab1 ./lab1
COPY templates ./templates

EXPOSE 8080

CMD ["uvicorn", "lab1.api.api:app", "--host", "0.0.0.0", "--port", "8080"]
