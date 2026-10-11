FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch wheels (sentence-transformers) instead of ~2.5 GB of CUDA libs
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source
COPY . .

# Remove pycache and local env files
RUN find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
RUN rm -f .env

# Non-root user for security
# Chroma volume mount point must exist and be owned by agentuser
RUN mkdir -p /app/data/chroma && useradd -m -u 1000 agentuser && chown -R agentuser:agentuser /app
USER agentuser

# Env defaults (overridden by docker-compose / CI)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_ENV=production

CMD ["python", "main.py"]
