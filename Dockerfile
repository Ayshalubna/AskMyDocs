# ─────────────────────────────────────────────
# Stage 1: Builder — install dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps into a prefix
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="RAG-DocQA" \
      description="RAG-Powered Document Q&A API" \
      version="1.0.0"

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1001 appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create storage dirs with correct ownership
RUN mkdir -p /app/storage/uploads /app/storage/vectorstore \
    && chown -R appuser:appuser /app

USER appuser

# Pre-download embedding model at build time (optional; speeds up first run)
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log"]
