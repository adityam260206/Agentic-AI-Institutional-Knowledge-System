# ── EduMind AI — Application Dockerfile ────────────────────────────────────────
# Multi-stage: deps layer (cacheable) + app layer
# Base: python:3.11-slim (no GPU needed — Ollama runs separately)

FROM python:3.11-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages required by python-docx, sentence-transformers, chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ── App layer ────────────────────────────────────────────────────────────────
FROM deps AS app

WORKDIR /app

# Copy application source
COPY . .

# Create persistent directories (volumes will mount here)
RUN mkdir -p data/staging vector_store/chroma_db state

# Expose FastAPI port
EXPOSE 8000

# Health-check: the web UI is served at /
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/', timeout=5).status_code == 200 else 1)"

# Startup: seed users once, wait for Ollama, launch uvicorn
CMD ["sh", "docker/start.sh"]
