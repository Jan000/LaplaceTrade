# CryptoTrader — 24/7 dashboard + trading engine
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Build deps for lightgbm/numpy wheels are usually unneeded (manylinux), but libgomp is.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .          # fastapi/uvicorn are core deps; the legacy streamlit extra is not needed

COPY config ./config
COPY scripts ./scripts

# Persisted data (DB, models, cache) live on a mounted volume.
VOLUME ["/app/data", "/app/models", "/app/.cache"]
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# Bind to all interfaces inside the container (a reverse proxy / Coolify terminates TLS in
# front). --proxy-headers so X-Forwarded-* from the proxy are trusted.
CMD ["uvicorn", "cryptotrader.api.server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
