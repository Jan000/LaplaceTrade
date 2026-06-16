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

# Baked config is only a *seed*. At runtime the entrypoint copies it into the /app/config
# volume on first boot, so dashboard config edits (autostart, trade symbols, feature
# toggles, …) survive redeploys instead of resetting to the image defaults.
COPY config ./config_default
COPY scripts ./scripts

# Persisted state lives on mounted volumes (DB, models, cache, and the editable config).
VOLUME ["/app/data", "/app/models", "/app/.cache", "/app/config"]
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

# Seed /app/config from the baked defaults on first boot (only when the volume is empty),
# then exec the CMD. Bind to all interfaces inside the container (a reverse proxy / Coolify
# terminates TLS in front). --proxy-headers so X-Forwarded-* from the proxy are trusted.
ENTRYPOINT ["/bin/sh", "-c", \
  "if [ ! -f /app/config/config.yaml ] && [ -d /app/config_default ]; then mkdir -p /app/config && cp -a /app/config_default/. /app/config/ 2>/dev/null || true; fi; exec \"$@\"", \
  "--"]
CMD ["uvicorn", "cryptotrader.api.server:app", "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
