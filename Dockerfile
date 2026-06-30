# syntax=docker/dockerfile:1

# ---- Builder: install the package and its (compiled) dependencies ----
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[prophet,postgres,kaggle]"

# ---- Runtime: slim image with only the virtualenv ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    SCTOWER_ENV=production \
    SCTOWER_LOG_JSON=true \
    SCTOWER_DATA_RAW_DIR=/app/data/raw \
    SCTOWER_DATA_CURATED_DIR=/app/data/curated

RUN useradd --create-home --uid 10001 appuser
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY scripts ./scripts
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/data/raw /app/data/curated \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8050/health').getcode()==200 else 1)"

ENTRYPOINT ["entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "--timeout", "120", "sctower.app.main:server"]
