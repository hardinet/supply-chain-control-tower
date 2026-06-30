#!/bin/sh
# Container entrypoint: fetch and curate the dataset on first boot when Kaggle
# credentials are provided, then hand off to the given command (gunicorn).
set -e

CURATED="${SCTOWER_DATA_CURATED_DIR:-/app/data/curated}/sales.parquet"

if [ ! -f "$CURATED" ]; then
  if [ -n "$KAGGLE_API_TOKEN" ] || { [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ]; }; then
    echo "[entrypoint] curated dataset missing - downloading from Kaggle..."
    if python -m scripts.download_data && python -m sctower.cli curate; then
      echo "[entrypoint] dataset ready."
    else
      echo "[entrypoint] WARNING: data preparation failed; app will show a no-data state."
    fi
  else
    echo "[entrypoint] No Kaggle credentials and no curated dataset; app starts in no-data state."
  fi
fi

exec "$@"
