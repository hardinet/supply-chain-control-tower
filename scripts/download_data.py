"""Download the Rossmann Store Sales dataset from Kaggle (reproducible).

Requirements:
    1. A free Kaggle account.
    2. An API token saved at ``~/.kaggle/kaggle.json`` (Kaggle > Account >
       Create New API Token), or the ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``
       environment variables.
    3. Acceptance of the competition rules on the dataset page (one-time click).

Usage:
    python -m scripts.download_data
"""

from __future__ import annotations

import gzip
import shutil
import sys
import zipfile
from pathlib import Path

from sctower.config import get_settings
from sctower.logging import get_logger

logger = get_logger(__name__)

COMPETITION = "rossmann-store-sales"
REQUIRED_FILES = ("train.csv", "store.csv")


def _extract_archives(raw_dir: Path) -> None:
    """Unzip ``.zip`` files and decompress ``.gz`` files in place."""
    for archive in raw_dir.glob("*.zip"):
        logger.info("extracting_zip", file=archive.name)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw_dir)
        archive.unlink()
    for gz in raw_dir.glob("*.gz"):
        target = gz.with_suffix("")
        logger.info("extracting_gz", file=gz.name)
        with gzip.open(gz, "rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        gz.unlink()


def download(raw_dir: Path | None = None) -> Path:
    """Download and extract the competition files into the raw data directory."""
    settings = get_settings()
    raw_dir = raw_dir or settings.data_raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if all((raw_dir / name).exists() for name in REQUIRED_FILES):
        logger.info("already_present", path=str(raw_dir))
        return raw_dir

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - optional extra
        raise SystemExit(
            "The 'kaggle' package is required. Install with: pip install -e '.[kaggle]'"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    logger.info("downloading", competition=COMPETITION, dest=str(raw_dir))
    api.competition_download_files(COMPETITION, path=str(raw_dir), quiet=False)
    _extract_archives(raw_dir)

    missing = [name for name in REQUIRED_FILES if not (raw_dir / name).exists()]
    if missing:
        raise SystemExit(f"download incomplete, missing: {missing}")
    logger.info("download_complete", files=list(REQUIRED_FILES))
    return raw_dir


def main() -> int:
    """CLI entrypoint."""
    try:
        download()
    except SystemExit as exc:
        logger.error("download_failed", reason=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
