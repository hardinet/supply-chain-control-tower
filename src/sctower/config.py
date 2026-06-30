"""Centralized, validated application configuration.

All settings are read from environment variables (prefix ``SCTOWER_``) and an
optional ``.env`` file. Using ``pydantic-settings`` gives us validation, typing
and a single source of truth, so no module ever reads ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: ``src/sctower/config.py`` -> parents[2] is the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Attributes are populated from the environment; defaults are safe for a local
    file-based run with no external services.
    """

    model_config = SettingsConfigDict(
        env_prefix="SCTOWER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Runtime ---
    env: Literal["local", "ci", "staging", "production"] = "local"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # --- Data paths (resolved relative to the repo root when not absolute) ---
    data_raw_dir: Path = Path("data/raw")
    data_curated_dir: Path = Path("data/curated")

    # --- Serving database ---
    db_url: str = "postgresql+psycopg://sctower:sctower@localhost:5432/sctower"

    # --- Inventory policy defaults ---
    service_level: float = Field(default=0.95, ge=0.50, le=0.9999)
    lead_time_days: int = Field(default=7, ge=1, le=180)

    # --- Forecasting defaults ---
    forecast_horizon_days: int = Field(default=42, ge=1, le=365)
    backtest_folds: int = Field(default=4, ge=1, le=12)

    @field_validator("data_raw_dir", "data_curated_dir")
    @classmethod
    def _resolve_under_root(cls, value: Path) -> Path:
        """Make data directories absolute and anchored at the repo root."""
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @property
    def rossmann_train_csv(self) -> Path:
        """Path to the raw Rossmann ``train.csv``."""
        return self.data_raw_dir / "train.csv"

    @property
    def rossmann_store_csv(self) -> Path:
        """Path to the raw Rossmann ``store.csv``."""
        return self.data_raw_dir / "store.csv"

    @property
    def curated_sales_parquet(self) -> Path:
        """Path to the curated, analysis-ready sales table."""
        return self.data_curated_dir / "sales.parquet"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance (single source of truth)."""
    return Settings()
