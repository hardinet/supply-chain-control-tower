"""Load and curate the Rossmann dataset into an analysis-ready table.

Raw inputs are the two Kaggle CSVs (``train.csv`` and ``store.csv``). Curation
merges them, normalizes types and names, and writes a single tidy parquet table
(one row per store and day) that the rest of the system consumes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sctower.config import Settings, get_settings
from sctower.logging import get_logger

logger = get_logger(__name__)

# Curated schema (snake_case, stable contract for downstream code).
CURATED_COLUMNS = [
    "store",
    "ds",
    "day_of_week",
    "sales",
    "customers",
    "open",
    "promo",
    "state_holiday",
    "school_holiday",
    "store_type",
    "assortment",
    "competition_distance",
]


def load_raw_rossmann(
    train_csv: Path | None = None,
    store_csv: Path | None = None,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the raw Rossmann CSVs into DataFrames."""
    settings = settings or get_settings()
    train_path = train_csv or settings.rossmann_train_csv
    store_path = store_csv or settings.rossmann_store_csv
    for path in (train_path, store_path):
        if not path.exists():
            raise FileNotFoundError(
                f"missing raw data file: {path}. Run `make data` to download it."
            )
    logger.info("loading_raw", train=str(train_path), store=str(store_path))
    train = pd.read_csv(train_path, parse_dates=["Date"], dtype={"StateHoliday": "string"})
    store = pd.read_csv(store_path)
    return train, store


def curate_rossmann(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Merge and normalize raw frames into the curated schema.

    - merges store attributes onto each daily sales row;
    - renames to snake_case and casts types;
    - converts ``StateHoliday`` (0/a/b/c) into a binary flag.
    """
    merged = train.merge(store, on="Store", how="left")
    out = pd.DataFrame(
        {
            "store": merged["Store"].astype("int32"),
            "ds": pd.to_datetime(merged["Date"]),
            "day_of_week": merged["DayOfWeek"].astype("int8"),
            "sales": merged["Sales"].astype("float64"),
            "customers": merged["Customers"].astype("float64"),
            "open": merged["Open"].fillna(1).astype("int8"),
            "promo": merged["Promo"].astype("int8"),
            "state_holiday": (merged["StateHoliday"].fillna("0") != "0").astype("int8"),
            "school_holiday": merged["SchoolHoliday"].astype("int8"),
            "store_type": merged["StoreType"].astype("string"),
            "assortment": merged["Assortment"].astype("string"),
            "competition_distance": merged["CompetitionDistance"].astype("float64"),
        }
    )
    out = out.sort_values(["store", "ds"]).reset_index(drop=True)
    return out[CURATED_COLUMNS]


def curate(
    settings: Settings | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Run the full curation step and (optionally) persist the parquet output."""
    settings = settings or get_settings()
    train, store = load_raw_rossmann(settings=settings)
    curated = curate_rossmann(train, store)
    if write:
        out_path = settings.curated_sales_parquet
        out_path.parent.mkdir(parents=True, exist_ok=True)
        curated.to_parquet(out_path, index=False)
        logger.info("curated_written", path=str(out_path), rows=len(curated))
    return curated


def load_curated(settings: Settings | None = None) -> pd.DataFrame:
    """Load the curated sales table, raising a clear error if it is missing."""
    settings = settings or get_settings()
    path = settings.curated_sales_parquet
    if not path.exists():
        raise FileNotFoundError(f"curated dataset not found at {path}. Run `make data` first.")
    return pd.read_parquet(path)
