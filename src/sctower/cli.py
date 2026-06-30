"""Command-line interface for the control tower.

Subcommands:
    curate     build the curated parquet from the raw Rossmann CSVs
    backtest   compare forecasting models on the aggregate demand series
    alerts     compute stockout / overstock alerts for the top stores
    serve      launch the Dash application
"""

from __future__ import annotations

import argparse
import sys

from sctower.config import get_settings
from sctower.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _cmd_curate(_: argparse.Namespace) -> int:
    from sctower.io.loaders import curate

    curated = curate()
    logger.info("curate_done", rows=len(curated), stores=int(curated["store"].nunique()))
    return 0


def _cmd_backtest(args: argparse.Namespace) -> int:
    from sctower.domain.forecasting import available_models
    from sctower.io.loaders import load_curated
    from sctower.services.backtest import compare_models
    from sctower.services.pipeline import build_series

    curated = load_curated()
    series = build_series(curated, store=args.store)
    models = list(available_models())
    table = compare_models(series, models, horizon=args.horizon, folds=args.folds)
    print(table.to_string(index=False))
    out = get_settings().data_curated_dir / "backtest_results.csv"
    table.to_csv(out, index=False)
    logger.info("backtest_done", output=str(out))
    return 0


def _cmd_alerts(args: argparse.Namespace) -> int:
    from sctower.io.loaders import load_curated
    from sctower.services.alerts import alerts_summary, build_alerts

    curated = load_curated()
    alerts = build_alerts(curated, n_stores=args.n_stores)
    print(alerts.round(1).to_string(index=False))
    logger.info("alerts_summary", **alerts_summary(alerts))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from sctower.app.main import run

    run(host=args.host, port=args.port, debug=args.debug)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser."""
    settings = get_settings()
    parser = argparse.ArgumentParser(prog="sctower", description="Supply Chain Control Tower")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("curate", help="build the curated dataset").set_defaults(func=_cmd_curate)

    bt = sub.add_parser("backtest", help="compare forecasting models")
    bt.add_argument("--horizon", type=int, default=settings.forecast_horizon_days)
    bt.add_argument("--folds", type=int, default=settings.backtest_folds)
    bt.add_argument("--store", type=int, default=None, help="store id (default: aggregate)")
    bt.set_defaults(func=_cmd_backtest)

    al = sub.add_parser("alerts", help="compute stock alerts")
    al.add_argument("--n-stores", type=int, default=50)
    al.set_defaults(func=_cmd_alerts)

    sv = sub.add_parser("serve", help="run the Dash app")
    sv.add_argument("--host", default="0.0.0.0")  # container-facing by design
    sv.add_argument("--port", type=int, default=8050)
    sv.add_argument("--debug", action="store_true")
    sv.set_defaults(func=_cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point used by both ``python -m sctower.cli`` and the console script."""
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
