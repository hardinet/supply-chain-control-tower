# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding: typed Python package layout (`domain` / `services` / `io` / `app`),
  tooling (ruff, black, mypy, pre-commit), packaging metadata.
- Forecasting layer: pluggable models (seasonal-naive, SARIMA, LightGBM, Prophet)
  behind a single interface with a registry; Prophet isolated as an optional extra.
- Inventory mathematics: safety stock, reorder point, EOQ, days-of-cover and a
  continuous-review `(s, Q)` policy with stockout/overstock classification.
- Services: series building, rolling-origin backtesting, scenario simulation, alerting.
- IO: Rossmann loaders and curation, SQLAlchemy/Postgres access, Kaggle download script.
- Dash multi-page application (overview, forecast, inventory, scenarios) with a
  Teal/Amber design system and a `/health` endpoint.
- Tests (76, 92% coverage), GitHub Actions CI (lint, types, tests, Docker build),
  multi-stage Dockerfile, compose stack and Render blueprint.
- Documentation: case-study README with real backtest results and figures, two ADRs,
  animated SVG banner.

[Unreleased]: https://github.com/hardinet/supply-chain-control-tower/commits/main
