# ADR 0001 - Layered architecture (domain / services / io / app)

- Status: accepted
- Date: 2026-06-30

## Context

The control tower mixes pure business logic (forecasting math, inventory
policies), orchestration (pipelines, backtesting), I/O (files, database) and a
web UI. Without clear boundaries these concerns leak into each other, making the
code hard to test and reason about.

## Decision

Organize the package into strict layers with a one-directional dependency rule
(`app` -> `services` -> `domain`, and `io` as an adapter used by `services`):

- `domain`   pure functions and small classes, no I/O, no framework imports.
  Forecasting models, inventory mathematics and metrics live here.
- `services` orchestration that composes domain logic over data (series
  building, rolling-origin backtesting, scenario simulation, alerting).
- `io`       adapters to the outside world (Rossmann loaders, SQLAlchemy access).
- `app`      the Plotly Dash presentation layer; it only calls `services`/`domain`.

Configuration is centralized in `config.py` (pydantic-settings); logging in
`logging.py` (structlog).

## Consequences

- The domain layer is trivially unit-testable (no mocks, no fixtures beyond small
  DataFrames) and reaches near-100% coverage.
- Swapping the UI, the database or the dataset touches only one layer.
- A small amount of boilerplate (explicit interfaces, dependency passing) is the
  price paid for the isolation.
