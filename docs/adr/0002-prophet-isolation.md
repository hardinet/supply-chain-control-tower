# ADR 0002 - Isolate Prophet as an optional dependency

- Status: accepted
- Date: 2026-06-30

## Context

Prophet (and its cmdstan/stan backend) is valuable for interpretable seasonal
forecasting, but its installation is fragile on Python 3.13 and on Windows, where
wheels are frequently missing or fail to build. The rest of the stack (pandas,
LightGBM, statsmodels) installs cleanly on Python 3.12 and 3.13.

## Decision

- Make Prophet an **optional extra** (`pip install ".[prophet]"`) rather than a
  core dependency.
- The forecasting registry (`sctower.domain.forecasting`) imports the Prophet
  model inside a `contextlib.suppress(ImportError)` block, so the model registers
  only when Prophet is importable and `available_models()` reflects reality.
- Prophet is installed and exercised in Docker and CI (Python 3.12 on Linux),
  where it is reliable; local development on Python 3.13 works without it.

## Consequences

- The application, tests and CI never break because of a Prophet install issue.
- Forecast comparisons gracefully include or exclude Prophet depending on the
  environment, which is surfaced honestly in the UI and the backtest table.
- This is a concrete, defensible example of designing around a fragile
  dependency rather than letting it dictate the whole project's runtime.
