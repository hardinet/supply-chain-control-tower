"""Supply Chain Control Tower.

Demand forecasting, inventory optimization and stockout/overstock alerting on
real retail sales data (Rossmann Store Sales).

The package is organized in layers:

- ``sctower.domain``   pure business logic (forecasting models, inventory math,
  metrics); no I/O, fully unit-tested.
- ``sctower.services`` orchestration (training pipeline, backtesting, scenario
  simulation, alerting).
- ``sctower.io``       adapters to the outside world (file loaders, database).
- ``sctower.app``      the Plotly Dash multi-page user interface.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
