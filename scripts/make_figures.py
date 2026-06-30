"""Generate the README figures (PNG) from the real curated dataset.

Reproducible: requires the curated dataset (``make data``) and kaleido
(``pip install -e ".[dev]"``). Outputs go to ``docs/assets/``.

    python -m scripts.make_figures
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from sctower.app import data as appdata
from sctower.app.theme import PALETTE, STATUS_COLORS, STATUS_LABELS, plotly_template
from sctower.services.pipeline import fit_and_forecast

OUT = Path("docs/assets")
W, H, SCALE = 1000, 420, 2


def _demand_figure() -> go.Figure:
    series = appdata.get_total_series()
    roll = series["y"].rolling(30, min_periods=1).mean()
    fig = go.Figure()
    fig.add_scatter(
        x=series["ds"],
        y=series["y"],
        name="Demande quotidienne",
        line={"color": PALETTE["teal"], "width": 1},
        opacity=0.4,
    )
    fig.add_scatter(
        x=series["ds"],
        y=roll,
        name="Moyenne 30 jours",
        line={"color": PALETTE["amber"], "width": 2.5},
    )
    fig.update_layout(title="Demande agregee - Rossmann (1115 boutiques)")
    return fig


def _backtest_figure() -> go.Figure:
    table = appdata.get_backtest_table().sort_values("wape", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=table["wape"],
            y=table["model"],
            orientation="h",
            marker={"color": [PALETTE["teal"]] + [PALETTE["muted"]] * (len(table) - 1)},
            text=[f"{v:.1f}%" for v in table["wape"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Comparaison des modeles - WAPE (plus bas = mieux)",
        xaxis_title="WAPE (%)",
        yaxis={"autorange": "reversed"},
    )
    return fig


def _forecast_figure() -> go.Figure:
    series = appdata.get_total_series()
    result = fit_and_forecast(series, "gbm", 42)
    hist = series.tail(180)
    fig = go.Figure()
    fig.add_scatter(
        x=hist["ds"],
        y=hist["y"],
        name="Historique",
        line={"color": PALETTE["slate"], "width": 1.5},
    )
    band_x = result.ds.append(result.ds[::-1])
    band_y = np.concatenate([result.yhat_upper, result.yhat_lower[::-1]])
    fig.add_scatter(
        x=band_x,
        y=band_y,
        fill="toself",
        fillcolor="rgba(15,118,110,0.15)",
        line={"color": "rgba(0,0,0,0)"},
        name="Intervalle",
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=result.ds,
        y=result.yhat,
        name="Prevision GBM (42 j)",
        line={"color": PALETTE["amber"], "width": 2.5},
    )
    fig.update_layout(title="Prevision de demande a 42 jours (LightGBM)")
    return fig


def _inventory_figure() -> go.Figure:
    alerts = appdata.get_alerts()
    fig = go.Figure()
    for status in ("stockout", "at_risk", "healthy", "overstock"):
        subset = alerts.loc[alerts["status"] == status]
        if subset.empty:
            continue
        fig.add_scatter(
            x=subset["reorder_point"],
            y=subset["on_hand"],
            mode="markers",
            name=STATUS_LABELS[status],
            marker={
                "color": STATUS_COLORS[status],
                "size": 11,
                "line": {"width": 1, "color": "white"},
            },
        )
    fig.update_layout(
        title="Position de stock vs point de commande (40 boutiques)",
        xaxis_title="Point de commande",
        yaxis_title="Stock actuel (simule)",
    )
    return fig


def main() -> int:
    plotly_template()
    OUT.mkdir(parents=True, exist_ok=True)
    figures = {
        "fig_demand.png": _demand_figure(),
        "fig_backtest.png": _backtest_figure(),
        "fig_forecast.png": _forecast_figure(),
        "fig_inventory.png": _inventory_figure(),
    }
    for name, fig in figures.items():
        path = OUT / name
        fig.write_image(path, width=W, height=H, scale=SCALE)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
