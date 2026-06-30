"""Forecast page: per-store / aggregate forecasting with model comparison."""

from __future__ import annotations

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from sctower.app import data as appdata
from sctower.app.theme import PALETTE, no_data_message, section_title
from sctower.domain.forecasting import available_models
from sctower.services.pipeline import fit_and_forecast

dash.register_page(__name__, path="/forecast", name="Prevision", title="Prevision")

_HISTORY_TAIL = 180


def _series_for(store_value: str) -> pd.DataFrame:
    if store_value == "total":
        return appdata.get_total_series()
    return appdata.get_store_series(int(store_value))


def _store_options() -> list[dict[str, str]]:
    options = [{"label": "Toutes les boutiques (agrege)", "value": "total"}]
    options += [{"label": f"Boutique {s}", "value": str(s)} for s in appdata.get_store_ids()[:60]]
    return options


def _forecast_figure(series: pd.DataFrame, model_name: str, horizon: int) -> go.Figure:
    result = fit_and_forecast(series, model_name, horizon)
    hist = series.tail(_HISTORY_TAIL)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["ds"],
            y=hist["y"],
            name="Historique",
            mode="lines",
            line={"color": PALETTE["slate"], "width": 1.5},
        )
    )
    if result.yhat_upper is not None and result.yhat_lower is not None:
        fig.add_trace(
            go.Scatter(
                x=list(result.ds) + list(result.ds[::-1]),
                y=list(result.yhat_upper) + list(result.yhat_lower[::-1]),
                fill="toself",
                fillcolor="rgba(15,118,110,0.15)",
                line={"color": "rgba(0,0,0,0)"},
                name="Intervalle de prediction",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=result.ds,
            y=result.yhat,
            name=f"Prevision ({model_name})",
            mode="lines",
            line={"color": PALETTE["amber"], "width": 2.5},
        )
    )
    fig.update_layout(height=420, title=f"Prevision a {horizon} jours")
    return fig


def _backtest_table() -> dash_table.DataTable:
    table = appdata.get_backtest_table().copy()
    for col in ("wape", "mape", "mae", "bias", "smape", "rmse", "bias_pct"):
        if col in table.columns:
            table[col] = table[col].round(2)
    columns = [c for c in ("model", "folds", "wape", "mape", "mae", "bias") if c in table.columns]
    return dash_table.DataTable(
        data=table[columns].to_dict("records"),
        columns=[{"name": c.upper(), "id": c} for c in columns],
        style_as_list_view=True,
        style_header={
            "backgroundColor": PALETTE["mist"],
            "fontWeight": "600",
            "border": "none",
            "color": PALETTE["ink"],
        },
        style_cell={
            "fontFamily": "Inter, sans-serif",
            "padding": "10px 14px",
            "border": "none",
            "fontSize": "13px",
        },
        style_data_conditional=[
            {
                "if": {"row_index": 0},
                "backgroundColor": PALETTE["teal_soft"],
                "fontWeight": "600",
            }
        ],
    )


def layout() -> html.Div:
    """Render the forecast page."""
    if not appdata.is_data_available():
        return html.Div(no_data_message())

    models = list(available_models())
    default_model = "gbm" if "gbm" in models else models[0]

    controls = html.Div(
        [
            html.Div(
                [
                    html.Label("Perimetre", className="control-label"),
                    dcc.Dropdown(
                        id="fc-store",
                        options=_store_options(),
                        value="total",
                        clearable=False,
                        className="control-input",
                    ),
                ],
                className="control-group",
            ),
            html.Div(
                [
                    html.Label("Modele", className="control-label"),
                    dcc.Dropdown(
                        id="fc-model",
                        options=[{"label": m, "value": m} for m in models],
                        value=default_model,
                        clearable=False,
                        className="control-input",
                    ),
                ],
                className="control-group",
            ),
            html.Div(
                [
                    html.Label("Horizon (jours)", className="control-label"),
                    dcc.Slider(
                        id="fc-horizon",
                        min=7,
                        max=84,
                        step=7,
                        value=42,
                        marks={d: str(d) for d in range(7, 85, 14)},
                    ),
                ],
                className="control-group control-grow",
            ),
        ],
        className="control-bar card-panel",
    )

    return html.Div(
        [
            section_title(
                "Prevision de la demande",
                "Comparez les modeles et projetez la demande avec intervalle de prediction.",
            ),
            controls,
            html.Div(dcc.Loading(dcc.Graph(id="fc-graph")), className="card-panel"),
            html.Div(
                [
                    section_title(
                        "Comparaison des modeles (backtesting rolling-origin)",
                        "Metriques hors-echantillon, meilleure ligne (WAPE) surlignee.",
                    ),
                    _backtest_table(),
                ],
                className="card-panel",
            ),
        ],
        className="page",
    )


@callback(
    Output("fc-graph", "figure"),
    Input("fc-store", "value"),
    Input("fc-model", "value"),
    Input("fc-horizon", "value"),
)
def _update_forecast(store_value: str, model_name: str, horizon: int) -> go.Figure:
    series = _series_for(store_value)
    return _forecast_figure(series, model_name, int(horizon))
