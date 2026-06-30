"""Scenario page: interactive what-if on demand and lead time."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from sctower.app import data as appdata
from sctower.app.theme import PALETTE, kpi_card, no_data_message, section_title
from sctower.config import get_settings
from sctower.domain.forecasting import available_models
from sctower.services.scenarios import Scenario, apply_scenario, projected_demand

_HORIZON = 42


def _store_options() -> list[dict[str, str]]:
    options = [{"label": "Toutes les boutiques (agrege)", "value": "total"}]
    options += [{"label": f"Boutique {s}", "value": str(s)} for s in appdata.get_store_ids()[:60]]
    return options


def _slider_group(label: str, component: dcc.Slider) -> html.Div:
    return html.Div(
        [html.Label(label, className="control-label"), component],
        className="control-group control-grow",
    )


def layout() -> html.Div:
    """Render the scenario simulation page."""
    if not appdata.is_data_available():
        return html.Div(no_data_message())

    return html.Div(
        [
            section_title(
                "Simulation de scenarios",
                "Mesurez l'impact d'un choc de demande ou d'un allongement du delai.",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Perimetre", className="control-label"),
                            dcc.Dropdown(
                                id="sc-store",
                                options=_store_options(),
                                value="total",
                                clearable=False,
                                className="control-input",
                            ),
                        ],
                        className="control-group",
                    ),
                    _slider_group(
                        "Variation de demande (%)",
                        dcc.Slider(
                            id="sc-demand",
                            min=-50,
                            max=50,
                            step=5,
                            value=20,
                            marks={v: f"{v:+d}" for v in range(-50, 51, 25)},
                        ),
                    ),
                    _slider_group(
                        "Delai de reappro (+/- jours)",
                        dcc.Slider(
                            id="sc-lead",
                            min=-5,
                            max=14,
                            step=1,
                            value=0,
                            marks={v: str(v) for v in range(-5, 15, 5)},
                        ),
                    ),
                    _slider_group(
                        "Niveau de service",
                        dcc.Slider(
                            id="sc-service",
                            min=0.80,
                            max=0.99,
                            step=0.01,
                            value=0.95,
                            marks={0.80: "80%", 0.90: "90%", 0.95: "95%", 0.99: "99%"},
                        ),
                    ),
                ],
                className="control-bar card-panel",
            ),
            dcc.Loading(html.Div(id="sc-kpis", className="kpi-row")),
            html.Div(dcc.Graph(id="sc-graph"), className="card-panel"),
        ],
        className="page",
    )


def _delta_pct(base: float, value: float) -> str:
    if base == 0:
        return "n/a"
    return f"{(value - base) / base * 100:+.0f}% vs base"


@callback(
    Output("sc-kpis", "children"),
    Output("sc-graph", "figure"),
    Input("sc-store", "value"),
    Input("sc-demand", "value"),
    Input("sc-lead", "value"),
    Input("sc-service", "value"),
)
def _simulate(
    store_value: str, demand_pct: float, lead_delta: float, service_level: float
) -> tuple[html.Div, go.Figure]:
    settings = get_settings()
    models = list(available_models())
    model_name = "gbm" if "gbm" in models else models[0]

    base_forecast = appdata.get_base_forecast(store_value, model_name, _HORIZON)
    mean, std = appdata.get_demand_stats(store_value)

    base = apply_scenario(
        Scenario("Base"),
        base_forecast=base_forecast,
        avg_daily_demand=mean,
        demand_std=std,
        base_lead_time_days=settings.lead_time_days,
        service_level=service_level,
    )
    scenario = apply_scenario(
        Scenario(
            "Scenario", demand_multiplier=1.0 + demand_pct / 100.0, lead_time_delta_days=lead_delta
        ),
        base_forecast=base_forecast,
        avg_daily_demand=mean,
        demand_std=std,
        base_lead_time_days=settings.lead_time_days,
        service_level=service_level,
    )

    kpis = html.Div(
        [
            kpi_card(
                "Stock de securite",
                f"{scenario.policy.safety_stock:,.0f}",
                sub=_delta_pct(base.policy.safety_stock, scenario.policy.safety_stock),
                accent="amber",
            ),
            kpi_card(
                "Point de commande",
                f"{scenario.policy.reorder_point:,.0f}",
                sub=_delta_pct(base.policy.reorder_point, scenario.policy.reorder_point),
            ),
            kpi_card(
                "Quantite economique",
                f"{scenario.policy.order_quantity:,.0f}",
                sub="EOQ",
            ),
            kpi_card(
                "Demande projetee",
                f"{projected_demand(scenario):,.0f}",
                sub=_delta_pct(projected_demand(base), projected_demand(scenario)),
                accent="amber",
            ),
        ],
        className="row g-3",
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=base.forecast.ds,
            y=base.forecast.yhat,
            name="Base",
            mode="lines",
            line={"color": PALETTE["slate"], "width": 2, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scenario.forecast.ds,
            y=scenario.forecast.yhat,
            name="Scenario",
            mode="lines",
            line={"color": PALETTE["teal"], "width": 2.5},
        )
    )
    fig.update_layout(height=380, title="Prevision: base vs scenario")
    return kpis, fig


dash.register_page(
    __name__, path="/scenarios", name="Simulation", title="Simulation", layout=layout
)
