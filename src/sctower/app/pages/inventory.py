"""Inventory page: policy levels and stockout / overstock alerts per store."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import dash
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from sctower.app import data as appdata
from sctower.app.theme import (
    PALETTE,
    STATUS_COLORS,
    STATUS_LABELS,
    kpi_card,
    no_data_message,
    section_title,
)

_STATUS_ORDER = ["stockout", "at_risk", "healthy", "overstock"]
_TABLE_COLUMNS = [
    ("store", "Boutique"),
    ("avg_daily_demand", "Demande/j"),
    ("safety_stock", "Stock secu."),
    ("reorder_point", "Pt commande"),
    ("on_hand", "Stock actuel"),
    ("days_of_cover", "Jours couv."),
    ("status", "Statut"),
]


def _position_figure() -> go.Figure:
    alerts = appdata.get_alerts()
    fig = go.Figure()
    for status in _STATUS_ORDER:
        subset = alerts.loc[alerts["status"] == status]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["reorder_point"],
                y=subset["on_hand"],
                name=STATUS_LABELS[status],
                mode="markers",
                marker={
                    "color": STATUS_COLORS[status],
                    "size": 11,
                    "line": {"width": 1, "color": "white"},
                },
                customdata=subset[["store", "days_of_cover"]],
                hovertemplate=(
                    "Boutique %{customdata[0]}<br>Point de commande %{x:.0f}"
                    "<br>Stock actuel %{y:.0f}<br>Jours de couverture %{customdata[1]:.1f}"
                    "<extra></extra>"
                ),
            )
        )
    max_rop = float(alerts["reorder_point"].max()) if not alerts.empty else 1.0
    fig.add_trace(
        go.Scatter(
            x=[0, max_rop],
            y=[0, max_rop],
            mode="lines",
            line={"color": PALETTE["muted"], "dash": "dash", "width": 1},
            name="Stock = point de commande",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        height=440,
        title="Position de stock vs point de commande",
        xaxis_title="Point de commande",
        yaxis_title="Stock actuel (simule)",
    )
    return fig


def _kpis() -> html.Div:
    counts = appdata.get_alerts()["status"].value_counts()
    cards = [
        kpi_card(
            STATUS_LABELS[s],
            f"{int(counts.get(s, 0))}",
            sub="boutiques",
            accent="amber" if s in ("stockout", "at_risk") else "teal",
        )
        for s in _STATUS_ORDER
    ]
    return html.Div(html.Div(cards, className="row g-3"), className="kpi-row")


def layout() -> html.Div:
    """Render the inventory page."""
    if not appdata.is_data_available():
        return html.Div(no_data_message())

    return html.Div(
        [
            section_title(
                "Stocks et alertes",
                "Politique de reapprovisionnement et detection rupture / surstock.",
            ),
            html.Div(
                "Le stock actuel est simule de maniere deterministe (le jeu Rossmann "
                "ne contient pas de niveaux de stock) ; la politique est calculee sur la "
                "demande reelle.",
                className="note-banner",
            ),
            _kpis(),
            html.Div(dcc.Graph(figure=_position_figure()), className="card-panel"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Filtrer par statut", className="control-label"),
                            dcc.Dropdown(
                                id="inv-status",
                                options=[{"label": "Tous", "value": "all"}]
                                + [{"label": STATUS_LABELS[s], "value": s} for s in _STATUS_ORDER],
                                value="all",
                                clearable=False,
                                className="control-input",
                            ),
                        ],
                        className="control-group",
                    ),
                    dash_table.DataTable(
                        id="inv-table",
                        columns=[{"name": label, "id": cid} for cid, label in _TABLE_COLUMNS],
                        sort_action="native",
                        page_size=12,
                        style_as_list_view=True,
                        style_header={
                            "backgroundColor": PALETTE["mist"],
                            "fontWeight": "600",
                            "border": "none",
                        },
                        style_cell={
                            "fontFamily": "Inter, sans-serif",
                            "padding": "10px 14px",
                            "border": "none",
                            "fontSize": "13px",
                        },
                        style_data_conditional=[
                            {
                                "if": {
                                    "filter_query": f'{{status}} = "{s}"',
                                    "column_id": "status",
                                },
                                "color": STATUS_COLORS[s],
                                "fontWeight": "600",
                            }
                            for s in _STATUS_ORDER
                        ],
                    ),
                ],
                className="card-panel",
            ),
        ],
        className="page",
    )


@callback(Output("inv-table", "data"), Input("inv-status", "value"))
def _filter_table(status: str) -> list[dict[Hashable, Any]]:
    alerts = appdata.get_alerts().copy()
    if status != "all":
        alerts = alerts.loc[alerts["status"] == status]
    alerts = alerts.round(
        {
            "avg_daily_demand": 0,
            "safety_stock": 0,
            "reorder_point": 0,
            "on_hand": 0,
            "days_of_cover": 1,
        }
    )
    cols = [cid for cid, _ in _TABLE_COLUMNS]
    return alerts[cols].to_dict("records")


dash.register_page(__name__, path="/inventory", name="Stocks", title="Stocks", layout=layout)
