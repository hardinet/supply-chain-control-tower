"""Overview page: headline KPIs, demand trend and alert mix."""

from __future__ import annotations

import dash
import plotly.graph_objects as go
from dash import dcc, html

from sctower.app import data as appdata
from sctower.app.theme import (
    PALETTE,
    STATUS_COLORS,
    STATUS_LABELS,
    kpi_card,
    no_data_message,
    section_title,
)


def _demand_trend_figure() -> go.Figure:
    series = appdata.get_total_series()
    roll = series["y"].rolling(30, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["ds"],
            y=series["y"],
            name="Demande quotidienne",
            mode="lines",
            line={"color": PALETTE["teal"], "width": 1},
            opacity=0.45,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["ds"],
            y=roll,
            name="Moyenne 30 jours",
            mode="lines",
            line={"color": PALETTE["amber"], "width": 2.5},
        )
    )
    fig.update_layout(height=360, title="Demande agregee (toutes boutiques)")
    return fig


def _status_figure() -> go.Figure:
    alerts = appdata.get_alerts()
    counts = alerts["status"].value_counts()
    order = ["stockout", "at_risk", "healthy", "overstock"]
    labels = [STATUS_LABELS[s] for s in order]
    values = [int(counts.get(s, 0)) for s in order]
    colors = [STATUS_COLORS[s] for s in order]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker={"colors": colors},
            sort=False,
            textinfo="value",
        )
    )
    fig.update_layout(height=360, title="Repartition des statuts de stock")
    return fig


def _weekday_figure() -> go.Figure:
    curated = appdata.get_app_data().curated
    by_dow = curated.groupby("day_of_week")["sales"].mean()
    names = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    fig = go.Figure(
        go.Bar(
            x=names,
            y=[by_dow.get(i, 0) for i in range(1, 8)],
            marker={"color": PALETTE["teal"]},
        )
    )
    fig.update_layout(height=300, title="Vente moyenne par jour de semaine")
    return fig


def layout() -> html.Div:
    """Render the overview page (called fresh on navigation)."""
    if not appdata.is_data_available():
        return html.Div(no_data_message())

    series = appdata.get_total_series()
    alerts = appdata.get_alerts()
    open_days = series.loc[series["y"] > 0, "y"]
    counts = alerts["status"].value_counts()

    total_sales = float(series["y"].sum())
    avg_daily = float(open_days.mean()) if not open_days.empty else 0.0
    n_stores = len(appdata.get_store_ids())
    at_risk = int(counts.get("stockout", 0) + counts.get("at_risk", 0))

    kpis = dash.html.Div(
        dash.html.Div(
            [
                kpi_card("Demande cumulee", f"{total_sales / 1e6:,.1f} M", sub="unites vendues"),
                kpi_card(
                    "Demande / jour ouvre", f"{avg_daily:,.0f}", sub="moyenne", accent="amber"
                ),
                kpi_card("Boutiques", f"{n_stores}", sub="dans le perimetre"),
                kpi_card("Alertes stock", f"{at_risk}", sub="rupture + a risque", accent="amber"),
            ],
            className="row g-3",
        ),
        className="kpi-row",
    )

    return html.Div(
        [
            section_title(
                "Vue d'ensemble",
                "Sante de la demande et des stocks sur l'ensemble du reseau.",
            ),
            kpis,
            html.Div(
                [
                    html.Div(
                        dcc.Graph(figure=_demand_trend_figure()), className="card-panel col-lg-8"
                    ),
                    html.Div(dcc.Graph(figure=_status_figure()), className="card-panel col-lg-4"),
                ],
                className="row g-3 chart-row",
            ),
            html.Div(
                html.Div(dcc.Graph(figure=_weekday_figure()), className="card-panel col-12"),
                className="row g-3 chart-row",
            ),
        ],
        className="page",
    )


dash.register_page(__name__, path="/", name="Vue d'ensemble", title="Vue d'ensemble", layout=layout)
