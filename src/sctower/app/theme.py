"""Design system: color tokens, Plotly template and reusable UI components.

The visual identity for this project is Teal (#0F766E) + Amber (#F59E0B) on a
light, high-contrast surface (WCAG AA). Everything visual is centralized here so
pages stay declarative and consistent.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
from dash import html

# --- Color tokens -----------------------------------------------------------
PALETTE = {
    "teal": "#0F766E",
    "teal_dark": "#0B5A53",
    "teal_soft": "#CCFBF1",
    "amber": "#F59E0B",
    "amber_soft": "#FEF3C7",
    "ink": "#0F172A",
    "slate": "#475569",
    "muted": "#94A3B8",
    "mist": "#F8FAFC",
    "card": "#FFFFFF",
    "border": "#E2E8F0",
}

# Operational status colors (shared by inventory pages and charts).
STATUS_COLORS = {
    "stockout": "#DC2626",
    "at_risk": "#F59E0B",
    "healthy": "#0F766E",
    "overstock": "#6366F1",
}

STATUS_LABELS = {
    "stockout": "Rupture",
    "at_risk": "A risque",
    "healthy": "Sain",
    "overstock": "Surstock",
}

# Categorical sequence for multi-series charts (teal/amber-led).
COLORWAY = ["#0F766E", "#F59E0B", "#6366F1", "#0EA5E9", "#DB2777", "#65A30D"]

FONT_FAMILY = "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"


def plotly_template() -> go.layout.Template:
    """Build and register the shared Plotly template ('sctower')."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": FONT_FAMILY, "color": PALETTE["ink"], "size": 13},
        colorway=COLORWAY,
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["card"],
        margin={"l": 56, "r": 24, "t": 48, "b": 44},
        xaxis={
            "gridcolor": PALETTE["border"],
            "linecolor": PALETTE["border"],
            "zeroline": False,
            "ticks": "outside",
            "tickcolor": PALETTE["border"],
        },
        yaxis={
            "gridcolor": PALETTE["border"],
            "linecolor": PALETTE["border"],
            "zeroline": False,
        },
        legend={"orientation": "h", "y": 1.08, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"font": {"family": FONT_FAMILY, "size": 12}},
        colorscale={"sequential": [[0, PALETTE["teal_soft"]], [1, PALETTE["teal_dark"]]]},
    )
    pio.templates["sctower"] = template
    pio.templates.default = "plotly_white+sctower"
    return template


def kpi_card(
    label: str,
    value: str,
    *,
    sub: str | None = None,
    accent: str = "teal",
    icon_path: str | None = None,
) -> dbc.Col:
    """A compact KPI card with a colored accent bar."""
    accent_color = PALETTE.get(accent, accent)
    body = [
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
    ]
    if sub:
        body.append(html.Div(sub, className="kpi-sub"))
    card = html.Div(
        html.Div(body, className="kpi-body"),
        className="kpi-card",
        style={"borderTop": f"3px solid {accent_color}"},
    )
    return dbc.Col(card, xs=12, sm=6, lg=3, className="kpi-col")


def section_title(title: str, subtitle: str | None = None) -> html.Div:
    """A consistent section heading."""
    children: list[html.Div] = [html.Div(title, className="section-title")]
    if subtitle:
        children.append(html.Div(subtitle, className="section-subtitle"))
    return html.Div(children, className="section-head")


def status_badge(status: str) -> html.Span:
    """A colored pill for an inventory status."""
    color = STATUS_COLORS.get(status, PALETTE["slate"])
    label = STATUS_LABELS.get(status, status)
    return html.Span(
        label,
        className="status-badge",
        style={"backgroundColor": color},
    )


def no_data_message() -> html.Div:
    """Friendly placeholder shown when the curated dataset is not available."""
    return html.Div(
        [
            html.Div("Donnees non chargees", className="section-title"),
            html.P(
                "Le jeu de donnees Rossmann curated est introuvable. "
                "Telechargez-le puis preparez-le avec :",
                className="section-subtitle",
            ),
            html.Pre("make data", className="code-block"),
            html.P(
                "Un token Kaggle est requis (~/.kaggle/kaggle.json). "
                "Voir la section Source des donnees du README.",
                className="section-subtitle",
            ),
        ],
        className="empty-state card-panel",
    )
