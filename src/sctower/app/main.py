"""Dash application entrypoint.

Creates the multi-page app, registers a ``/health`` endpoint on the underlying
Flask server, and exposes ``server`` for production WSGI servers (gunicorn). The
four pages are registered by importing their modules.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, page_container
from flask import Flask, Response, jsonify

from sctower.app import data as appdata
from sctower.app.theme import PALETTE, plotly_template
from sctower.logging import configure_logging, get_logger

logger = get_logger(__name__)

FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"

NAV_LINKS = [
    ("Vue d'ensemble", "/"),
    ("Prevision", "/forecast"),
    ("Stocks", "/inventory"),
    ("Simulation", "/scenarios"),
]


def _navbar() -> dbc.Navbar:
    """Top navigation bar with brand, page links and a data-source badge."""
    available = appdata.is_data_available()
    badge_text = "Donnees reelles Rossmann" if available else "Donnees non chargees"
    badge_class = "data-badge ok" if available else "data-badge warn"
    links = [
        dbc.NavItem(dbc.NavLink(label, href=href, active="exact", className="nav-link-custom"))
        for label, href in NAV_LINKS
    ]
    return dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    html.Div(
                        [
                            html.Span("Supply Chain", className="brand-strong"),
                            html.Span("Control Tower", className="brand-light"),
                        ],
                        className="brand",
                    ),
                    href="/",
                    className="brand-link",
                ),
                dbc.Nav(links, className="ms-auto", navbar=True),
                html.Span(badge_text, className=badge_class),
            ],
            fluid=True,
        ),
        color=PALETTE["ink"],
        dark=True,
        className="app-navbar",
        sticky="top",
    )


def _footer() -> html.Footer:
    return html.Footer(
        html.Div(
            "Supply Chain Control Tower - donnees Rossmann Store Sales (Kaggle). "
            "Projet portfolio data/IA.",
            className="footer-inner",
        ),
        className="app-footer",
    )


def create_app() -> Dash:
    """Build and configure the Dash application."""
    configure_logging()
    plotly_template()

    server = Flask(__name__)

    @server.get("/health")
    def health() -> tuple[Response, int]:
        return jsonify(status="ok", data_available=appdata.is_data_available()), 200

    app = Dash(
        __name__,
        server=server,
        use_pages=True,
        pages_folder="",
        external_stylesheets=[dbc.themes.BOOTSTRAP, FONTS],
        title="Supply Chain Control Tower",
        suppress_callback_exceptions=True,
        update_title=None,
    )

    # Importing the page modules triggers their dash.register_page() calls.
    from sctower.app.pages import forecast, inventory, overview, scenarios  # noqa: F401

    app.layout = html.Div(
        [
            _navbar(),
            html.Main(dbc.Container(page_container, fluid=True), className="app-main"),
            _footer(),
        ],
        className="app-root",
    )
    logger.info("app_created", pages=[p["path"] for p in dash.page_registry.values()])
    return app


app = create_app()
server = app.server


def run(host: str = "0.0.0.0", port: int = 8050, *, debug: bool = False) -> None:
    """Run the development server (used by ``sctower serve``)."""
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run(debug=True)
