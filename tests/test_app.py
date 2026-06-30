"""Smoke tests for the Dash application wiring.

These guard the page registration and health endpoint without requiring data:
they would have caught the missing-layout routing bug.
"""

from __future__ import annotations

import dash

from sctower.app.main import server


def test_pages_registered_with_layouts() -> None:
    paths = {page["path"] for page in dash.page_registry.values()}
    assert {"/", "/forecast", "/inventory", "/scenarios"} <= paths
    assert all(page.get("layout") is not None for page in dash.page_registry.values())


def test_health_endpoint_ok() -> None:
    response = server.test_client().get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert "data_available" in body


def test_index_is_served() -> None:
    assert server.test_client().get("/").status_code == 200
