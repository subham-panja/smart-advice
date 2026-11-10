"""
Test Swagger & OpenAPI Endpoints and Schemas
============================================
Validates Swagger UI, Swagger JSON schema, OpenAPI specification export,
and namespace endpoint functionality.
"""

import json
import os

import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_swagger_ui_endpoint(client):
    """Ensure /api/docs serves the Swagger UI HTML."""
    res = client.get("/api/docs")
    assert res.status_code == 200
    assert b"swagger-ui" in res.data or b"Swagger UI" in res.data or b"flask-restx" in res.data


def test_swagger_json_schema(client):
    """Ensure /api/swagger.json serves valid OpenAPI 2.0 schema."""
    res = client.get("/api/swagger.json")
    assert res.status_code == 200
    data = res.get_json()
    assert data is not None

    # Validate info metadata
    info = data.get("info", {})
    assert info.get("title") == "Smart Advice Trading API"
    assert info.get("version") == "1.0.0"

    # Validate all 6 namespaces are present as tags
    tag_names = {t.get("name") for t in data.get("tags", [])}
    expected_tags = {"analysis", "positions", "orchestrator", "config", "backtests", "intraday"}
    assert expected_tags.issubset(tag_names), f"Missing tags: {expected_tags - tag_names}"

    # Validate paths exist
    paths = data.get("paths", {})
    assert len(paths) >= 25
    assert "/analysis/health" in paths or "/analysis/" in paths
    assert "/positions/" in paths or "/positions" in paths
    assert "/orchestrator/status" in paths
    assert "/config/strategies" in paths
    assert "/backtests/" in paths or "/backtests" in paths
    assert "/intraday/status" in paths


def test_exported_openapi_file():
    """Ensure backend/openapi.json exists, is valid, and matches specification."""
    openapi_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    assert os.path.exists(openapi_path), "openapi.json file does not exist"

    with open(openapi_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    assert spec.get("info", {}).get("title") == "Smart Advice Trading API"
    assert len(spec.get("paths", {})) >= 25


def test_swagger_namespace_endpoints(client):
    """Verify primary GET endpoints across all 6 namespaces return 200."""
    endpoints = [
        "/api/health",
        "/api/analysis/health",
        "/api/positions",
        "/api/orchestrator/status",
        "/api/config/strategies",
        "/api/config/cycle-stats",
        "/api/backtests",
        "/api/intraday/status",
        "/api/intraday/config",
    ]
    for ep in endpoints:
        res = client.get(ep)
        assert res.status_code == 200, f"Endpoint {ep} failed with status {res.status_code}"


def test_legacy_frontend_routes_preserved(client):
    """Ensure existing root-level routes used by Next.js frontend continue working."""
    legacy_endpoints = [
        "/",
        "/recommendations",
        "/positions",
        "/strategies",
        "/cycle-stats",
        "/backtests",
        "/intraday/status",
    ]
    for ep in legacy_endpoints:
        res = client.get(ep)
        assert res.status_code == 200, f"Legacy route {ep} failed with status {res.status_code}"
