"""Tests for all API routes after blueprint refactor."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def get_client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


def test_health():
    client = get_client()
    resp = client.get("/")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "ok"
    assert "timestamp" in data
    print("  PASS: GET /")


def test_recommendations():
    client = get_client()
    resp = client.get("/recommendations")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    assert "count" in data
    assert "recommendations" in data
    print("  PASS: GET /recommendations")


def test_positions_get():
    client = get_client()
    resp = client.get("/positions")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    assert "positions" in data
    print("  PASS: GET /positions")


def test_positions_get_open():
    client = get_client()
    resp = client.get("/positions?status=OPEN")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    print("  PASS: GET /positions?status=OPEN")


def test_positions_create_missing_fields():
    client = get_client()
    resp = client.post("/positions", json={"symbol": "TEST"})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "error" in data
    print("  PASS: POST /positions (missing fields)")


def test_positions_create_no_body():
    client = get_client()
    resp = client.post("/positions", data="", content_type="application/json")
    assert resp.status_code in (400, 415)
    print("  PASS: POST /positions (no body)")


def test_positions_update_no_body():
    client = get_client()
    resp = client.patch("/positions/TEST", data="", content_type="application/json")
    assert resp.status_code in (400, 415)
    print("  PASS: PATCH /positions/TEST (no body)")


def test_positions_update_symbol_change():
    client = get_client()
    resp = client.patch("/positions/TEST", json={"symbol": "OTHER"})
    assert resp.status_code == 400
    data = json.loads(resp.data)
    assert "Cannot change symbol" in data["error"]
    print("  PASS: PATCH /positions/TEST (symbol change blocked)")


def test_positions_update_not_found():
    client = get_client()
    resp = client.patch("/positions/NONEXISTENT", json={"entry_price": 100})
    assert resp.status_code == 404
    print("  PASS: PATCH /positions/NONEXISTENT (not found)")


def test_positions_close_not_found():
    client = get_client()
    resp = client.delete("/positions/NONEXISTENT")
    assert resp.status_code == 404
    print("  PASS: DELETE /positions/NONEXISTENT (not found)")


def test_strategies():
    client = get_client()
    resp = client.get("/strategies")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    assert data["count"] > 0
    assert len(data["strategies"]) > 0
    strat = data["strategies"][0]
    assert "name" in strat
    assert "enabled" in strat
    assert "file_name" in strat
    print(f"  PASS: GET /strategies ({data['count']} strategies)")


def test_trading_config():
    client = get_client()
    resp = client.get("/settings/trading")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    cfg = data["config"]
    assert "is_paper_trading" in cfg
    assert "initial_capital" in cfg
    print("  PASS: GET /settings/trading")


def test_cycle_stats():
    client = get_client()
    resp = client.get("/cycle-stats")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["status"] == "success"
    assert "open_positions" in data
    assert "total_equity" in data
    print("  PASS: GET /cycle-stats")


def test_orchestrator_status():
    client = get_client()
    resp = client.get("/orchestrator-status")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert "status" in data
    print("  PASS: GET /orchestrator-status")


def test_analyze_stock_invalid():
    client = get_client()
    resp = client.get("/analyze_stock/INVALID_SYMBOL_12345")
    assert resp.status_code == 500
    data = json.loads(resp.data)
    assert data["status"] == "error"
    print("  PASS: GET /analyze_stock/INVALID (error handled)")


if __name__ == "__main__":
    print("=== Backend API Route Tests ===\n")
    tests = [
        test_health,
        test_recommendations,
        test_positions_get,
        test_positions_get_open,
        test_positions_create_missing_fields,
        test_positions_create_no_body,
        test_positions_update_no_body,
        test_positions_update_symbol_change,
        test_positions_update_not_found,
        test_positions_close_not_found,
        test_strategies,
        test_trading_config,
        test_cycle_stats,
        test_orchestrator_status,
        test_analyze_stock_invalid,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\n=== Results: {passed} passed, {failed} failed ===")
    sys.exit(0 if failed == 0 else 1)
