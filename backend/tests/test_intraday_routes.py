"""
Integration tests for Intraday API Routes
=========================================
"""

from unittest.mock import MagicMock, patch

import pytest

from app import create_app
from scripts.intraday_engine import IntradayEngine


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_mongo():
    storage = []

    class MockCursor(list):
        def sort(self, field, dir=1):
            return self

    class MockCollection:
        def insert_one(self, doc):
            doc["_id"] = str(len(storage) + 1)
            storage.append(doc)
            return MagicMock(inserted_id=doc["_id"])

        def find(self, query=None):
            q = query or {}
            results = []
            for d in storage:
                match = True
                for k, v in q.items():
                    if d.get(k) != v:
                        match = False
                        break
                if match:
                    results.append(d.copy())
            return MockCursor(results)

        def find_one(self, query):
            q = query or {}
            for d in storage:
                match = True
                for k, v in q.items():
                    if d.get(k) != v:
                        match = False
                        break
                if match:
                    return d.copy()
            return None

        def update_one(self, query, update):
            for d in storage:
                match = True
                for k, v in query.items():
                    if d.get(k) != v:
                        match = False
                        break
                if match:
                    if "$set" in update:
                        d.update(update["$set"])
                    return MagicMock(modified_count=1)
            return MagicMock(modified_count=0)

    mock_col = MockCollection()
    with patch.object(IntradayEngine, "_get_collection", return_value=mock_col):
        yield mock_col


def test_get_intraday_status(client, mock_mongo):
    """Verify /intraday/status returns clock, phase, and day stats."""
    res = client.get("/intraday/status")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "clock" in data
    assert "current_time_ist" in data["clock"]
    assert "stats" in data
    assert "realized_pnl" in data["stats"]


def test_get_intraday_config(client):
    """Verify /intraday/config returns strategy json."""
    res = client.get("/intraday/config")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert data["config"]["name"] == "Intraday_Trading_v1"


def test_intraday_positions_crud(client, mock_mongo):
    """Verify creating and listing intraday positions."""
    # 1. Create Position
    payload = {
        "symbol": "TCS",
        "entry_price": 3500.0,
        "quantity": 10,
        "stop_loss": 3450.0,
        "target_1": 3550.0,
        "target_2": 3600.0,
        "pattern": "orb_15m_breakout",
        "note": "Test intraday note",
    }
    post_res = client.post("/intraday/positions", json=payload)
    assert post_res.status_code == 201
    post_data = post_res.get_json()
    assert post_data["status"] == "success"
    assert post_data["position"]["symbol"] == "TCS"

    # 2. Get Positions
    get_res = client.get("/intraday/positions")
    assert get_res.status_code == 200
    get_data = get_res.get_json()
    assert get_data["count"] == 1
    assert get_data["positions"][0]["symbol"] == "TCS"


def test_square_off_all_endpoint(client, mock_mongo):
    """Verify /intraday/square-off-all squares off positions."""
    # Seed 1 position
    client.post("/intraday/positions", json={"symbol": "INFY", "entry_price": 1500.0, "quantity": 5})

    sq_res = client.post("/intraday/square-off-all")
    assert sq_res.status_code == 200
    sq_data = sq_res.get_json()
    assert sq_data["status"] == "success"
    assert sq_data["closed_count"] == 1


def test_close_single_position_endpoint(client, mock_mongo):
    """Verify /intraday/positions/<id>/close closes a single position."""
    post_res = client.post("/intraday/positions", json={"symbol": "WIPRO", "entry_price": 400.0, "quantity": 10})
    pos_id = post_res.get_json()["position"]["_id"]

    close_res = client.post(f"/intraday/positions/{pos_id}/close", json={"reason": "MANUAL_TEST_CLOSE"})
    assert close_res.status_code == 200
    close_data = close_res.get_json()
    assert close_data["status"] == "success"
    assert close_data["position"]["status"] == "CLOSED"
    assert close_data["position"]["exit_reason"] == "MANUAL_TEST_CLOSE"


def test_get_intraday_signals(client):
    """Verify /intraday/signals returns candidates list."""
    res = client.get("/intraday/signals")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "signals" in data
