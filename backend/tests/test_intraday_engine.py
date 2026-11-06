"""
Unit tests for Intraday Engine & Scanner
========================================
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.intraday_engine import IntradayEngine, load_intraday_config
from scripts.intraday_scanner import IntradayScanner


@pytest.fixture
def mock_mongo():
    """Provides a memory mock for MongoDB intraday_positions collection."""
    storage = []

    class MockCollection:
        def insert_one(self, doc):
            doc["_id"] = str(len(storage) + 1)
            storage.append(doc)
            return MagicMock(inserted_id=doc["_id"])

        def find(self, query):
            results = []
            for d in storage:
                match = True
                for k, v in query.items():
                    if d.get(k) != v:
                        match = False
                        break
                if match:
                    results.append(d.copy())
            return results

        def find_one(self, query):
            res = self.find(query)
            return res[0] if res else None

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


def test_load_intraday_config():
    """Verify intraday strategy json loads and has required keys."""
    cfg = load_intraday_config()
    assert cfg.get("name") == "Intraday_Trading_v1"
    assert "timeframe_config" in cfg
    assert "intraday_gates" in cfg
    assert "exit_rules" in cfg
    assert cfg["exit_rules"]["time_exit"]["mandatory_square_off"] == "15:15"


def test_intraday_position_lifecycle(mock_mongo):
    """Test position creation, T1 hit with breakeven shift, and T2 close."""
    engine = IntradayEngine()

    pos = engine.create_position(
        symbol="TCS",
        entry_price=3500.0,
        quantity=10,
        stop_loss=3450.0,
        target_1=3550.0,
        target_2=3600.0,
        pattern="orb_15m_breakout",
    )

    assert pos["symbol"] == "TCS"
    assert pos["status"] == "OPEN"
    assert pos["quantity"] == 10

    # 1. Price moves up to Target 1 (3550): Partial exit 50%, stop loss moved to breakeven (3500)
    res_t1 = engine.process_price_tick("TCS", current_price=3550.0, current_time_str="10:30")
    assert res_t1 is not None
    assert res_t1["action"] == "TARGET_1_HIT"
    assert res_t1["new_sl"] == 3500.0  # Breakeven

    # Verify position state in memory
    open_pos = engine.get_open_positions()[0]
    assert open_pos["quantity"] == 5  # Half sold
    assert open_pos["t1_hit"] is True
    assert open_pos["stop_loss"] == 3500.0

    # 2. Price hits Target 2 (3600): Close remainder
    res_t2 = engine.process_price_tick("TCS", current_price=3600.0, current_time_str="11:15")
    assert res_t2 is not None
    assert res_t2["action"] == "TARGET_2_HIT"

    # Position should now be CLOSED
    assert len(engine.get_open_positions()) == 0


def test_mandatory_1515_square_off(mock_mongo):
    """Verify that any open intraday position is automatically closed at 15:15."""
    engine = IntradayEngine()

    engine.create_position(
        symbol="INFY",
        entry_price=1500.0,
        quantity=20,
        stop_loss=1470.0,
        target_1=1530.0,
        target_2=1560.0,
    )

    # Before 15:15 (e.g. 14:00) -> remains OPEN
    res_normal = engine.process_price_tick("INFY", current_price=1510.0, current_time_str="14:00")
    assert res_normal is None
    assert len(engine.get_open_positions()) == 1

    # At 15:15 -> AUTO_SQUARE_OFF
    res_square = engine.process_price_tick("INFY", current_price=1512.0, current_time_str="15:15")
    assert res_square is not None
    assert res_square["action"] == "AUTO_SQUARE_OFF"
    assert len(engine.get_open_positions()) == 0


def test_emergency_square_off_all(mock_mongo):
    """Test manual emergency square-off button."""
    engine = IntradayEngine()
    engine.create_position("SBIN", 800.0, 50, 785.0, 815.0, 830.0)
    engine.create_position("HDFCBANK", 1700.0, 30, 1680.0, 1720.0, 1740.0)

    assert len(engine.get_open_positions()) == 2

    closed = engine.square_off_all(reason="MANUAL_EMERGENCY")
    assert len(closed) == 2
    assert len(engine.get_open_positions()) == 0


def test_intraday_scanner():
    """Verify scanner evaluates candidate confluence and ranks signals."""
    scanner = IntradayScanner()
    dates = pd.date_range("2026-11-06 09:15", periods=6, freq="5min")

    mock_data = {
        "RELIANCE": {
            "df_5m": pd.DataFrame(
                {
                    "Open": [100.0, 102.0, 104.0, 106.0, 108.0, 112.0],
                    "High": [103.0, 105.0, 106.0, 108.0, 110.0, 115.0],
                    "Low": [99.0, 101.0, 103.0, 105.0, 107.0, 111.0],
                    "Close": [102.0, 104.0, 105.0, 107.0, 109.0, 114.0],
                    "Volume": [1000, 1200, 1100, 1500, 2000, 5000],
                },
                index=dates,
            ),
            "prev_day": {"High": 105.0, "Low": 95.0, "Close": 100.0},
        }
    }

    results = scanner.scan_universe(symbols_5m_data=mock_data, current_time_str="10:00")
    assert len(results) == 1
    assert results[0]["symbol"] == "RELIANCE"
    assert results[0]["signal"] == "BUY"
    assert results[0]["confluence_score"] == 1.0
