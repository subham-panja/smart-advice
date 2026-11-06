"""
Unit tests for Intraday Indicators & Confluence Logic
=====================================================
"""

import pandas as pd
import pytest

from scripts.intraday_indicators import (
    calculate_cpr,
    calculate_orb,
    calculate_vwap,
    evaluate_intraday_confluence,
)


def test_calculate_cpr():
    """Verify Central Pivot Range (CPR) formulas and bounds."""
    # Example: High=1050, Low=950, Close=1000
    prev_high = 1050.0
    prev_low = 950.0
    prev_close = 1000.0

    cpr = calculate_cpr(prev_high, prev_low, prev_close)

    # Pivot = (1050 + 950 + 1000) / 3 = 1000.0
    assert cpr["pivot"] == 1000.0

    # BC = (1050 + 950) / 2 = 1000.0
    # TC = (1000 - 1000) + 1000 = 1000.0
    assert cpr["tc"] >= cpr["bc"]
    assert cpr["cpr_width_pct"] >= 0.0

    # Non-symmetric case
    cpr2 = calculate_cpr(1100.0, 900.0, 1050.0)
    assert cpr2["pivot"] == pytest.approx(1016.67, 0.01)
    assert cpr2["tc"] >= cpr2["bc"]
    assert cpr2["r1"] > cpr2["pivot"]
    assert cpr2["s1"] < cpr2["pivot"]


def test_calculate_cpr_invalid_input():
    with pytest.raises(ValueError):
        calculate_cpr(-100, 100, 100)


def test_calculate_vwap():
    """Verify VWAP calculation across typical bars."""
    dates = pd.date_range("2026-11-06 09:15", periods=5, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0, 103.0, 105.0],
            "High": [103.0, 105.0, 106.0, 105.0, 107.0],
            "Low": [99.0, 101.0, 103.0, 102.0, 104.0],
            "Close": [102.0, 104.0, 103.0, 105.0, 106.0],
            "Volume": [1000, 2000, 1500, 1200, 1800],
        },
        index=dates,
    )

    vwap = calculate_vwap(df)
    assert len(vwap) == 5
    assert not vwap.isna().any()
    # Typical price of bar 0: (103 + 99 + 102) / 3 = 101.33
    assert vwap.iloc[0] == pytest.approx(101.33, 0.05)


def test_calculate_orb():
    """Verify Opening Range Breakout boundaries."""
    dates = pd.date_range("2026-11-06 09:15", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "High": [100.0, 105.0, 103.0, 108.0, 109.0, 110.0],
            "Low": [95.0, 98.0, 97.0, 102.0, 104.0, 106.0],
            "Close": [98.0, 102.0, 101.0, 107.0, 108.0, 109.0],
        },
        index=dates,
    )

    orb = calculate_orb(df, start_time_str="09:15", end_time_str="09:30")
    # First 3 bars (09:15, 09:20, 09:25, 09:30)
    assert orb["orb_high"] >= 105.0
    assert orb["orb_low"] <= 98.0
    assert orb["orb_range"] > 0


def test_evaluate_intraday_confluence_buy_and_hold():
    """Verify buy signal generation when CPR, VWAP, and ORB align."""
    dates = pd.date_range("2026-11-06 09:15", periods=6, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100.0, 102.0, 104.0, 106.0, 108.0, 112.0],
            "High": [103.0, 105.0, 106.0, 108.0, 110.0, 115.0],
            "Low": [99.0, 101.0, 103.0, 105.0, 107.0, 111.0],
            "Close": [102.0, 104.0, 105.0, 107.0, 109.0, 114.0],
            "Volume": [1000, 1200, 1100, 1500, 2000, 5000],  # Volume surge on breakout
        },
        index=dates,
    )

    prev_day = {"High": 105.0, "Low": 95.0, "Close": 100.0}

    # During market hours (10:15) with breakout above CPR (100) and ORB (106)
    result = evaluate_intraday_confluence(
        symbol="RELIANCE",
        df_5m=df,
        prev_day_ohlc=prev_day,
        current_time_str="10:15",
    )

    assert result["symbol"] == "RELIANCE"
    assert result["all_gates_passed"] is True
    assert result["signal"] == "BUY"
    assert result["stop_loss"] < result["current_price"]
    assert result["target_1"] > result["current_price"]
    assert result["rr_ratio"] > 1.0

    # Test outside market hours: should HOLD
    outside_result = evaluate_intraday_confluence(
        symbol="RELIANCE",
        df_5m=df,
        prev_day_ohlc=prev_day,
        current_time_str="09:20",
    )
    assert outside_result["signal"] == "HOLD"
