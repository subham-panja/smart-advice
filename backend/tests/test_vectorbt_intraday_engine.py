"""
Test suite for Vectorbt Intraday Backtesting Engine.
Tests data alignment, indicator computation, signal generation,
and ultimate backtest integration.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.vectorbt_intraday_engine import (
    align_intraday_2d,
    compute_intraday_indicators_vbt,
    generate_intraday_signals_vbt,
)
from utils.strategy_loader import StrategyLoader


@pytest.fixture
def sample_intraday_data():
    """Create a synthetic 1-hour OHLCV DataFrame for testing."""
    dates = pd.date_range("2026-01-05 09:15:00", periods=28, freq="1h")
    symbols = ["INFY", "RELIANCE"]

    symbols_data = {}
    for sym in symbols:
        np.random.seed(42 if sym == "INFY" else 43)
        base = 1500.0 if sym == "INFY" else 2500.0
        close = base + np.cumsum(np.random.randn(28) * 5)
        high = close + np.random.uniform(2, 6, size=28)
        low = close - np.random.uniform(2, 6, size=28)
        open_p = close + np.random.randn(28) * 2
        volume = np.random.randint(100000, 500000, size=28)

        symbols_data[sym] = pd.DataFrame(
            {
                "Open": open_p,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )

    return symbols_data


def test_align_intraday_2d(sample_intraday_data):
    """Verify 2D multi-symbol alignment."""
    aligned = align_intraday_2d(sample_intraday_data)
    assert "Close" in aligned
    assert "High" in aligned
    assert "Low" in aligned
    assert "Volume" in aligned
    assert aligned["Close"].shape == (28, 2)
    assert list(aligned["Close"].columns) == ["INFY", "RELIANCE"]


def test_compute_intraday_indicators_vbt(sample_intraday_data):
    """Verify vectorized VWAP, CPR, ORB, ATR, and Volume indicators."""
    aligned = align_intraday_2d(sample_intraday_data)
    strat = StrategyLoader.get_strategy_by_name("Intraday_Trading_v1")
    indicators = compute_intraday_indicators_vbt(aligned, strat)

    assert "vwap" in indicators
    assert "cpr_pivot" in indicators
    assert "cpr_tc" in indicators
    assert "orb_high" in indicators
    assert "atr" in indicators
    assert "rsi" in indicators

    # Verify no multiindex on columns
    assert not isinstance(indicators["atr"].columns, pd.MultiIndex)
    assert not isinstance(indicators["vol_sma"].columns, pd.MultiIndex)
    assert not isinstance(indicators["rsi"].columns, pd.MultiIndex)


def test_generate_intraday_signals_vbt(sample_intraday_data):
    """Verify signal generation and EOD mandatory square-off exit matrix."""
    aligned = align_intraday_2d(sample_intraday_data)
    strat = StrategyLoader.get_strategy_by_name("Intraday_Trading_v1")
    indicators = compute_intraday_indicators_vbt(aligned, strat)
    entries, exits = generate_intraday_signals_vbt(indicators, strat)

    assert entries.shape == aligned["Close"].shape
    assert exits.shape == aligned["Close"].shape
    assert isinstance(entries, pd.DataFrame)
    assert isinstance(exits, pd.DataFrame)

    # Exits must have True flags for mandatory square-off
    assert exits.values.any()
