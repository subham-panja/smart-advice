import numpy as np
import pandas as pd

from scripts.swing_trading_signals import SwingTradingSignalAnalyzer
from utils.strategy_loader import StrategyLoader


def _create_base_df(num_bars=200, base_price=500.0):
    np.random.seed(42)
    dates = pd.date_range(start="2026-01-01", periods=num_bars, freq="D")

    # Mild upward trend with realistic oscillating price movements
    t = np.linspace(0, 4 * np.pi, num_bars)
    trend = np.linspace(0, 50, num_bars)
    noise = np.sin(t) * 10
    closes = base_price + trend + noise

    highs = closes + np.random.uniform(1.0, 3.0, num_bars)
    lows = closes - np.random.uniform(1.0, 3.0, num_bars)
    opens = closes - np.random.uniform(-1.0, 1.0, num_bars)
    volumes = np.random.uniform(200000, 350000, num_bars)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )
    return df


def test_swing_trading_v2_ema_pullback():
    analyzer = SwingTradingSignalAnalyzer()
    strategy = StrategyLoader.get_strategy_by_name("Swing_Trading_v2")
    assert strategy is not None

    df = _create_base_df()
    res = analyzer.analyze_swing_opportunity("TEST_STOCK", df, strategy_config=strategy)
    assert res is not None
    assert "all_gates_passed" in res


def test_minervini_vcp_breakout_trigger():
    analyzer = SwingTradingSignalAnalyzer()
    strategy = StrategyLoader.get_strategy_by_name("Swing_Trading_v2")

    df = _create_base_df(num_bars=200, base_price=500.0)

    # Setup consolidation over the last 20 days around 550
    df.iloc[-25:-1, df.columns.get_loc("High")] = 550.0
    df.iloc[-25:-1, df.columns.get_loc("Low")] = 540.0
    df.iloc[-25:-1, df.columns.get_loc("Close")] = 545.0
    # Dry-up volume prior to breakout
    df.iloc[-5:-1, df.columns.get_loc("Volume")] = 80000

    # Breakout bar on high volume
    df.iloc[-1, df.columns.get_loc("Close")] = 555.0
    df.iloc[-1, df.columns.get_loc("High")] = 556.0
    df.iloc[-1, df.columns.get_loc("Open")] = 548.0
    df.iloc[-1, df.columns.get_loc("Volume")] = 600000  # 2.5x volume

    res = analyzer.analyze_swing_opportunity("TEST_VCP", df, strategy_config=strategy)
    assert res is not None
    assert "gates" in res


def test_mandatory_pattern_enforcement():
    analyzer = SwingTradingSignalAnalyzer()
    strategy = StrategyLoader.get_strategy_by_name("Swing_Trading_v2")

    df = _create_base_df(num_bars=200, base_price=500.0)
    # Price is not in a breakout and not at 21 EMA
    df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-2] * 1.08
    df.iloc[-1, df.columns.get_loc("High")] = df["Close"].iloc[-1] + 1.0
    df.iloc[-1, df.columns.get_loc("Volume")] = 50000  # Low volume

    res = analyzer.analyze_swing_opportunity("TEST_NO_TRIGGER", df, strategy_config=strategy)
    assert res.get("recommendation", "HOLD") == "HOLD" or res["all_gates_passed"] is False
