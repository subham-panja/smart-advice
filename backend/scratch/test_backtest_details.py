import os
import sys

import pandas as pd

# Add backend to path
sys.path.append(os.path.abspath("backend"))

from scripts.backtesting import BacktestingEngine
from scripts.strategies.base_strategy import BacktraderStrategy


def test_single_stock():
    symbol = "DMART.NS"
    print(f"--- Testing Backtest Details for {symbol} ---")

    # Load sample data
    import yfinance as yf

    df = yf.download(symbol, start="2020-01-01", end="2024-01-01")

    # Flatten columns if multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        print("Error: No data fetched")
        return

    # Load real strategy config
    import json

    with open("backend/strategies/delayed_ep.json", "r") as f:
        config = json.load(f)

    # Override for testing to force more trades
    config["swing_trading_gates"]["TREND_GATE"]["enabled"] = False
    config["swing_trading_gates"]["VOLATILITY_GATE"]["enabled"] = False
    config["swing_trading_gates"]["VOLUME_GATE"]["enabled"] = False
    config["entry_patterns"] = []  # Allow all

    class SimpleStrategy(BacktraderStrategy):
        pass

        def notify_order(self, order):
            if order.status in [order.Completed]:
                self.last_executed_size = abs(order.executed.size)

        def notify_trade(self, trade):
            if trade.isclosed:
                # The analyzer will pick up the size from self.last_executed_size
                pass

    engine = BacktestingEngine(initial_cash=100000, commission=0.0005)
    res = engine.run_backtest(SimpleStrategy, df, params={"symbol": symbol, "strat_params": config})

    print(f"Total Trades: {len(res['trades'])}")
    print(f"ROI: {res['roi']:.2f}%")

    if res["trades"]:
        print("\n--- SAMPLE TRADE DATA ---")
        for i, t in enumerate(res["trades"][:5]):
            print(f"Trade {i+1}: {t['entry_date']} -> {t['exit_date']}")
            print(f"  Entry: {t['entry_price']} | Exit: {t['exit_price']}")
            print(f"  PnL: {t['pnl']} | PnL%: {t['pnl_pct']}")
            print(f"  Profitable: {t['is_profitable']}")
    else:
        print("WARNING: No trades captured in tradelist!")


if __name__ == "__main__":
    test_single_stock()
