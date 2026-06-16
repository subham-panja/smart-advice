#!/usr/bin/env python3
"""Run regular portfolio backtest with custom date ranges."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from scripts.run_portfolio_backtest import (
    FilterTracker,
    _prepare_index_data,
    _run_with_filter_tracking,
    fetch_symbols_data,
)
from utils.logger import setup_logging
from utils.stock_scanner import StockScanner
from utils.strategy_loader import StrategyLoader


def run_date_range(strategy_name, start_date, end_date, max_stocks=50):
    """Run backtest for a specific date range."""
    setup_logging(verbose=False)

    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    symbols = StockScanner().get_symbols(strategy_config=strategy)
    symbols_list = list(symbols.keys())[:max_stocks]

    # Fetch enough data to cover warmup + simulation
    period = "10y"
    symbols_data = fetch_symbols_data(symbols, period=period, verbose=False)

    # Filter to date range
    mask = {}
    for sym, df in symbols_data.items():
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if df.index.tz is not None:
            start_ts = start_ts.tz_localize(df.index.tz)
            end_ts = end_ts.tz_localize(df.index.tz)
        sliced = df[(df.index >= start_ts) & (df.index <= end_ts)]
        if len(sliced) >= 200:
            mask[sym] = sliced

    print(f"  {len(mask)} symbols with valid data for {start_date} → {end_date}")
    if len(mask) < 5:
        print("  Too few symbols, skipping")
        return None

    index_data = _prepare_index_data(strategy, mask, period)

    # Pre-compute indicators and per-day stock prefilter
    try:
        from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

        indicators = compute_all_indicators(mask, strategy)
        store = IndicatorStore(indicators)

        from scripts.vectorbt_signal_generator import compute_stock_prefilter

        prefilter = compute_stock_prefilter(indicators, strategy)
    except Exception as e:
        print(f"  WARNING: Indicator pre-computation failed ({e}), falling back to TA-Lib")
        store = None
        prefilter = None

    engine = PortfolioBacktestSession(strategy_config=strategy)
    if index_data is not None:
        engine._index_data_override = index_data
    if store is not None:
        engine.set_indicator_store(store)
    if prefilter is not None:
        engine._stock_prefilter = prefilter

    tracker = FilterTracker()
    tracker.scanner_total = len(symbols_list)
    tracker.data_valid = len(mask)
    tracker.data_rejected = tracker.scanner_total - tracker.data_valid

    results = _run_with_filter_tracking(engine, mask, strategy, tracker, index_data)
    return results


if __name__ == "__main__":
    STRATEGY_NAME = "Swing_Trading"  # Change to test other strategies

    tests = [
        ("2019-01-01", "2024-06-30", "COVID crash + recovery + choppy"),
        ("2021-05-01", "2026-05-12", "Bull run (baseline)"),
        ("2022-01-01", "2023-06-30", "Rate hike crash period"),
        ("2019-01-01", "2021-12-31", "Pre-bull + COVID crash"),
    ]

    all_results = []
    for start, end, label in tests:
        print(f"\n{'='*60}")
        print(f"Testing: {label}")
        print(f"Period: {start} → {end}")
        print(f"{'='*60}")
        result = run_date_range(STRATEGY_NAME, start, end)
        if result:
            all_results.append(
                {
                    "label": label,
                    "period": f"{start} → {end}",
                    "cagr": result["cagr"],
                    "total_return": result["total_return_pct"],
                    "max_dd": result["max_drawdown_pct"],
                    "sharpe": result["sharpe_ratio"],
                    "trades": result["total_trades"],
                    "win_rate": result["win_rate"],
                    "profit_factor": result["profit_factor"],
                }
            )

    print(f"\n\n{'='*80}")
    print("COMPARISON ACROSS MARKET REGIMES")
    print(f"{'='*80}")
    print(f"{'Label':<40} {'CAGR':>8} {'Ret%':>8} {'MaxDD':>8} {'Sharpe':>8} {'Trades':>8} {'Win%':>8} {'PF':>6}")
    print(f"{'-'*80}")
    for r in all_results:
        print(
            f"{r['label']:<40} {r['cagr']:>7.1f}% {r['total_return']:>7.1f}% {r['max_dd']:>7.1f}% {r['sharpe']:>8.2f} {r['trades']:>8d} {r['win_rate']:>7.1f}% {r['profit_factor']:>6.2f}"
        )
    print(f"{'='*80}")
