#!/usr/bin/env python3
"""
Intraday Backtest CLI Runner
============================
Runs vectorbt-accelerated intraday portfolio backtests and ultimate validation pipelines.
Usage:
    python scripts/run_intraday_backtest.py
    python scripts/run_intraday_backtest.py --strategy Intraday_Trading_v1 --period 730d --ultimate
    python scripts/run_intraday_backtest.py --symbols RELIANCE,TCS,INFY --period 60d
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.vectorbt_intraday_engine import (
    align_intraday_2d,
    compute_intraday_indicators_vbt,
    fetch_intraday_1h_data,
    generate_intraday_signals_vbt,
    run_intraday_simulation,
    run_intraday_ultimate_backtest,
)
from utils.strategy_loader import StrategyLoader


def main():
    parser = argparse.ArgumentParser(description="Vectorbt Intraday Backtest CLI Runner")
    parser.add_argument(
        "--strategy",
        type=str,
        default="Intraday_Trading_v1",
        help="Intraday strategy name (e.g. Intraday_Trading_v1)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="730d",
        help="Historical period for 1h candles (e.g. 730d, 365d, 60d)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated list of symbols (e.g. RELIANCE,TCS,INFY)",
    )
    parser.add_argument(
        "--ultimate",
        action="store_true",
        default=True,
        help="Run full 6-phase Ultimate Validation pipeline with confidence score",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip saving backtest session to MongoDB",
    )

    args = parser.parse_args()

    symbols_list = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else None

    if args.ultimate:
        results = run_intraday_ultimate_backtest(
            strategy_name=args.strategy,
            period=args.period,
            symbols=symbols_list,
            save_to_db=not args.no_db,
        )
        print(f"\n[✓] Finished Intraday Ultimate Backtest in {results['elapsed_seconds']}s")
    else:
        strat = StrategyLoader.get_strategy_by_name(args.strategy)
        data = fetch_intraday_1h_data(
            symbols_list or ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"], period=args.period
        )
        aligned = align_intraday_2d(data)
        indicators = compute_intraday_indicators_vbt(aligned, strat)
        entries, exits = generate_intraday_signals_vbt(indicators, strat)
        res = run_intraday_simulation(aligned, entries, exits, strat)
        print("\n--- INTRADAY SIMULATION RESULTS ---")
        print(f"Total Return: {res['total_return_pct']}%")
        print(f"CAGR: {res['cagr']}%")
        print(f"Sharpe Ratio: {res['sharpe_ratio']}")
        print(f"Max Drawdown: {res['max_drawdown_pct']}%")
        print(f"Total Trades: {res['total_trades']}")
        print(f"Win Rate: {res['win_rate']}%")
        print(f"Profit Factor: {res['profit_factor']}")


if __name__ == "__main__":
    main()
