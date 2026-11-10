#!/usr/bin/env python3
"""Run serialized portfolio backtests for 1y through 20y and 2y intraday backtest."""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_portfolio_backtest import run_portfolio_backtest
from scripts.vectorbt_intraday_engine import (
    align_intraday_2d,
    compute_intraday_indicators_vbt,
    fetch_intraday_1h_data,
    generate_intraday_signals_vbt,
    run_intraday_simulation,
)
from utils.strategy_loader import StrategyLoader

PERIODS = [f"{i}y" for i in range(1, 21)]
STRATEGY = "Swing_Trading_v2"
OUTPUT_FILE = "serialized_portfolio_results_1_to_20y.json"

results_summary = []

print("============================================================")
print("STARTING SERIALIZED PORTFOLIO BACKTESTS: 1y to 20y")
print(f"Strategy: {STRATEGY}")
print("============================================================\n")

for p in PERIODS:
    print(f"\n>>> [RUNNING PORTFOLIO BACKTEST {p}] <<<")
    t0 = time.time()
    try:
        res = run_portfolio_backtest(
            strategy_name=STRATEGY, period=p, save_to_db=True, verbose=False, track_filters=False
        )
        elapsed = time.time() - t0

        data = {
            "period": p,
            "status": "SUCCESS",
            "duration_sec": round(elapsed, 1),
            "date_range": f"{res.get('date_range', {}).get('start', '')} → {res.get('date_range', {}).get('end', '')}",
            "initial_capital": res.get("initial_capital", 3000000.0),
            "final_value": res.get("final_value", 0.0),
            "total_return_pct": res.get("total_return_pct", 0.0),
            "cagr": res.get("cagr", 0.0),
            "max_drawdown_pct": res.get("max_drawdown_pct", 0.0),
            "sharpe_ratio": res.get("sharpe_ratio", 0.0),
            "profit_factor": res.get("profit_factor", 0.0),
            "total_trades": res.get("total_trades", 0),
            "win_rate": res.get("win_rate", 0.0),
            "expectancy": res.get("expectancy", 0.0),
            "avg_positions_held": res.get("avg_positions_held", 0.0),
            "session_id": str(res.get("session_id", "")),
        }
    except Exception as e:
        print(f"ERROR running {p}: {e}")
        data = {"period": p, "status": "FAILED", "error": str(e)}

    results_summary.append(data)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(
        f">>> [FINISHED {p} in {elapsed:.1f}s]: CAGR = {data.get('cagr', 0):.2f}%, Total Return = {data.get('total_return_pct', 0):.2f}% <<<\n"
    )

print("\n============================================================")
print("STARTING 2-YEAR INTRADAY TRADING BACKTEST (730d)")
print("============================================================\n")

try:
    strat = StrategyLoader.get_strategy_by_name("Intraday_Trading")
    data = fetch_intraday_1h_data(
        ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "ITC"],
        period="730d",
    )
    aligned = align_intraday_2d(data)
    indicators = compute_intraday_indicators_vbt(aligned, strat)
    entries, exits = generate_intraday_signals_vbt(indicators, strat)
    intraday_res = run_intraday_simulation(aligned, entries, exits, strat)

    with open("intraday_2y_results.json", "w") as f:

        def serialize(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        json.dump(intraday_res, f, indent=2, default=serialize)
    print(">>> [FINISHED 2-YEAR INTRADAY BACKTEST] <<<")
    print(f"Total Return: {intraday_res.get('total_return_pct')}%")
    print(f"CAGR: {intraday_res.get('cagr')}%")
    print(f"Win Rate: {intraday_res.get('win_rate')}%")
    print(f"Profit Factor: {intraday_res.get('profit_factor')}")
except Exception as e:
    print(f"ERROR running Intraday Backtest: {e}")

print("\n============================================================")
print("ALL RUNS COMPLETE.")
print("============================================================")
