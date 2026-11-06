#!/usr/bin/env python3
"""Run serialized portfolio backtests for periods 1y through 10y and print JSON / structured output."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_portfolio_backtest import run_portfolio_backtest

PERIODS = [f"{i}y" for i in range(1, 11)]
STRATEGY = "Swing_Trading_v2"
OUTPUT_FILE = "serialized_portfolio_results_1_to_10y.json"

results_summary = []

print("============================================================")
print("STARTING SERIALIZED PORTFOLIO BACKTESTS: 1y to 10y")
print(f"Strategy: {STRATEGY}")
print(f"Periods: {', '.join(PERIODS)}")
print("============================================================\n")

for p in PERIODS:
    print(f"\n>>> [RUNNING {p}] <<<")
    t0 = time.time()
    try:
        res = run_portfolio_backtest(
            strategy_name=STRATEGY, period=p, save_to_db=True, verbose=False, track_filters=False
        )
        elapsed = time.time() - t0

        # Extract core metrics
        dr = res.get("date_range", {})
        start_dt = dr.get("start_date", dr.get("start", ""))
        end_dt = dr.get("end_date", dr.get("end", ""))
        data = {
            "period": p,
            "status": "SUCCESS",
            "duration_sec": round(elapsed, 1),
            "date_range": f"{start_dt} → {end_dt}",
            "initial_capital": res.get("initial_capital", 3000000.0),
            "final_value": res.get("final_portfolio_value", res.get("final_value", 0.0)),
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
print("ALL 1y-10y RUNS FINISHED. Results written to:", OUTPUT_FILE)
print("============================================================")
