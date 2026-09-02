#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Backtest Runner
Runs 5-year portfolio backtests across all enabled strategies and generates a comprehensive performance comparison.
"""

import os
import sys

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_portfolio_backtest import run_portfolio_backtest
from utils.strategy_loader import StrategyLoader


def main():
    strategies = StrategyLoader.load_all_strategies()
    print("=" * 80)
    print(f"MULTI-STRATEGY 5-YEAR PORTFOLIO BACKTEST ({len(strategies)} Strategies Enabled)")
    print("=" * 80)

    results = []

    for idx, strat in enumerate(strategies, 1):
        strat_name = strat["name"]
        print(f"\n[{idx}/{len(strategies)}] Running 5-Year Backtest for: {strat_name}...")
        try:
            res = run_portfolio_backtest(
                strategy_name=strat_name,
                period="5y",
                save_to_db=True,
                verbose=False,
                track_filters=True,
            )
            results.append(
                {
                    "strategy": strat_name,
                    "status": "SUCCESS",
                    "final_value": res.get("final_portfolio_value", 0.0),
                    "total_return_pct": res.get("total_return_pct", 0.0),
                    "cagr": res.get("cagr", 0.0),
                    "max_drawdown": res.get("max_drawdown_pct", 0.0),
                    "win_rate": res.get("win_rate", 0.0),
                    "profit_factor": res.get("profit_factor", 0.0),
                    "total_trades": res.get("total_trades", 0),
                    "expectancy": res.get("expectancy", 0.0),
                }
            )
        except Exception as e:
            print(f"❌ Error backtesting {strat_name}: {e}")
            results.append(
                {
                    "strategy": strat_name,
                    "status": f"FAILED: {e}",
                    "final_value": 0.0,
                    "total_return_pct": 0.0,
                    "cagr": 0.0,
                    "max_drawdown": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "total_trades": 0,
                    "expectancy": 0.0,
                }
            )

    # Summary Table
    print("\n" + "=" * 105)
    print("ALL STRATEGIES 5-YEAR BACKTEST COMPARISON SUMMARY")
    print("=" * 105)
    print(
        f"{'Strategy':<32} | {'Final Val (₹)':<13} | {'Return %':<10} | {'CAGR (Annual %)':<16} | {'Win %':<7} | {'PF':<6} | {'Max DD %':<9} | {'Trades':<6}"
    )
    print("-" * 105)

    for r in results:
        strat = r["strategy"]
        if r["status"] == "SUCCESS":
            fv = f"₹{r['final_value']:,.0f}"
            ret = f"{r['total_return_pct']:+.2f}%"
            cagr = f"{r['cagr']:.2f}%/yr"
            win = f"{r['win_rate']:.1f}%"
            pf = f"{r['profit_factor']:.2f}"
            dd = f"{r['max_drawdown']:.2f}%"
            trades = str(r["total_trades"])
            print(f"{strat:<32} | {fv:<13} | {ret:<10} | {cagr:<16} | {win:<7} | {pf:<6} | {dd:<9} | {trades:<6}")
        else:
            print(f"{strat:<32} | {r['status']}")

    print("=" * 105)


if __name__ == "__main__":
    main()
