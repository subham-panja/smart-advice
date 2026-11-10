#!/usr/bin/env python3
"""
Year-by-Year Portfolio Backtest Runner (Vectorized & Fast)
=========================================================
Runs portfolio backtests for 1y, 2y, ..., 10y windows using:
1. Pre-cached parquet files on disk
2. Market Cap pre-filtering (> ₹1,500 Cr)
3. VectorBT batch indicator pre-computation
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.data_fetcher import get_all_nse_symbols, get_market_caps
from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators
from utils.strategy_loader import StrategyLoader


def run_yearly_series(strategy_name: str = "Swing_Trading_v2", max_years: int = 10):
    print("=" * 80)
    print(f"YEAR-BY-YEAR BACKTEST: {strategy_name.upper()} (1y to {max_years}y)")
    print("=" * 80, flush=True)

    strategy = StrategyLoader.get_strategy_by_name(strategy_name)

    # 1. Quality Market Cap Filter
    all_nse = get_all_nse_symbols()
    all_symbols = {s: s for s in all_nse} if isinstance(all_nse, list) else dict(all_nse)

    mc_filter = next((f for f in strategy.get("stock_filters", []) if f.get("type") == "market_cap"), None)
    if mc_filter and mc_filter.get("value"):
        min_mc = float(mc_filter.get("value", 0))
        mc_dict = get_market_caps(list(all_symbols.keys()))
        all_symbols = {s: s for s in all_symbols if mc_dict.get(s) and mc_dict.get(s) >= min_mc}
        print(
            f"Filtered universe to {len(all_symbols)} quality symbols with Market Cap >= ₹{min_mc:.0f} Cr", flush=True
        )

    # 2. Load Index Data
    index_path = os.path.join(config.BACKEND_DIR, "data", "historical", "^NSEI.parquet")
    index_df = None
    if os.path.exists(index_path):
        index_df = pd.read_parquet(index_path)
        if index_df.index.tz is not None:
            index_df.index = index_df.index.tz_localize(None)

    # 3. Load Parquet Data for Quality Universe
    symbols_data = {}
    for sym in all_symbols.keys():
        fpath = os.path.join(config.BACKEND_DIR, "data", "historical", f"{sym}.parquet")
        if os.path.exists(fpath):
            try:
                df = pd.read_parquet(fpath)
                if not df.empty and len(df) > 50:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    symbols_data[sym] = df
            except Exception:
                pass

    print(f"Loaded {len(symbols_data)} symbols from local parquet cache.", flush=True)

    # 4. Precompute Indicators with VectorBT in One Fast Vectorized Pass
    print("Precomputing indicators in one batch across universe...", flush=True)
    t_vbt = datetime.now()
    indicators = compute_all_indicators(symbols_data, strategy)
    store = IndicatorStore(indicators)
    print(f"VectorBT precomputation complete in {(datetime.now() - t_vbt).total_seconds():.1f}s", flush=True)

    # 5. Determine Common Date Union
    all_sets = [set(df.index) for df in symbols_data.values()]
    union_dates = sorted(set.union(*all_sets)) if all_sets else []
    latest_date = union_dates[-1]
    earliest_date = union_dates[0]
    total_avail_years = (latest_date - earliest_date).days / 365.25

    print(
        f"Date Range: {earliest_date.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')} (~{total_avail_years:.1f}y)",
        flush=True,
    )
    print("-" * 80, flush=True)

    results_table = []
    years_to_test = [y for y in range(1, max_years + 1) if y <= total_avail_years + 0.5]

    for y in years_to_test:
        target_start = latest_date - pd.DateOffset(years=y)
        sim_start = next((d for d in union_dates if d >= target_start), union_dates[0])

        engine = PortfolioBacktestSession(strategy_config=strategy)
        if index_df is not None:
            engine._index_data_override = index_df
        engine.set_indicator_store(store)

        t0 = datetime.now()
        res = engine.run(symbols_data, sim_start_date=sim_start)
        duration = (datetime.now() - t0).total_seconds()

        init_cap = res.get("initial_capital", config.PORTFOLIO_BACKTEST_CONFIG.get("initial_capital", 10000.0))
        final_val = res.get("final_portfolio_value", res.get("final_value", init_cap))
        tot_return = res.get("total_return_pct", ((final_val - init_cap) / init_cap) * 100.0)
        cagr = res.get("cagr", 0.0)
        max_dd = res.get("max_drawdown_pct", 0.0)
        sharpe = res.get("sharpe_ratio", 0.0)
        win_rate = res.get("win_rate", 0.0)
        total_trades = res.get("total_trades", 0)

        row = {
            "Years": f"{y}y",
            "Period": f"{sim_start.strftime('%Y-%m-%d')} → {latest_date.strftime('%Y-%m-%d')}",
            "Initial": f"₹{init_cap:,.0f}",
            "Final Value": f"₹{final_val:,.0f}",
            "Total Return": f"{tot_return:+.1f}%",
            "CAGR": f"{cagr:.2f}%",
            "Max Drawdown": f"{max_dd:.2f}%",
            "Sharpe": f"{sharpe:.2f}",
            "Win Rate": f"{win_rate:.1f}%",
            "Trades": total_trades,
            "Time": f"{duration:.1f}s",
        }
        results_table.append(row)
        print(
            f"[{y:2d}y] {row['Period']} | Final: {row['Final Value']:>11} | Return: {row['Total Return']:>8} | CAGR: {row['CAGR']:>7} | MaxDD: {row['Max Drawdown']:>7} | Sharpe: {row['Sharpe']:>5} | WinRate: {row['Win Rate']:>5} | Trades: {row['Trades']:>4} ({duration:.1f}s)",
            flush=True,
        )

    print("\n" + "=" * 95)
    print("CONSOLIDATED YEAR-BY-YEAR PERFORMANCE SUMMARY")
    print("=" * 95)
    df_res = pd.DataFrame(results_table)
    try:
        print(df_res.to_markdown(index=False))
    except Exception:
        print(df_res.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="Swing_Trading_v2")
    parser.add_argument("--max-years", type=int, default=10)
    args = parser.parse_args()
    run_yearly_series(args.strategy, args.max_years)
