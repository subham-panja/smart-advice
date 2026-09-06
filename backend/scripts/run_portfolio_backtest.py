#!/usr/bin/env python3
"""
Portfolio Backtest Runner
=========================

CLI script to run a portfolio-level backtest across all NSE stocks.

Usage:
    cd backend
    python scripts/run_portfolio_backtest.py --strategy Swing_Trading
    python scripts/run_portfolio_backtest.py --strategy Swing_Trading --period 10y --track-filters
    python scripts/run_portfolio_backtest.py --strategy Swing_Trading --walk-forward --mc-iterations 10

The backtest uses a shared capital pool and parquet-cached historical data
and compounds returns across all stocks simultaneously.
Per-date stock filtering ensures only stocks that would have passed
the screener on each historical date are considered for entry.
"""

import argparse
import logging
import multiprocessing as mp
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from scripts.signal_precomputer import (
    _walk_forward_mc_worker_sequential,
    _walk_forward_mc_worker_with_signals,
    precompute_full_signals,
)
from utils.data_cache import fetch_multiple_symbols_cached
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filter Tracker — tracks gate-level pass/fail per symbol per day
# ---------------------------------------------------------------------------


class FilterTracker:
    """Tracks how many stocks pass/fail each filter stage."""

    def __init__(self):
        self.scanner_total = 0
        self.data_valid = 0
        self.data_rejected = 0
        self.gate_results = defaultdict(lambda: {"pass": set(), "fail": set()})
        self.entry_patterns_triggered = defaultdict(set)
        self.symbol_summary = defaultdict(lambda: {"buy_signals": 0, "days_scanned": 0})
        self.traded_symbols = set()
        self.trade_count = 0

    def record_scan(self, symbol: str, swing_result: dict):
        summary = self.symbol_summary[symbol]
        summary["days_scanned"] += 1

        gates = swing_result.get("gates", {})
        for gate_name, gate_ok in gates.items():
            if gate_ok:
                self.gate_results[gate_name]["pass"].add(symbol)
            else:
                self.gate_results[gate_name]["fail"].add(symbol)

        if swing_result.get("all_gates_passed"):
            reason = swing_result.get("reason", "")
            patterns = re.findall(r"Pattern\((\w+)\)", reason)
            for p in patterns:
                self.entry_patterns_triggered[p].add(symbol)

        if swing_result.get("all_gates_passed") and swing_result.get("recommendation") == "BUY":
            summary["buy_signals"] += 1

    def record_trade(self, symbol: str):
        self.traded_symbols.add(symbol)
        self.trade_count += 1

    def print_report(self):
        print("\n" + "=" * 70)
        print("STRATEGY FILTER TRACKING REPORT")
        print("=" * 70)

        print("\n  Scanner Stage:")
        print(f"    Stocks from scanner:       {self.scanner_total}")
        print(f"    With valid data:           {self.data_valid}")
        print(f"    Rejected (insufficient):   {self.data_rejected}")

        print("\n  Gate-Level Filtering (unique symbols that ever passed/failed):")
        for gate in ["trend", "volume", "volatility", "mtf"]:
            passed = self.gate_results[gate]["pass"]
            failed = self.gate_results[gate]["fail"]
            total = len(passed | failed)
            if total == 0:
                continue
            pass_pct = len(passed) / total * 100
            fail_pct = len(failed) / total * 100
            print(
                f"    {gate:20s}: PASS {len(passed):4d} ({pass_pct:5.1f}%)  FAIL {len(failed):4d} ({fail_pct:5.1f}%)  [{total} scanned]"
            )

        print("\n  Entry Patterns Triggered (unique symbols):")
        for pattern, symbols in sorted(self.entry_patterns_triggered.items(), key=lambda x: -len(x[1])):
            print(f"    {pattern:35s}: {len(symbols):4d} symbols")

        active_gates = [g for g in self.gate_results if len(self.gate_results[g]["pass"]) > 0]
        if active_gates:
            all_pass = set.intersection(*[self.gate_results[g]["pass"] for g in active_gates])
        else:
            all_pass = set()
        print(f"\n  All Gates Passed (intersection): {len(all_pass)} unique symbols")
        if all_pass:
            for s in sorted(all_pass)[:20]:
                print(f"    - {s}")
            if len(all_pass) > 20:
                print(f"    ... and {len(all_pass) - 20} more")

        print("\n  Top 20 Symbols by Buy Signal Count:")
        sorted_symbols = sorted(self.symbol_summary.items(), key=lambda x: -x[1]["buy_signals"])
        count = 0
        for sym, data in sorted_symbols:
            if data["buy_signals"] > 0:
                traded = "TRADED" if sym in self.traded_symbols else "not filled"
                print(
                    f"    {sym:15s}: {data['buy_signals']:4d} buy signals from {data['days_scanned']:4d} scans  [{traded}]"
                )
                count += 1
                if count >= 20:
                    break

        print("\n  Execution Stage:")
        print(f"    Unique symbols traded:     {len(self.traded_symbols)}")
        print(f"    Total trades executed:     {self.trade_count}")
        if self.traded_symbols:
            for s in sorted(self.traded_symbols):
                print(f"      - {s}")

        print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prepare_index_data(strategy: dict, symbols_data: dict, period: str) -> Optional[pd.DataFrame]:
    """Pre-fetch index data for regime detection and remove from symbols_data.

    Always fetches 10y of index history to ensure at least 250 index trading days
    are available for SMA(200) market regime detection from day 1.
    """
    regime_enabled = strategy.get("analysis_config", {}).get("market_regime_detection", False)
    if not regime_enabled:
        return None

    index_symbol = strategy.get("market_regime_config", {}).get("index", "^NSEI")
    pop_df = symbols_data.pop(index_symbol, None)

    try:
        index_data = fetch_multiple_symbols_cached({index_symbol: index_symbol}, period="10y", verbose=False)
        res_df = index_data.get(index_symbol)
        if res_df is not None and not res_df.empty:
            return res_df
    except Exception as e:
        logger.warning(f"Failed to pre-fetch index data for {index_symbol}: {e}")

    return pop_df


def _run_with_filter_tracking(
    engine: PortfolioBacktestSession,
    symbols_data: dict,
    strategy: dict,
    tracker: FilterTracker,
    index_data: Optional[pd.DataFrame],
    sim_start_date: Optional[pd.Timestamp] = None,
) -> dict:
    """Run the engine with filter tracking patches applied."""
    if index_data is not None:
        engine._index_data_override = index_data

    original_scan = engine._scan_for_signals

    def patched_scan(date, sd):
        candidates = original_scan(date, sd)
        for cand in candidates:
            tracker.record_scan(cand["symbol"], cand.get("swing_result", {}))
        return candidates

    engine._scan_for_signals = patched_scan
    return engine.run(symbols_data, sim_start_date=sim_start_date)


def fetch_symbols_data(symbols: Dict[str, str], period: str = "5y", verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols using incremental parquet cache."""
    return fetch_multiple_symbols_cached(symbols, period=period, verbose=verbose)


# ---------------------------------------------------------------------------
# Simple Portfolio Backtest
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Simple Portfolio Backtest
# ---------------------------------------------------------------------------


def run_portfolio_backtest(
    strategy_name: str,
    period: str = "5y",
    save_to_db: bool = True,
    verbose: bool = False,
    track_filters: bool = True,
):
    """Run a complete portfolio backtest session."""
    if verbose:
        setup_logging(verbose=True)

    logger.info("=" * 60)
    logger.info("PORTFOLIO BACKTEST RUNNER")
    logger.info("=" * 60)

    # 1. Load Strategy
    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    logger.info(f"Strategy: {strategy['name']}")

    # 2. Get Symbols — load ALL NSE symbols (realistic universe, no artificial cap)
    from scripts.data_fetcher import get_all_nse_symbols

    all_nse = get_all_nse_symbols()
    symbols = {s: s for s in all_nse} if isinstance(all_nse, list) else dict(all_nse)

    # Filter universe by market cap if specified in strategy config
    mc_filter = next((f for f in strategy.get("stock_filters", []) if f.get("type") == "market_cap"), None)
    if mc_filter and mc_filter.get("value"):
        from scripts.data_fetcher import get_market_caps

        min_mc = float(mc_filter.get("value", 0))
        mc_dict = get_market_caps(list(symbols.keys()))
        symbols = {s: s for s in symbols if mc_dict.get(s) and mc_dict.get(s) >= min_mc}
        logger.info(f"Filtered universe to {len(symbols)} quality symbols with Market Cap >= ₹{min_mc:.0f} Cr")
    else:
        logger.info(f"Using full NSE universe: {len(symbols)} symbols")

    if not symbols:
        raise RuntimeError("No symbols to backtest")

    # 3. Fetch Data with warmup buffer for technical indicators
    fetch_period = "10y" if period in ("5y", "2y", "1y", "6mo", "3mo") else "max"
    logger.info(f"Fetching historical data ({fetch_period} for indicator warmup) for {len(symbols)} symbols...")
    symbols_data = fetch_symbols_data(symbols, period=fetch_period, verbose=verbose)
    logger.info(f"Successfully loaded data for {len(symbols_data)} symbols")

    if len(symbols_data) < 5:
        raise RuntimeError(f"Too few symbols with valid data: {len(symbols_data)}")

    # 4. Prepare index data for regime detection
    index_data = _prepare_index_data(strategy, symbols_data, fetch_period)

    # 5. Pre-compute indicators using vectorbt
    start_time = datetime.now()
    session_id = None
    persistence = PersistenceHandler() if save_to_db else None

    logger.info(f"Pre-computing indicators for {len(symbols_data)} symbols using vectorbt...")
    try:
        from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

        indicators = compute_all_indicators(symbols_data, strategy)
        store = IndicatorStore(indicators)
        logger.info("  Indicators computed in one vectorized pass")
    except Exception as e:
        logger.warning(f"  Indicator pre-computation failed ({e}), falling back to TA-Lib")
        store = None
        indicators = None

    prefilter_matrix = None
    if indicators is not None:
        try:
            from scripts.vectorbt_signal_generator import compute_stock_prefilter

            prefilter_matrix = compute_stock_prefilter(indicators, strategy)
            n_pass = int(prefilter_matrix.iloc[-1].sum()) if len(prefilter_matrix) > 0 else 0
            logger.info(f"  Stock prefilter computed: {n_pass} stocks pass on latest date")
        except Exception as e:
            logger.warning(f"  Stock prefilter computation failed ({e}), skipping per-date filtering")

    if save_to_db and persistence:
        capital_cfg = config.PORTFOLIO_BACKTEST_CONFIG
        symbols_to_save = list(symbols_data.keys())
        session_id = persistence.create_backtest_session(
            strategy_name=strategy["name"],
            strategy_config=strategy,
            capital_config=capital_cfg,
            symbols=symbols_to_save,
        )
        logger.info(f"DB Session created: {session_id}")

    # 6. Run backtest
    engine = PortfolioBacktestSession(strategy_config=strategy)
    engine.session_id = session_id
    if store is not None:
        engine.set_indicator_store(store)
    if prefilter_matrix is not None:
        engine._stock_prefilter = prefilter_matrix

    index_symbol = strategy.get("market_regime_config", {}).get("index", "^NSEI")
    stock_only = {k: v for k, v in symbols_data.items() if k != index_symbol}
    all_sets = [set(df.index) for df in stock_only.values()]
    union_dates = sorted(set.union(*all_sets)) if all_sets else []

    # Determine simulation start date based on requested period while keeping prior data for warmup
    if union_dates and period.endswith("y"):
        years_back = int(period[:-1])
        target_start = union_dates[-1] - pd.DateOffset(years=years_back)
        sim_start = next((d for d in union_dates if d >= target_start), union_dates[0])
    elif union_dates and period.endswith("mo"):
        months_back = int(period[:-2])
        target_start = union_dates[-1] - pd.DateOffset(months=months_back)
        sim_start = next((d for d in union_dates if d >= target_start), union_dates[0])
    else:
        sim_start = None

    if track_filters:
        tracker = FilterTracker()
        tracker.scanner_total = len(symbols)
        tracker.data_valid = len(symbols_data)
        tracker.data_rejected = tracker.scanner_total - tracker.data_valid
        results = _run_with_filter_tracking(engine, symbols_data, strategy, tracker, index_data, sim_start)
    else:
        if index_data is not None:
            engine._index_data_override = index_data

        results = engine.run(symbols_data, sim_start_date=sim_start)

    duration = (datetime.now() - start_time).total_seconds()

    # 7. Save Results to DB
    if save_to_db and session_id:
        if results.get("daily_snapshots"):
            persistence.save_portfolio_backtest_snapshots(session_id, results["daily_snapshots"])
            logger.info(f"Saved {len(results['daily_snapshots'])} daily snapshots")

        if results.get("trades"):

            def _to_native(obj):
                if hasattr(obj, "item"):
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: _to_native(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_to_native(v) for v in obj]
                return obj

            trade_dicts = [_to_native(t.__dict__) for t in results["trades"]]
            persistence.save_portfolio_backtest_trades(session_id, trade_dicts)
            logger.info(f"Saved {len(trade_dicts)} trades")

        summary_metrics = {k: v for k, v in results.items() if k not in ("trades", "daily_snapshots")}
        persistence.complete_backtest_session(
            session_id=session_id,
            summary=summary_metrics,
            date_range=results.get("date_range", {}),
        )
        logger.info(f"Session {session_id} marked as completed")

    # 8. Record trades for tracker
    if track_filters:
        for trade in results.get("trades", []):
            tracker.record_trade(trade.symbol)

    # 9. Print Summary
    print("\n" + "=" * 60)
    print("PORTFOLIO BACKTEST RESULTS")
    print("=" * 60)
    print(f"Strategy:          {results['strategy_name']}")
    print(f"Date Range:        {results['date_range']['start_date']} → {results['date_range']['end_date']}")
    print(f"Duration:          {duration:.1f}s")
    print(f"Initial Capital:   ₹{results['initial_capital']:,.0f}")
    print(f"Final Value:       ₹{results['final_portfolio_value']:,.0f}")
    print(f"Total Return:      {results['total_return_pct']:+.2f}%")
    print(f"CAGR:              {results['cagr']:.2f}%")
    print(f"Max Drawdown:      {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
    print(f"Total Trades:      {results['total_trades']}")
    print(f"Win Rate:          {results['win_rate']:.1f}%")
    print(f"Profit Factor:     {results['profit_factor']:.2f}")
    print(f"Expectancy:        ₹{results['expectancy']:.2f}")
    print(f"Avg Positions:     {results['avg_positions_held']:.1f}")
    print("=" * 60)

    if session_id:
        print(f"\nSession saved to MongoDB: {session_id}")

    if track_filters:
        tracker.print_report()

    return results


# ---------------------------------------------------------------------------
# Walk-Forward + Monte Carlo
# ---------------------------------------------------------------------------


def run_walk_forward_backtest(
    strategy_name: str,
    period: str = "5y",
    mc_iterations: int = 10,
    verbose: bool = False,
    save_to_db: bool = True,
    symbols: dict = None,
    symbols_data: dict = None,
    indicators=None,
    prefilter=None,
    precomputed_signals=None,
) -> Dict:
    """Run walk-forward backtesting with Monte Carlo sampling."""

    logger.info("=" * 70)
    logger.info("WALK-FORWARD + MONTE CARLO PORTFOLIO BACKTEST")
    logger.info("=" * 70)

    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy {strategy_name} not found")

    if symbols_data is None:
        if symbols is None:
            from scripts.data_fetcher import get_all_nse_symbols

            all_nse = get_all_nse_symbols()
            symbols = {s: s for s in all_nse} if isinstance(all_nse, list) else dict(all_nse)
        symbols_list = list(symbols.keys())
        logger.info(f"Using full NSE universe: {len(symbols_list)} symbols")

        logger.info(f"Fetching {period} historical data for {len(symbols_list)} symbols...")
        symbols_data = fetch_symbols_data(symbols, period=period, verbose=verbose)
    else:
        logger.info(f"Using pre-fetched data for {len(symbols_data)} symbols")

    all_dates = set()
    for df in symbols_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    start_date = all_dates[0]
    end_date = all_dates[-1]
    total_days = (end_date - start_date).days
    logger.info(f"Full date range: {start_date.date()} → {end_date.date()} ({total_days} days)")

    window_days = 180
    step_days = 90
    windows = []
    current_start = start_date
    while current_start + pd.Timedelta(days=window_days) <= end_date:
        window_end = current_start + pd.Timedelta(days=window_days)
        windows.append((current_start, window_end))
        current_start += pd.Timedelta(days=step_days)

    logger.info(f"Walk-forward windows: {len(windows)}")
    for i, (ws, we) in enumerate(windows):
        logger.info(f"  Window {i+1}: {ws.date()} → {we.date()}")

    all_results = []
    start_time = datetime.now()
    persistence = PersistenceHandler() if save_to_db else None
    wf_session_id = None

    if save_to_db and persistence:
        capital_cfg = config.PORTFOLIO_BACKTEST_CONFIG
        wf_session_id = persistence.create_walk_forward_session(
            strategy_name=strategy_name,
            strategy_config=strategy,
            capital_config=capital_cfg,
            windows=windows,
            mc_iterations=mc_iterations,
        )
        logger.info(f"Walk-forward DB session created: {wf_session_id}")

    total_runs = len(windows) * mc_iterations
    completed_count = 0
    all_cagrs = []

    # Pre-compute indicators ONCE for the full dataset (shared across all windows)
    # Reuse from Phase 1 if passed in, otherwise compute fresh
    full_indicators = indicators
    full_prefilter = prefilter
    if full_indicators is None:
        try:
            from scripts.vectorbt_indicator_batch import compute_all_indicators

            logger.info(f"  Pre-computing indicators for {len(symbols_data)} symbols (full dataset)...")
            full_indicators = compute_all_indicators(symbols_data, strategy)
            logger.info("  Full indicators computed in one vectorized pass")

            from scripts.vectorbt_signal_generator import compute_stock_prefilter

            full_prefilter = compute_stock_prefilter(full_indicators, strategy)
            logger.info("  Stock prefilter computed for walk-forward")
        except Exception as e:
            logger.warning(f"  Full indicator pre-computation failed: {e}")
    else:
        logger.info("  Reusing pre-computed indicators from Phase 1")

    # Pre-compute ALL signals ONCE for the full dataset (Phase 1 optimization)
    # Reuse from Phase 1 if passed in
    full_signals = precomputed_signals if precomputed_signals else {}
    if not full_signals:
        try:
            full_signals = precompute_full_signals(symbols_data, strategy, full_indicators, full_prefilter)
        except Exception as e:
            logger.warning(f"  Signal pre-computation failed: {e}. Falling back to per-run signal generation.")
            full_signals = {}
    else:
        logger.info(f"  Reusing pre-computed signals from Phase 1 ({len(full_signals)} symbols)")

    max_workers = min(os.cpu_count() or 4, len(windows))
    logger.info(
        f"  Walk-forward: {len(windows)} windows x {mc_iterations} MC = {total_runs} total runs, {max_workers} parallel windows"
    )

    warmup_days = 200

    # Pre-slice window data ONCE per window in main process (avoids 1180x redundant slicing)
    window_tasks = []
    for window_idx, (window_start, window_end) in enumerate(windows):
        window_data = {}
        for sym, df in symbols_data.items():
            sliced = df[(df.index >= window_start - pd.Timedelta(days=warmup_days)) & (df.index <= window_end)]
            if len(sliced) >= 200:
                window_data[sym] = sliced

        if len(window_data) < 5:
            logger.warning(f"Too few symbols in window {window_idx+1} ({len(window_data)}), skipping")
            continue

        # Pre-compute MC samples for this window
        n = len(window_data)
        sample_size = min(max(int(n * 0.7), 5), n)
        mc_samples = []
        for mc_iter in range(mc_iterations):
            sampled_symbols = random.sample(list(window_data.keys()), sample_size)
            mc_samples.append((mc_iter + 1, sampled_symbols))

        window_tasks.append((window_idx, window_start, window_end, window_data, mc_samples))

    # Submit one task per (window, mc_iter) — pre-sliced data avoids redundant slicing
    use_precomputed = bool(full_signals)
    fork_ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=fork_ctx) as executor:
        futures = {}
        for window_idx, window_start, window_end, window_data, mc_samples in window_tasks:
            for mc_iter, sampled_symbols in mc_samples:
                sampled_data = {sym: window_data[sym] for sym in sampled_symbols if sym in window_data}
                if use_precomputed:
                    future = executor.submit(
                        _walk_forward_mc_worker_with_signals,
                        strategy,
                        sampled_data,
                        window_idx + 1,
                        mc_iter,
                        window_start,
                        window_end,
                        full_signals,
                    )
                else:
                    future = executor.submit(
                        _walk_forward_mc_worker_sequential,
                        strategy,
                        sampled_data,
                        window_idx + 1,
                        mc_iter,
                        window_start,
                        window_end,
                        full_indicators,
                        full_prefilter,
                    )
                futures[future] = (window_idx, mc_iter)

        for future in as_completed(futures):
            result = future.result()
            completed_count += 1
            if result and result.get("status") in ("success", "completed"):
                all_cagrs.append(result["cagr"])
                all_results.append(result)

            elapsed = (datetime.now() - start_time).total_seconds()
            pct = completed_count / total_runs * 100
            if completed_count % max(1, total_runs // 10) == 0 or completed_count >= total_runs:
                remaining = total_runs - completed_count
                eta = (elapsed / completed_count * remaining) if completed_count > 0 else 0
                eta_td = timedelta(seconds=int(eta))
                logger.info(f"Walk-forward progress: {pct:.0f}% ({completed_count}/{total_runs}) | " f"ETA: {eta_td}")

    # Batch DB writes at end
    if save_to_db and persistence and wf_session_id and all_results:
        for result in all_results:
            persistence.save_walk_forward_run(
                session_id=wf_session_id,
                window=result["window"],
                mc_iteration=result["mc_iteration"],
                symbols_count=result["symbols_count"],
                sampled_symbols=result.get("sampled_symbols", []),
                result=result,
            )

        elapsed = (datetime.now() - start_time).total_seconds()
        running_cagrs = [c for c in all_cagrs if c != 0]
        persistence.update_walk_forward_progress(
            session_id=wf_session_id,
            current_window=len(windows),
            completed_runs=completed_count,
            total_runs=total_runs,
            elapsed=elapsed,
            cagrs_so_far=running_cagrs,
        )

    if not all_results:
        return {"status": "failed", "reason": "No successful runs"}

    cagrs = [r.get("cagr", 0.0) for r in all_results]
    win_rates = [r.get("win_rate", 0.0) for r in all_results]
    sharpe_ratios = [r.get("sharpe", r.get("sharpe_ratio", 0.0)) for r in all_results]
    max_drawdowns = [r.get("max_drawdown", r.get("max_drawdown_pct", 0.0)) for r in all_results]
    profit_factors = [r.get("profit_factor", 0.0) for r in all_results]

    positive_cagr_pct = sum(1 for c in cagrs if c > 0) / len(cagrs) * 100

    mean_cagr = sum(cagrs) / len(cagrs)
    std_cagr = (sum((c - mean_cagr) ** 2 for c in cagrs) / len(cagrs)) ** 0.5
    cv = abs(std_cagr / mean_cagr) if mean_cagr != 0 else 999
    robustness_score = max(0, 100 - cv * 100)

    duration = (datetime.now() - start_time).total_seconds()

    aggregated = {
        "status": "completed",
        "total_runs": len(all_results),
        "windows_tested": len(windows),
        "mc_iterations_per_window": mc_iterations,
        "date_range": {
            "start": str(start_date.date()),
            "end": str(end_date.date()),
            "total_days": total_days,
        },
        "cagr": {
            "mean": round(mean_cagr, 2),
            "std": round(std_cagr, 2),
            "min": round(min(cagrs), 2),
            "max": round(max(cagrs), 2),
            "median": round(sorted(cagrs)[len(cagrs) // 2], 2),
        },
        "win_rate": {
            "mean": round(sum(win_rates) / len(win_rates), 2),
            "min": round(min(win_rates), 2),
            "max": round(max(win_rates), 2),
        },
        "sharpe": {
            "mean": round(sum(sharpe_ratios) / len(sharpe_ratios), 2),
            "min": round(min(sharpe_ratios), 2),
            "max": round(max(sharpe_ratios), 2),
        },
        "max_drawdown": {
            "mean": round(sum(max_drawdowns) / len(max_drawdowns), 2),
            "worst": round(min(max_drawdowns), 2),
        },
        "profit_factor": {
            "mean": round(sum(profit_factors) / len(profit_factors), 2),
        },
        "positive_cagr_pct": round(positive_cagr_pct, 1),
        "robustness_score": round(robustness_score, 1),
        "per_run_results": all_results,
        "_precomputed_signals": full_signals if full_signals else {},
        "_indicators": full_indicators,
        "_prefilter": full_prefilter,
    }

    if save_to_db and persistence and wf_session_id:
        persistence.complete_walk_forward_session(
            session_id=wf_session_id,
            aggregated_metrics=aggregated,
            duration=duration,
        )
        logger.info(f"Walk-forward session {wf_session_id} marked as completed")

    print("\n" + "=" * 70)
    print("WALK-FORWARD + MONTE CARLO SUMMARY")
    print("=" * 70)
    print(f"Strategy:          {strategy_name}")
    print(f"Period:            {start_date.date()} → {end_date.date()} ({total_days} days)")
    print(f"Windows:           {len(windows)} | MC per window: {mc_iterations}")
    print(f"Total runs:        {len(all_results)}")
    print()
    print("CAGR:")
    print(f"  Mean:   {mean_cagr:.1f}% ± {std_cagr:.1f}%")
    print(f"  Range:  {min(cagrs):.1f}% → {max(cagrs):.1f}%")
    print(f"  Median: {sorted(cagrs)[len(cagrs)//2]:.1f}%")
    print()
    print("Risk & Return:")
    print(f"  Avg Win Rate:  {aggregated['win_rate']['mean']:.1f}%")
    print(f"  Avg Sharpe:    {aggregated['sharpe']['mean']:.2f}")
    print(f"  Avg Max DD:    {aggregated['max_drawdown']['mean']:.1f}%")
    print(f"  Worst Max DD:  {aggregated['max_drawdown']['worst']:.1f}%")
    print(f"  Avg Profit F:  {aggregated['profit_factor']['mean']:.2f}")
    print()
    print("Robustness:")
    print(f"  Positive CAGR in: {positive_cagr_pct:.0f}% of runs")
    print(f"  Robustness Score: {robustness_score:.0f}/100")
    print(
        f"  {'STRATEGY IS ROBUST' if robustness_score > 60 and positive_cagr_pct > 70 else 'STRATEGY NEEDS IMPROVEMENT'}"
    )
    print("=" * 70)

    if wf_session_id:
        print(f"\nWalk-forward session saved to MongoDB: {wf_session_id}")
        print(f"   Duration: {duration:.1f}s")

    return aggregated


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Backtest CLI — run any strategy with optional walk-forward MC"
    )
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (from strategies/*.json)")
    parser.add_argument("--period", type=str, default="5y", help="Historical data period (e.g., 5y, 10y)")
    parser.add_argument("--no-db", action="store_true", help="Skip saving to database")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward + Monte Carlo backtest")
    parser.add_argument("--mc-iterations", type=int, default=10, help="Monte Carlo iterations per window")

    args = parser.parse_args()

    from utils.logger import setup_logging

    if args.verbose:
        config.VERBOSE = True
    setup_logging()

    if args.walk_forward:
        run_walk_forward_backtest(
            strategy_name=args.strategy,
            period=args.period,
            mc_iterations=args.mc_iterations,
            verbose=config.VERBOSE,
            save_to_db=not args.no_db,
        )
    else:
        run_portfolio_backtest(
            strategy_name=args.strategy,
            period=args.period,
            save_to_db=not args.no_db,
            verbose=config.VERBOSE,
        )


if __name__ == "__main__":
    main()
