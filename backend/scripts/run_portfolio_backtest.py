#!/usr/bin/env python3
"""
Portfolio Backtest Runner
=========================

CLI script to run a portfolio-level backtest across multiple stocks.

Usage:
    cd backend
    python scripts/run_portfolio_backtest.py --strategy Swing_Trading --max-stocks 50
    python scripts/run_portfolio_backtest.py --strategy Swing_Trading --symbols RELIANCE,INFY,TCS
    python scripts/run_portfolio_backtest.py --strategy Swing_Trading --walk-forward --mc-iterations 10
    python scripts/run_portfolio_backtest.py --strategy Swing_Trading --period 10y --track-filters

The backtest uses a shared capital pool and parquet-cached historical data
and compounds returns across all stocks simultaneously.
"""

import argparse
import logging
import os
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from utils.data_cache import fetch_multiple_symbols_cached
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.stock_scanner import StockScanner
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

    Returns the index DataFrame, and removes it from symbols_data so it
    doesn't inflate the common-date union.
    """
    regime_enabled = strategy.get("analysis_config", {}).get("market_regime_detection", False)
    if not regime_enabled:
        return None

    index_symbol = strategy.get("market_regime_config", {}).get("index", "^NSEI")
    if index_symbol in symbols_data:
        # Pop it from symbols_data (caller will set engine._index_data_override)
        return symbols_data.pop(index_symbol, None)

    # Fetch it separately
    try:
        index_data = fetch_multiple_symbols_cached({index_symbol: index_symbol}, period=period, verbose=False)
        return index_data.get(index_symbol)
    except Exception as e:
        logger.warning(f"Failed to pre-fetch index data for {index_symbol}: {e}")
        return None


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
        market_breadth_ok = engine._check_market_breadth(date, sd)
        for symbol, df in sd.items():
            if symbol in engine.positions:
                continue
            if date not in df.index:
                continue
            hist = df.loc[:date]
            if len(hist) < 50:
                continue
            try:
                swing = engine.swing_analyzer.analyze_swing_opportunity(
                    symbol,
                    hist,
                    strategy_config=strategy,
                    indicator_store=engine._indicator_store,
                    market_breadth_ok=market_breadth_ok,
                )
                tracker.record_scan(symbol, swing)
            except Exception:
                pass
        return candidates

    engine._scan_for_signals = patched_scan
    return engine.run(symbols_data, sim_start_date=sim_start_date)


def fetch_symbols_data(symbols: Dict[str, str], period: str = "5y", verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols using incremental parquet cache."""
    return fetch_multiple_symbols_cached(symbols, period=period, verbose=verbose)


# ---------------------------------------------------------------------------
# Pre-compute signals for a window (avoids redundant scanning in each MC worker)
# ---------------------------------------------------------------------------


def _precompute_window_signals(window_data, indicator_store, strategy_config):
    """Pre-compute BUY/HOLD signals for all (symbol, date) pairs in a window.

    Uses the same analyze_swing_opportunity as the live/slow path so results
    are identical, but computes them once instead of once per MC iteration.

    Returns:
        Dict[symbol, Dict[date, {"score": float, "swing_result": dict}]]
    """
    from scripts.swing_trading_signals import SwingTradingSignalAnalyzer

    analyzer = SwingTradingSignalAnalyzer()
    signals = {}

    for symbol, df in window_data.items():
        symbol_signals = {}
        for date in df.index:
            hist = df.loc[:date]
            if len(hist) < 50:
                continue
            try:
                swing = analyzer.analyze_swing_opportunity(
                    symbol,
                    hist,
                    strategy_config=strategy_config,
                    indicator_store=indicator_store,
                )
                if swing.get("all_gates_passed") and swing.get("recommendation") == "BUY":
                    score = swing.get("technical_score", 0.0)
                    # Normalize date to tz-naive for dict key compatibility
                    date_key = date.tz_localize(None) if date.tzinfo else date
                    symbol_signals[date_key] = {
                        "score": score,
                        "swing_result": swing,
                    }
            except Exception:
                pass
        if symbol_signals:
            signals[symbol] = symbol_signals

    return signals


# ---------------------------------------------------------------------------
# Walk-Forward MC Worker (multiprocessing)
# ---------------------------------------------------------------------------


def _walk_forward_mc_worker(args):
    """Worker function for walk-forward Monte Carlo iteration.

    Each worker builds its own IndicatorStore and computes signals on-the-fly.
    This avoids pickling massive pre-computed signal dicts across processes.
    """
    strategy_config, sampled_data, window_idx, mc_iter, sim_start_date, sim_end_date = args[:6]
    indicator_store_data = args[6] if len(args) > 6 else None

    engine = PortfolioBacktestSession(strategy_config=strategy_config)

    if indicator_store_data is not None:
        try:
            from scripts.vectorbt_indicator_batch import IndicatorStore

            store = IndicatorStore(indicator_store_data)
            engine.set_indicator_store(store)
        except Exception:
            pass

    try:
        result = engine.run(sampled_data, sim_start_date=sim_start_date, sim_end_date=sim_end_date)
        return {
            "window": window_idx,
            "mc_iteration": mc_iter,
            "symbols_count": len(sampled_data),
            "status": "success",
            "cagr": float(result["cagr"]),
            "total_return": float(result["total_return_pct"]),
            "max_drawdown": float(result["max_drawdown_pct"]),
            "sharpe": float(result["sharpe_ratio"]),
            "total_trades": int(result["total_trades"]),
            "win_rate": float(result["win_rate"]),
            "profit_factor": float(result["profit_factor"]),
        }
    except Exception as e:
        return {
            "window": window_idx,
            "mc_iteration": mc_iter,
            "symbols_count": len(sampled_data),
            "status": "failed",
            "error": str(e),
        }


def _walk_forward_mc_worker_sequential(
    strategy_config,
    sampled_data,
    window_idx,
    mc_iter,
    sim_start_date,
    sim_end_date,
    indicator_store=None,
):
    """Sequential worker using IndicatorStore for O(1) indicator lookups.

    No signal pre-computation needed — the IndicatorStore makes per-day
    signal scans fast enough to run sequentially.
    """
    from scripts.portfolio_backtest_engine import PortfolioBacktestSession

    engine = PortfolioBacktestSession(strategy_config=strategy_config)

    if indicator_store is not None:
        engine.set_indicator_store(indicator_store)

    try:
        result = engine.run(sampled_data, sim_start_date=sim_start_date, sim_end_date=sim_end_date)
        return {
            "window": window_idx,
            "mc_iteration": mc_iter,
            "symbols_count": len(sampled_data),
            "status": "success",
            "cagr": float(result["cagr"]),
            "total_return": float(result["total_return_pct"]),
            "max_drawdown": float(result["max_drawdown_pct"]),
            "sharpe": float(result["sharpe_ratio"]),
            "total_trades": int(result["total_trades"]),
            "win_rate": float(result["win_rate"]),
            "profit_factor": float(result["profit_factor"]),
        }
    except Exception as e:
        return {
            "window": window_idx,
            "mc_iteration": mc_iter,
            "symbols_count": len(sampled_data),
            "status": "failed",
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Simple Portfolio Backtest
# ---------------------------------------------------------------------------


def run_portfolio_backtest(
    strategy_name: str,
    max_stocks: int = 50,
    symbol_list: List[str] = None,
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

    # 2. Get Symbols
    if symbol_list:
        symbols = {s: s for s in symbol_list}
        logger.info(f"Using {len(symbols)} provided symbols")
    else:
        symbols = StockScanner.get_symbols(strategy_config=strategy)
        symbols = dict(list(symbols.items())[:max_stocks])
        logger.info(f"Scanner returned {len(symbols)} symbols (limited to {max_stocks})")

    if not symbols:
        raise RuntimeError("No symbols to backtest")

    # 3. Fetch Data
    logger.info(f"Fetching {period} historical data for {len(symbols)} symbols...")
    symbols_data = fetch_symbols_data(symbols, period=period, verbose=verbose)
    logger.info(f"Successfully loaded data for {len(symbols_data)} symbols")

    if len(symbols_data) < 5:
        raise RuntimeError(f"Too few symbols with valid data: {len(symbols_data)}")

    # 4. Prepare index data for regime detection
    index_data = _prepare_index_data(strategy, symbols_data, period)

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

    if track_filters:
        tracker = FilterTracker()
        tracker.scanner_total = len(symbols) if not symbol_list else len(symbol_list)
        tracker.data_valid = len(symbols_data)
        tracker.data_rejected = tracker.scanner_total - tracker.data_valid
        sim_start = None
        if index_data is not None:
            index_symbol = strategy.get("market_regime_config", {}).get("index", "^NSEI")
            stock_only = {k: v for k, v in symbols_data.items() if k != index_symbol}
            all_sets = [set(df.index) for df in stock_only.values()]
            union_dates = sorted(set.union(*all_sets))
            for d in union_dates:
                if len(index_data.loc[:d]) >= 250:
                    sim_start = d
                    break
        results = _run_with_filter_tracking(engine, symbols_data, strategy, tracker, index_data, sim_start)
    else:
        sim_start = None
        if index_data is not None:
            engine._index_data_override = index_data
            # Skip early dates where index has < 250 rows (regime detection needs warmup)
            stock_only = {
                k: v
                for k, v in symbols_data.items()
                if k != (strategy.get("market_regime_config", {}).get("index", "^NSEI"))
            }
            all_sets = [set(df.index) for df in stock_only.values()]
            union_dates = sorted(set.union(*all_sets))
            # Find the first date where index_data has 250+ rows
            for d in union_dates:
                if len(index_data.loc[:d]) >= 250:
                    sim_start = d
                    break

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
    max_stocks: int = 200,
    mc_iterations: int = 10,
    verbose: bool = False,
    save_to_db: bool = True,
    symbols: dict = None,
) -> Dict:
    """Run walk-forward backtesting with Monte Carlo sampling."""

    logger.info("=" * 70)
    logger.info("WALK-FORWARD + MONTE CARLO PORTFOLIO BACKTEST")
    logger.info("=" * 70)

    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy {strategy_name} not found")

    if symbols is None:
        scanner = StockScanner()
        symbols = scanner.get_symbols(strategy_config=strategy)
    symbols_list = list(symbols.keys())[:max_stocks]
    logger.info(f"Using {len(symbols_list)} symbols for universe")

    logger.info(f"Fetching {period} historical data for {len(symbols_list)} symbols...")
    symbols_data = fetch_symbols_data(symbols, period=period, verbose=verbose)

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
    full_indicator_store = None
    try:
        from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

        logger.info(f"  Pre-computing indicators for {len(symbols_data)} symbols (full dataset)...")
        full_indicators = compute_all_indicators(symbols_data, strategy)
        full_indicator_store = IndicatorStore(full_indicators)
        logger.info("  Full indicators computed in one vectorized pass")
    except Exception as e:
        logger.warning(f"  Full indicator pre-computation failed: {e}")

    for window_idx, (window_start, window_end) in enumerate(windows):
        logger.info(f"\n{'='*50}")
        logger.info(f"WINDOW {window_idx+1}/{len(windows)}: {window_start.date()} → {window_end.date()}")
        logger.info(f"{'='*50}")

        warmup_days = 200
        window_data = {}
        for sym, df in symbols_data.items():
            sliced = df[(df.index >= window_start - pd.Timedelta(days=warmup_days)) & (df.index <= window_end)]
            if len(sliced) >= 200:
                window_data[sym] = sliced

        if len(window_data) < 5:
            logger.warning(f"Too few symbols in window {window_idx+1} ({len(window_data)}), skipping")
            continue

        # Run MC iterations sequentially using IndicatorStore for O(1) lookups
        # (no signal pre-computation needed — IndicatorStore makes per-day scans fast)
        window_results = []
        for mc_iter in range(mc_iterations):
            sample_size = max(int(len(window_data) * 0.7), 20)
            sampled_symbols = random.sample(list(window_data.keys()), sample_size)
            sampled_data = {sym: window_data[sym] for sym in sampled_symbols}

            result = _walk_forward_mc_worker_sequential(
                strategy,
                sampled_data,
                window_idx + 1,
                mc_iter + 1,
                window_start,
                window_end,
                full_indicator_store,
            )

            completed_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()

            if result["status"] == "success":
                all_cagrs.append(result["cagr"])
                all_results.append(result)
                window_results.append(result)

            pct = completed_count / total_runs * 100
            if completed_count % max(1, total_runs // 10) == 0 or completed_count == total_runs:
                remaining = total_runs - completed_count
                eta = (elapsed / completed_count * remaining) if completed_count > 0 else 0
                logger.info(f"Walk-forward progress: {pct:.0f}% ({completed_count}/{total_runs}) | " f"ETA: {eta:.0f}s")

        # Batch DB writes per window (instead of per-iteration)
        if save_to_db and persistence and wf_session_id and window_results:
            for mc_iter_idx, result in enumerate(window_results):
                syms = random.sample(list(window_data.keys()), max(int(len(window_data) * 0.7), 20))
                persistence.save_walk_forward_run(
                    session_id=wf_session_id,
                    window=result["window"],
                    mc_iteration=result["mc_iteration"],
                    symbols_count=result["symbols_count"],
                    sampled_symbols=syms,
                    result=result,
                )

            running_cagrs = [c for c in all_cagrs if c != 0]
            persistence.update_walk_forward_progress(
                session_id=wf_session_id,
                current_window=window_idx + 1,
                completed_runs=completed_count,
                total_runs=total_runs,
                elapsed=elapsed,
                cagrs_so_far=running_cagrs,
            )

        if window_results:
            window_cagrs = [r["cagr"] for r in window_results]
            avg_cagr = sum(window_cagrs) / len(window_cagrs)
            logger.info(f"  Window {window_idx+1} complete: {len(window_cagrs)} MC runs, avg CAGR {avg_cagr:.1f}%")

    if not all_results:
        return {"status": "failed", "reason": "No successful runs"}

    cagrs = [r["cagr"] for r in all_results]
    win_rates = [r["win_rate"] for r in all_results]
    sharpe_ratios = [r["sharpe"] for r in all_results]
    max_drawdowns = [r["max_drawdown"] for r in all_results]
    profit_factors = [r["profit_factor"] for r in all_results]

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
    parser.add_argument("--max-stocks", type=int, default=50, help="Max stocks to include")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (overrides scanner)")
    parser.add_argument("--period", type=str, default="5y", help="Historical data period (e.g., 5y, 10y)")
    parser.add_argument("--no-db", action="store_true", help="Skip saving to database")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward + Monte Carlo backtest")
    parser.add_argument("--mc-iterations", type=int, default=10, help="Monte Carlo iterations per window")

    args = parser.parse_args()

    symbol_list = args.symbols.split(",") if args.symbols else None

    if args.walk_forward:
        run_walk_forward_backtest(
            strategy_name=args.strategy,
            period=args.period,
            max_stocks=args.max_stocks,
            mc_iterations=args.mc_iterations,
            verbose=args.verbose,
            save_to_db=not args.no_db,
        )
    else:
        run_portfolio_backtest(
            strategy_name=args.strategy,
            max_stocks=args.max_stocks,
            symbol_list=symbol_list,
            period=args.period,
            save_to_db=not args.no_db,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()
