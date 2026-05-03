#!/usr/bin/env python3
"""
Portfolio Backtest Runner
=========================

CLI script to run a portfolio-level backtest across multiple stocks.

Usage:
    cd backend
    python scripts/run_portfolio_backtest.py --strategy Delayed_EP --max-stocks 50
    python scripts/run_portfolio_backtest.py --strategy Delayed_EP --symbols RELIANCE,INFY,TCS

The backtest uses a shared capital pool and parquet-cached historical data
and compounds returns across all stocks simultaneously.
"""

import argparse
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from typing import Dict, List

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


def _compute_signals_worker(strategy_config, symbols_chunk):
    """Worker function for multiprocessing - pre-computes daily signals for a chunk of symbols."""
    from scripts.swing_trading_signals import SwingTradingSignalAnalyzer

    analyzer = SwingTradingSignalAnalyzer()
    all_signals = {}

    for symbol, df in symbols_chunk.items():
        symbol_signals = {}
        for date in df.index:
            hist = df.loc[:date]
            if len(hist) < 50:
                continue
            try:
                swing = analyzer.analyze_swing_opportunity(symbol, hist, strategy_config=strategy_config)
                if swing.get("all_gates_passed") and swing.get("recommendation") == "BUY":
                    symbol_signals[date] = {
                        "score": swing.get("technical_score", 0.0),
                        "swing_result": swing,
                    }
            except Exception:
                pass
        if symbol_signals:
            all_signals[symbol] = symbol_signals

    return all_signals


def merge_signal_results(results_list):
    """Merge multiple partial signal results into a single signal dictionary."""
    merged = {}
    for result in results_list:
        merged.update(result)
    return merged


def _walk_forward_mc_worker(args):
    """Worker function for walk-forward Monte Carlo iteration.

    Args:
        args: tuple of (strategy_config_dict, data_dir, symbol_list, window_idx, mc_iter, sim_start_date, sim_end_date)

    Returns:
        dict with metrics or error info (pickle-serializable)
    """
    import shutil

    strategy_config, data_dir, symbol_list, window_idx, mc_iter, sim_start_date, sim_end_date = args

    # Load data from parquet files
    sampled_data = {}
    for sym in symbol_list:
        fpath = os.path.join(data_dir, f"{sym}.parquet")
        if os.path.exists(fpath):
            sampled_data[sym] = pd.read_parquet(fpath)

    # Each worker creates its own engine (spawn-safe, no shared state)
    engine = PortfolioBacktestSession(strategy_config=strategy_config)

    try:
        result = engine.run(sampled_data, sim_start_date=sim_start_date, sim_end_date=sim_end_date)
        # Clean up temp data
        shutil.rmtree(data_dir, ignore_errors=True)
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
        shutil.rmtree(data_dir, ignore_errors=True)
        return {
            "window": window_idx,
            "mc_iteration": mc_iter,
            "symbols_count": len(sampled_data),
            "status": "failed",
            "error": str(e),
        }


def fetch_symbols_data(symbols: Dict[str, str], period: str = "5y", verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols using parquet cache."""
    # Get staleness from strategy config if available, default 24h
    staleness_hours = 24
    return fetch_multiple_symbols_cached(symbols, period=period, staleness_hours=staleness_hours, verbose=verbose)


def run_portfolio_backtest(
    strategy_name: str,
    max_stocks: int = 50,
    symbol_list: List[str] = None,
    period: str = "5y",
    save_to_db: bool = True,
    verbose: bool = False,
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

    # 4. Run Backtest (multiprocessing if enabled)
    start_time = datetime.now()
    session_id = None
    persistence = PersistenceHandler() if save_to_db else None

    if config.USE_MULTIPROCESSING_PIPELINE and len(symbols_data) >= 20:
        # Split symbols into chunks for parallel processing
        num_workers = min(config.NUM_WORKER_PROCESSES, len(symbols_data))
        symbols_list = list(symbols_data.items())
        chunk_size = len(symbols_list) // num_workers
        chunks = [dict(symbols_list[i : i + chunk_size]) for i in range(0, len(symbols_list), chunk_size)]

        logger.info(f"🚀 Running portfolio backtest with {num_workers} parallel workers...")
        logger.info(f"   Split {len(symbols_data)} symbols into {len(chunks)} chunks for signal generation")

        # Create session before multiprocessing
        if save_to_db and persistence:
            capital_cfg = config.PORTFOLIO_BACKTEST_CONFIG
            session_id = persistence.create_backtest_session(
                strategy_name=strategy["name"],
                strategy_config=strategy,
                capital_config=capital_cfg,
                symbols=list(symbols_data.keys()),
            )
            logger.info(f"DB Session created: {session_id}")

        # Phase 1: Parallel signal generation
        pool = None
        try:
            pool = multiprocessing.Pool(processes=num_workers)
            partial_signals = pool.starmap(
                _compute_signals_worker,
                [(strategy, chunk) for chunk in chunks],
            )
            pool.close()
            pool.join()
        except Exception as e:
            logger.error(f"Multiprocessing error during signal generation: {e}")
            if pool:
                pool.terminate()
                pool.join()
                logger.info(f"Terminated {num_workers} worker processes")
            raise e

        precomputed_signals = merge_signal_results(partial_signals)
        logger.info(f"   Pre-computed signals for {len(precomputed_signals)} symbols")

        # Phase 2: Single-threaded simulation with pre-computed signals
        engine = PortfolioBacktestSession(strategy_config=strategy)
        engine.session_id = session_id
        results = engine.run_with_signals(symbols_data, precomputed_signals)
    else:
        # Single-threaded backtest (signal generation + simulation)
        logger.info(f"🚀 Starting portfolio backtest for strategy: {strategy['name']}")
        if save_to_db:
            capital_cfg = config.PORTFOLIO_BACKTEST_CONFIG
            session_id = persistence.create_backtest_session(
                strategy_name=strategy["name"],
                strategy_config=strategy,
                capital_config=capital_cfg,
                symbols=list(symbols_data.keys()),
            )
            logger.info(f"DB Session created: {session_id}")

        engine = PortfolioBacktestSession(strategy_config=strategy)
        engine.session_id = session_id
        results = engine.run(symbols_data)
    duration = (datetime.now() - start_time).total_seconds()

    # 6. Save Results to DB
    if save_to_db and session_id:
        # Save daily snapshots
        if results.get("daily_snapshots"):
            persistence.save_portfolio_backtest_snapshots(session_id, results["daily_snapshots"])
            logger.info(f"Saved {len(results['daily_snapshots'])} daily snapshots")

        # Save trades (convert numpy types to native Python types for MongoDB)
        if results.get("trades"):

            def _to_native(obj):
                if hasattr(obj, "item"):  # numpy scalar
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: _to_native(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_to_native(v) for v in obj]
                return obj

            trade_dicts = [_to_native(t.__dict__) for t in results["trades"]]
            persistence.save_portfolio_backtest_trades(session_id, trade_dicts)
            logger.info(f"Saved {len(trade_dicts)} trades")

        # Complete session (pass metrics only, not raw trade objects)
        summary_metrics = {k: v for k, v in results.items() if k not in ("trades", "daily_snapshots")}
        persistence.complete_backtest_session(
            session_id=session_id,
            summary=summary_metrics,
            date_range=results.get("date_range", {}),
        )
        logger.info(f"Session {session_id} marked as completed")

    # 7. Print Summary
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
        print(f"\n💾 Session saved to MongoDB: {session_id}")

    return results


def run_walk_forward_backtest(
    strategy_name: str,
    period: str = "5y",
    max_stocks: int = 200,
    mc_iterations: int = 10,
    verbose: bool = False,
    save_to_db: bool = True,
) -> Dict:
    """Run walk-forward backtesting with Monte Carlo sampling to validate strategy robustness.

    Approach:
    1. Split the historical period into rolling windows (6-month test, roll forward 3 months)
    2. For each window: run backtest on stocks that existed at window start
    3. Monte Carlo: randomly subsample stocks to test universe independence
    4. Aggregate results: mean CAGR, std dev, min/max, consistency score

    Returns dict with aggregated metrics and per-run breakdown.
    """

    logger.info("=" * 70)
    logger.info("WALK-FORWARD + MONTE CARLO PORTFOLIO BACKTEST")
    logger.info("=" * 70)

    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy {strategy_name} not found")

    # Fetch all symbols from Chartink scanner
    scanner = StockScanner()
    symbols = scanner.get_symbols(strategy_config=strategy)
    symbols_list = list(symbols.keys())[:max_stocks]
    logger.info(f"Scanner returned {len(symbols_list)} symbols for universe")

    # Fetch full historical data for all symbols
    logger.info(f"Fetching {period} historical data for {len(symbols_list)} symbols...")
    symbols_data = fetch_symbols_data(symbols, period=period, verbose=verbose)

    # Determine date range
    all_dates = set()
    for df in symbols_data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)
    start_date = all_dates[0]
    end_date = all_dates[-1]
    total_days = (end_date - start_date).days
    logger.info(f"Full date range: {start_date.date()} → {end_date.date()} ({total_days} days)")

    # Define walk-forward windows (6-month test, 3-month step)
    window_days = 180  # ~6 months
    step_days = 90  # ~3 months
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

    # Create walk-forward session in DB
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

    # Run each window
    for window_idx, (window_start, window_end) in enumerate(windows):
        logger.info(f"\n{'='*50}")
        logger.info(f"WINDOW {window_idx+1}/{len(windows)}: {window_start.date()} → {window_end.date()}")
        logger.info(f"{'='*50}")

        # Slice data to window + 200-day warm-up (needed for SMA 200, ATR lookback, etc.)
        warmup_days = 200
        window_data = {}
        for sym, df in symbols_data.items():
            sliced = df[(df.index >= window_start - pd.Timedelta(days=warmup_days)) & (df.index <= window_end)]
            if len(sliced) >= 200:  # Min data required (warmup + partial window)
                window_data[sym] = sliced

        if len(window_data) < 20:
            logger.warning(f"Too few symbols in window {window_idx+1}, skipping")
            continue

        # Pre-sample stocks for each MC iteration and write to parquet files
        import random
        import tempfile
        import uuid

        sample_map = {}
        task_args = []
        for mc_iter in range(mc_iterations):
            sample_size = max(int(len(window_data) * 0.7), 20)
            sampled_symbols = random.sample(list(window_data.keys()), sample_size)
            sample_map[(window_idx + 1, mc_iter + 1)] = sampled_symbols

            # Write sampled data to temp parquet files
            data_dir = os.path.join(tempfile.gettempdir(), f"wf_data_{uuid.uuid4().hex[:8]}")
            os.makedirs(data_dir, exist_ok=True)
            for sym in sampled_symbols:
                window_data[sym].to_parquet(os.path.join(data_dir, f"{sym}.parquet"))

            task_args.append(
                (strategy, data_dir, sampled_symbols, window_idx + 1, mc_iter + 1, window_start, window_end)
            )

        # Run MC iterations in parallel
        num_workers = min(config.NUM_WORKER_PROCESSES, mc_iterations)
        ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(_walk_forward_mc_worker, task_args):
                completed_count += 1
                elapsed = (datetime.now() - start_time).total_seconds()

                # Track CAGR for running mean
                if result["status"] == "success":
                    all_cagrs.append(result["cagr"])
                    all_results.append(result)

                # Save to DB
                if save_to_db and persistence and wf_session_id:
                    syms = sample_map.get((result["window"], result["mc_iteration"]), [])
                    persistence.save_walk_forward_run(
                        session_id=wf_session_id,
                        window=result["window"],
                        mc_iteration=result["mc_iteration"],
                        symbols_count=result["symbols_count"],
                        sampled_symbols=syms,
                        result=result,
                    )

                    # Update progress
                    running_cagrs = [c for c in all_cagrs if c != 0]
                    persistence.update_walk_forward_progress(
                        session_id=wf_session_id,
                        current_window=result["window"],
                        completed_runs=completed_count,
                        total_runs=total_runs,
                        elapsed=elapsed,
                        cagrs_so_far=running_cagrs,
                    )

                # Log progress every 10% or at completion
                pct = completed_count / total_runs * 100
                if completed_count % max(1, total_runs // 10) == 0 or completed_count == total_runs:
                    remaining = total_runs - completed_count
                    eta = (elapsed / completed_count * remaining) if completed_count > 0 else 0
                    logger.info(
                        f"Walk-forward progress: {pct:.0f}% ({completed_count}/{total_runs}) | " f"ETA: {eta:.0f}s"
                    )

        if all_results:
            window_cagrs = [r["cagr"] for r in all_results if r.get("window") == window_idx + 1]
            if window_cagrs:
                avg_cagr = sum(window_cagrs) / len(window_cagrs)
                logger.info(f"  Window {window_idx+1} complete: {len(window_cagrs)} MC runs, avg CAGR {avg_cagr:.1f}%")

    # Aggregate results
    if not all_results:
        return {"status": "failed", "reason": "No successful runs"}

    cagrs = [r["cagr"] for r in all_results]
    win_rates = [r["win_rate"] for r in all_results]
    sharpe_ratios = [r["sharpe"] for r in all_results]
    max_drawdowns = [r["max_drawdown"] for r in all_results]
    profit_factors = [r["profit_factor"] for r in all_results]

    # Consistency score: % of runs with positive CAGR
    positive_cagr_pct = sum(1 for c in cagrs if c > 0) / len(cagrs) * 100

    # Robustness score: inverse of coefficient of variation (lower variance = more robust)
    mean_cagr = sum(cagrs) / len(cagrs)
    std_cagr = (sum((c - mean_cagr) ** 2 for c in cagrs) / len(cagrs)) ** 0.5
    cv = abs(std_cagr / mean_cagr) if mean_cagr != 0 else 999
    robustness_score = max(0, 100 - cv * 100)  # 100 = perfectly consistent, 0 = highly variable

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

    # Complete session
    if save_to_db and persistence and wf_session_id:
        persistence.complete_walk_forward_session(
            session_id=wf_session_id,
            aggregated_metrics=aggregated,
            duration=duration,
        )
        logger.info(f"Walk-forward session {wf_session_id} marked as completed")

    # Print summary
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
        f"  {'✅ STRATEGY IS ROBUST' if robustness_score > 60 and positive_cagr_pct > 70 else '⚠️ STRATEGY NEEDS IMPROVEMENT'}"
    )
    print("=" * 70)

    if wf_session_id:
        print(f"\n💾 Walk-forward session saved to MongoDB: {wf_session_id}")
        print(f"   Duration: {duration:.1f}s")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Run a portfolio-level backtest")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (e.g., Delayed_EP)")
    parser.add_argument("--max-stocks", type=int, default=50, help="Max stocks to include")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (overrides scanner)")
    parser.add_argument("--period", type=str, default="5y", help="Historical data period")
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
