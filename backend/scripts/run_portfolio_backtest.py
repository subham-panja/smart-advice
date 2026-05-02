#!/usr/bin/env python3
"""
Portfolio Backtest Runner
=========================

CLI script to run a portfolio-level backtest across multiple stocks.

Usage:
    cd backend
    python scripts/run_portfolio_backtest.py --strategy Delayed_EP --max-stocks 50
    python scripts/run_portfolio_backtest.py --strategy Delayed_EP --symbols RELIANCE,INFY,TCS

The backtest uses a shared capital pool (default 10 Lakhs from config.py)
and compounds returns across all stocks simultaneously.
"""

import argparse
import logging
import multiprocessing
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.data_fetcher import get_historical_data
from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.stock_scanner import StockScanner
from utils.strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)


def _run_partial_backtest(strategy_config, symbols_chunk, initial_capital, max_positions):
    """Worker function for multiprocessing - runs backtest on a chunk of symbols."""
    from scripts.portfolio_backtest_engine import PortfolioBacktestSession

    chunk_capital = (
        initial_capital // len(multiprocessing.current_process().name.split("-"))
        if "-" in multiprocessing.current_process().name
        else initial_capital
    )

    engine = PortfolioBacktestSession(
        strategy_config=strategy_config,
        capital_config={
            "initial_capital": chunk_capital,
            "brokerage_charges": 0.002,
            "risk_per_trade": 2.0,
            "max_position_pct": 10.0,
            "max_concurrent_positions": max(2, max_positions // 4),
            "ranking_method": "combined_score",
            "save_daily_snapshots": False,
            "pyramid_counts_as_new_position": False,
            "same_day_cash_recycling": True,
            "force_close_delisted": True,
        },
    )

    results = engine.run(symbols_chunk)
    return results


def merge_backtest_results(results_list):
    """Merge multiple partial backtest results into a single summary."""
    if not results_list:
        return {}

    total_value = sum(r.get("final_portfolio_value", 0) for r in results_list)
    total_initial = sum(r.get("initial_capital", 0) for r in results_list)
    total_trades = sum(r.get("total_trades", 0) for r in results_list)
    total_return = ((total_value - total_initial) / total_initial * 100) if total_initial > 0 else 0

    # Average CAGR across chunks
    cagrs = [r.get("cagr", 0) for r in results_list if r.get("cagr")]
    avg_cagr = sum(cagrs) / len(cagrs) if cagrs else 0

    # Average win rate
    win_rates = [r.get("win_rate", 0) for r in results_list if r.get("win_rate")]
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0

    # Max drawdown (worst case)
    max_dd = min(r.get("max_drawdown_pct", 0) for r in results_list) if results_list else 0

    # Merge trades
    all_trades = []
    for r in results_list:
        all_trades.extend(r.get("trades", []))

    return {
        "strategy_name": results_list[0].get("strategy_name", "Unknown"),
        "status": "completed",
        "initial_capital": total_initial,
        "final_portfolio_value": total_value,
        "cash_remaining": sum(r.get("cash_remaining", 0) for r in results_list),
        "total_return_pct": total_return,
        "cagr": avg_cagr,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sum(r.get("sharpe_ratio", 0) for r in results_list) / len(results_list),
        "total_trades": total_trades,
        "win_rate": avg_win_rate,
        "profit_factor": sum(r.get("profit_factor", 0) for r in results_list) / len(results_list),
        "expectancy": sum(r.get("expectancy", 0) for r in results_list) / len(results_list),
        "avg_positions_held": sum(r.get("avg_positions_held", 0) for r in results_list) / len(results_list),
        "trades": all_trades,
        "daily_snapshots": [],
        "date_range": results_list[0].get("date_range", {}),
    }


def fetch_symbols_data(symbols: Dict[str, str], period: str = "5y", verbose: bool = False) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols in parallel."""
    data = {}
    total = len(symbols)
    logger.info(f"Fetching data for {total} symbols in parallel...")

    with ThreadPoolExecutor(max_workers=config.DATA_FETCH_THREADS) as executor:
        future_to_sym = {executor.submit(get_historical_data, sym, period=period): sym for sym in symbols.keys()}

        for i, future in enumerate(as_completed(future_to_sym)):
            sym = future_to_sym[future]
            try:
                df = future.result()
                if df is not None and not df.empty and len(df) > 100:
                    data[sym] = df
                    if verbose:
                        print(f"  [{i+1}/{total}] {sym}: {len(df)} bars")
                else:
                    logger.warning(f"Insufficient data for {sym}: {len(df) if df is not None else 0} bars")
            except Exception as e:
                logger.error(f"Failed to fetch {sym}: {e}")

    logger.info(f"Successfully fetched data for {len(data)}/{total} symbols")
    return data


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

    if config.PORTFOLIO_BACKTEST_CONFIG.get("use_multiprocessing", True) and len(symbols_data) >= 20:
        # Split symbols into chunks for parallel processing
        num_workers = min(config.NUM_WORKER_PROCESSES, len(symbols_data))
        symbols_list = list(symbols_data.items())
        chunk_size = len(symbols_list) // num_workers
        chunks = [dict(symbols_list[i : i + chunk_size]) for i in range(0, len(symbols_list), chunk_size)]

        logger.info(f"🚀 Running portfolio backtest with {num_workers} parallel workers...")
        logger.info(f"   Split {len(symbols_data)} symbols into {len(chunks)} chunks")

        with multiprocessing.Pool(processes=num_workers) as pool:
            partial_results = pool.starmap(
                _run_partial_backtest,
                [
                    (
                        strategy,
                        chunk,
                        config.INITIAL_CAPITAL,
                        config.PORTFOLIO_BACKTEST_CONFIG["max_concurrent_positions"],
                    )
                    for chunk in chunks
                ],
            )

        results = merge_backtest_results(partial_results)
    else:
        # Single-threaded backtest
        logger.info(f"🚀 Starting portfolio backtest for strategy: {strategy['name']}")
        persistence = PersistenceHandler()
        session_id = None
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


def main():
    parser = argparse.ArgumentParser(description="Run a portfolio-level backtest")
    parser.add_argument("--strategy", type=str, required=True, help="Strategy name (e.g., Delayed_EP)")
    parser.add_argument("--max-stocks", type=int, default=50, help="Max stocks to include")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (overrides scanner)")
    parser.add_argument("--period", type=str, default="5y", help="Historical data period")
    parser.add_argument("--no-db", action="store_true", help="Skip saving to database")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    symbol_list = args.symbols.split(",") if args.symbols else None

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
