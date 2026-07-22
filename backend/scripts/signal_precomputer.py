"""
Signal Pre-computer & Walk-Forward Workers for Portfolio Backtesting
=====================================================================

Pre-computes BUY/SELL signals in parallel across symbols and dates, and provides
worker functions for Walk-Forward Monte Carlo simulations.
"""

import logging
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from scripts.portfolio_backtest_engine import PortfolioBacktestSession

logger = logging.getLogger(__name__)


def _precompute_window_signals(window_data, indicator_store, strategy_config):
    """Pre-compute BUY/HOLD signals for all (symbol, date) pairs in a window."""
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


def _compute_signals_batch(args):
    """Worker: compute signals for a batch of symbols."""
    symbol_list, symbols_data, strategy_config, indicator_store_data = args
    from scripts.swing_trading_signals import SwingTradingSignalAnalyzer
    from scripts.vectorbt_indicator_batch import IndicatorStore

    analyzer = SwingTradingSignalAnalyzer()
    if hasattr(indicator_store_data, "_series_cache"):
        store = indicator_store_data
    elif indicator_store_data is not None:
        store = IndicatorStore(indicator_store_data)
    else:
        store = None
    signals = {}

    for symbol in symbol_list:
        if symbol not in symbols_data:
            continue
        df = symbols_data[symbol]
        n = len(df)
        if n < 100:
            continue
        symbol_signals = {}
        for i in range(50, n):
            date = df.index[i]
            hist = df.iloc[: i + 1]
            try:
                swing = analyzer.analyze_swing_opportunity(
                    symbol,
                    hist,
                    strategy_config=strategy_config,
                    indicator_store=store,
                )
                if swing.get("all_gates_passed") and swing.get("recommendation") == "BUY":
                    score = swing.get("technical_score", 0.0)
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


def precompute_full_signals(symbols_data, strategy_config, indicator_store_data, prefilter, num_workers=None):
    """Pre-compute BUY signals for ALL eligible symbols across ALL dates (parallel)."""
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, 8)

    eligible_symbols = set()
    if prefilter is not None:
        for sym in prefilter.columns:
            if prefilter[sym].any():
                eligible_symbols.add(sym)
    else:
        eligible_symbols = set(symbols_data.keys())

    eligible_symbols = [s for s in eligible_symbols if s in symbols_data]
    logger.info(f"  Pre-computing signals for {len(eligible_symbols)} eligible symbols across {num_workers} workers...")

    t0 = time.time()

    batches = np.array_split(eligible_symbols, num_workers)
    batch_args = [(list(b), symbols_data, strategy_config, indicator_store_data) for b in batches if len(b) > 0]

    all_signals = {}
    fork_ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=fork_ctx) as executor:
        futures = [executor.submit(_compute_signals_batch, args) for args in batch_args]
        for future in as_completed(futures):
            batch_result = future.result()
            all_signals.update(batch_result)

    elapsed = time.time() - t0
    total_dates = sum(len(v) for v in all_signals.values())
    logger.info(
        f"  Signal pre-computation done: {len(all_signals)} symbols, {total_dates} buy signals "
        f"in {elapsed:.0f}s ({total_dates/max(elapsed,1):.0f} signals/sec)"
    )
    return all_signals


def _walk_forward_mc_worker_with_signals(
    strategy_config,
    sampled_data,
    window_idx,
    mc_iter,
    sim_start_date,
    sim_end_date,
    precomputed_signals,
):
    """Worker: run a single Walk-Forward simulation using precomputed signals."""
    try:
        session = PortfolioBacktestSession(strategy_config=strategy_config)

        if precomputed_signals and len(precomputed_signals) > 0:
            window_signals = {}
            for sym in sampled_data.keys():
                if sym in precomputed_signals:
                    window_signals[sym] = precomputed_signals[sym]

            res = session.run_with_signals(
                symbols_data=sampled_data,
                precomputed_signals=window_signals,
                sim_start_date=sim_start_date,
                sim_end_date=sim_end_date,
            )
        else:
            res = session.run(
                symbols_data=sampled_data,
                sim_start_date=sim_start_date,
                sim_end_date=sim_end_date,
            )

        res["window_idx"] = window_idx
        res["mc_iter"] = mc_iter
        return res
    except Exception as e:
        logger.error(f"  ❌ Error in WF worker (w={window_idx}, iter={mc_iter}): {e}")
        return None


def _walk_forward_mc_worker(args):
    """Picklable entry point for multiprocessing ProcessPoolExecutor."""
    (
        strategy_config,
        window_symbols_data,
        window_idx,
        mc_iter,
        sample_ratio,
        sim_start_date,
        sim_end_date,
        indicator_store_data,
        precomputed_signals,
    ) = args

    symbols = list(window_symbols_data.keys())
    k = max(5, int(len(symbols) * sample_ratio))
    rng = random.Random(window_idx * 10000 + mc_iter)
    sampled_syms = rng.sample(symbols, min(k, len(symbols)))
    sampled_data = {s: window_symbols_data[s] for s in sampled_syms}

    if precomputed_signals is not None:
        return _walk_forward_mc_worker_with_signals(
            strategy_config,
            sampled_data,
            window_idx,
            mc_iter,
            sim_start_date,
            sim_end_date,
            precomputed_signals,
        )

    from scripts.vectorbt_indicator_batch import IndicatorStore

    store = IndicatorStore(indicator_store_data) if indicator_store_data is not None else None
    window_signals = _precompute_window_signals(sampled_data, store, strategy_config)

    session = PortfolioBacktestSession(strategy_config=strategy_config)

    if store is not None:
        session.set_indicator_store(store)

    if window_signals:
        res = session.run_with_signals(
            symbols_data=sampled_data,
            precomputed_signals=window_signals,
            sim_start_date=sim_start_date,
            sim_end_date=sim_end_date,
        )
    else:
        res = session.run(
            symbols_data=sampled_data,
            sim_start_date=sim_start_date,
            sim_end_date=sim_end_date,
        )

    res["window_idx"] = window_idx
    res["mc_iter"] = mc_iter
    return res


def _walk_forward_mc_worker_sequential(
    strategy_config,
    window_symbols_data,
    window_idx,
    mc_iter,
    sample_ratio,
    sim_start_date,
    sim_end_date,
    indicator_store_data,
    precomputed_signals,
):
    """Run sequential fallback for Walk-Forward Monte Carlo iteration."""
    args = (
        strategy_config,
        window_symbols_data,
        window_idx,
        mc_iter,
        sample_ratio,
        sim_start_date,
        sim_end_date,
        indicator_store_data,
        precomputed_signals,
    )
    return _walk_forward_mc_worker(args)
