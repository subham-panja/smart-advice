"""
Replay Helpers for Slippage & Execution Tests
==============================================

Helper routines for pre-caching data, generating vectorbt signal matrices,
and pre-populating MongoDB recommendations for offline test replay.
"""

import glob
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np

# Add backend/ to path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


def pre_cache_nse_stocks(period="5y"):
    """Pre-cache historical data for all NSE stocks to prevent yfinance rate limiting during replay."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from scripts.data_fetcher import get_all_nse_symbols, get_historical_data

    print("\n  [PRE-CACHE] Loading all NSE symbols...")
    all_syms = get_all_nse_symbols()
    symbols_list = list(all_syms.keys()) if isinstance(all_syms, dict) else list(all_syms)
    data_dir = os.path.join(BACKEND_DIR, "data", "historical")

    def _count_cached(period_name):
        suffix = f"_{period_name}_1d.csv"
        cached = set()
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith(suffix) and not f.startswith("^"):
                    cached.add(f.replace(suffix, ""))
        return cached

    cached_long = _count_cached(period)
    uncached_long = [s for s in symbols_list if s not in cached_long]
    print(f"  [PRE-CACHE] Long-period ({period}): {len(cached_long)} cached, {len(uncached_long)} to download")

    cached_60d = _count_cached("60d")
    uncached_60d = [s for s in symbols_list if s not in cached_60d]
    print(f"  [PRE-CACHE] Short-period (60d): {len(cached_60d)} cached, {len(uncached_60d)} to download")

    def _download_batch(uncached, period_name, max_workers=4):
        if not uncached:
            print(f"  [PRE-CACHE] {period_name}: All cached - skipping")
            return 0, 0

        print(f"  [PRE-CACHE] Downloading {period_name} data for {len(uncached)} stocks ({max_workers} threads)...")
        downloaded = 0
        failed = 0
        t0 = time.time()

        def _fetch(symbol):
            try:
                df = get_historical_data(symbol, period=period_name)
                return symbol, df is not None and not df.empty
            except Exception:
                return symbol, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch, s): s for s in uncached}
            for i, future in enumerate(as_completed(futures), 1):
                symbol, success = future.result()
                if success:
                    downloaded += 1
                else:
                    failed += 1

        elapsed = time.time() - t0
        print(f"  [PRE-CACHE] Downloaded {downloaded} files in {elapsed:.1f}s ({failed} failed)")
        return downloaded, failed

    dl_long, fail_long = _download_batch(uncached_long, period)
    total_long = len(cached_long) + dl_long
    dl_60d, fail_60d = _download_batch(uncached_60d, "60d", max_workers=6)
    total_60d = len(cached_60d) + dl_60d

    print("  [PRE-CACHE] Complete:")
    print(f"    Long-period ({period}): {total_long} stocks cached ({fail_long} failed/delisted)")
    print(f"    Short-period (60d): {total_60d} stocks cached ({fail_60d} failed/delisted)")
    return total_long


def setup_signals(max_stocks=0, period="5y") -> Tuple[Any, Any, Any, Dict]:
    """One-time setup: load strategy, fetch data, compute signals for all (date, symbol) pairs."""
    from scripts.data_fetcher import get_historical_data
    from scripts.vectorbt_indicator_batch import compute_all_indicators
    from scripts.vectorbt_signal_generator import compute_signal_matrix
    from utils.strategy_loader import StrategyLoader

    print(f"\n{'='*60}")
    print("  PHASE A: SIGNAL PRE-COMPUTATION")
    print(f"{'='*60}")

    print("\n  [1/5] Loading strategy config...")
    strategy_config = StrategyLoader.get_strategy_by_name("Swing_Trading")
    if not strategy_config:
        raise RuntimeError("Strategy 'Swing_Trading' not found")

    data_dir = os.path.join(BACKEND_DIR, "data", "historical")
    cache_pattern = os.path.join(data_dir, f"*_{period}_1d.csv")
    cached_files = glob.glob(cache_pattern)

    suffix = f"_{period}_1d.csv"
    cached_symbols = [
        os.path.basename(f).replace(suffix, "") for f in cached_files if not os.path.basename(f).startswith("^")
    ]

    if max_stocks and len(cached_symbols) > max_stocks:
        cached_symbols = cached_symbols[:max_stocks]

    if len(cached_symbols) < 10:
        raise RuntimeError(f"Only {len(cached_symbols)} cached stocks found.")

    symbols_data = {}
    failed = 0
    t0 = time.time()
    for i, symbol in enumerate(cached_symbols):
        try:
            df = get_historical_data(symbol, period=period)
            if not df.empty and len(df) >= 250:
                symbols_data[symbol] = df
            else:
                failed += 1
        except Exception:
            failed += 1

    elapsed = time.time() - t0
    print(f"  [2/5] Loaded {len(symbols_data)} symbols in {elapsed:.1f}s")
    indicators = compute_all_indicators(symbols_data, strategy_config)

    pass_matrix, score_matrix = compute_signal_matrix(indicators, strategy_config)

    from scripts.vectorbt_signal_generator import compute_stock_prefilter

    prefilter_matrix = compute_stock_prefilter(indicators, strategy_config)

    pass_matrix = pass_matrix & prefilter_matrix
    score_matrix = score_matrix.where(pass_matrix, 0.0)

    return pass_matrix, score_matrix, indicators, strategy_config


def prepopulate_recommendations(date, pass_matrix, score_matrix, indicators, strategy_config) -> int:
    """Pre-populate MongoDB recommended_shares with BUY signals for a specific date."""
    from database import get_mongodb

    db = get_mongodb()

    matrix_dates = pass_matrix.index
    matrix_dates_naive = matrix_dates.tz_localize(None) if matrix_dates.tz is not None else matrix_dates

    mask = matrix_dates_naive.date == date
    if not mask.any():
        return 0

    date_idx = matrix_dates[mask][0]
    day_pass = pass_matrix.loc[date_idx]
    day_score = score_matrix.loc[date_idx]
    buy_symbols = day_pass[day_pass].index.tolist()

    if not buy_symbols:
        return 0

    risk_cfg = strategy_config.get("risk_management", {})
    max_positions = risk_cfg.get("max_positions", 15)
    initial_capital = 10000.0
    capital_per_position = initial_capital / max_positions

    exit_rules = strategy_config.get("exit_rules", {})
    sl_atr_mult = exit_rules.get("stop_loss_atr_multiplier", 2.0)
    target_atr_mult = exit_rules.get("target_atr_multiplier", 3.0)

    rec_date = datetime(date.year, date.month, date.day, 0, 0, 0)
    now = datetime.utcnow()
    inserted = 0
    strat_name = strategy_config["name"]

    for symbol in buy_symbols:
        try:
            close_price = float(indicators.close.at[date_idx, symbol])
            atr = float(indicators.atr_14.at[date_idx, symbol])
            tech_score = float(day_score[symbol])
        except (KeyError, TypeError, ValueError):
            continue

        if np.isnan(close_price) or np.isnan(atr) or close_price <= 0 or atr <= 0:
            continue

        buy_price = round(close_price, 2)
        stop_loss = round(close_price - sl_atr_mult * atr, 2)
        sell_price = round(close_price + target_atr_mult * atr, 2)
        quantity = max(1, math.floor(capital_per_position / buy_price))

        risk = buy_price - stop_loss
        reward = sell_price - buy_price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
        allocation_pct = round((buy_price * quantity / initial_capital) * 100, 2)

        doc = {
            "symbol": symbol,
            "filtered_stock_id": None,
            "company_name": symbol,
            "technical_score": tech_score,
            "combined_score": tech_score,
            "recommendation_strength": "BUY",
            "reason": f"Vectorbt signal: score={tech_score:.2f}",
            "buy_price": buy_price,
            "sell_price": sell_price,
            "stop_loss": stop_loss,
            "backtest_metrics": {},
            "suggested_quantity": quantity,
            "allocation_pct": allocation_pct,
            "rr_ratio": rr_ratio,
            "strategy_name": strat_name,
            "recommendation_date": rec_date,
            "created_at": now,
            "updated_at": now,
        }

        db.recommended_shares.update_one(
            {"symbol": symbol, "recommendation_date": rec_date},
            {"$set": doc},
            upsert=True,
        )
        inserted += 1

    return inserted
