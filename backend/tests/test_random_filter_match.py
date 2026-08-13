#!/usr/bin/env python3
"""
Random Filter Comparison Test: Local vs Chartink
==================================================

Generates 10 completely random filter configurations and compares results
between local compute_stock_prefilter and Chartink API.

Goal: Verify that our local filtering logic matches Chartink's screening
to ensure backtest results will align with live trading.

Pass Criteria: Each filter must achieve ≥95% match rate.

Usage:
    cd backend
    python tests/test_random_filter_match.py
    python tests/test_random_filter_match.py --filters 15 --verbose
    python tests/test_random_filter_match.py --seed 42 --min-match 90
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

# ---------------------------------------------------------------------------
# Random Filter Generator
# ---------------------------------------------------------------------------


def generate_random_filters(seed: int = None) -> list:
    """Generate 10 diverse random filter configurations.

    Each configuration tests different edge cases:
    1. Price-only filters
    2. Volume-only filters
    3. RSI filters
    4. Moving average crossover filters
    5. Volume spike filters
    6. Multi-filter combinations (price + volume)
    7. Multi-filter combinations (RSI + MA)
    8. Complex multi-filter (price + volume + RSI + MA)
    9. Edge case: very restrictive filters
    10. Edge case: very permissive filters
    """
    if seed is not None:
        random.seed(seed)

    filters_list = []

    # 1. Price range filter (between)
    filters_list.append(
        {
            "name": "Price Range (Between)",
            "filters": [
                {"type": "price", "op": "between", "min": random.uniform(20, 100), "max": random.uniform(5000, 10000)}
            ],
        }
    )

    # 2. Price threshold filter (greater than)
    filters_list.append(
        {"name": "Price Threshold (>)", "filters": [{"type": "price", "op": ">", "value": random.uniform(50, 500)}]}
    )

    # 3. Volume filter
    filters_list.append(
        {"name": "Volume Threshold", "filters": [{"type": "volume", "op": ">", "value": random.randint(50000, 500000)}]}
    )

    # 4. RSI filter
    rsi_period = random.choice([14, 9])
    rsi_threshold = random.uniform(30, 70)
    rsi_op = random.choice([">", "<"])
    filters_list.append(
        {
            "name": f"RSI({rsi_period}) {rsi_op} {rsi_threshold:.0f}",
            "filters": [{"type": "rsi", "op": rsi_op, "period": rsi_period, "value": rsi_threshold}],
        }
    )

    # 5. Moving average filter (close vs SMA)
    ma_period = random.choice([20, 50, 150, 200])
    ma_op = random.choice([">", "<"])
    filters_list.append(
        {
            "name": f"Close {ma_op} SMA({ma_period})",
            "filters": [{"type": "moving_average", "kind": "SMA", "period": ma_period, "op": ma_op, "target": "close"}],
        }
    )

    # 6. Volume spike lookup
    filters_list.append(
        {
            "name": "Volume Spike (Lookback)",
            "filters": [
                {
                    "type": "volume_spike_lookup",
                    "lookback_days": random.randint(5, 15),
                    "multiplier": random.uniform(1.0, 2.0),
                    "ma_period": random.choice([20, 50]),
                }
            ],
        }
    )

    # 7. Multi-filter: Price + Volume
    filters_list.append(
        {
            "name": "Multi: Price + Volume",
            "filters": [
                {"type": "price", "op": ">", "value": random.uniform(100, 300)},
                {"type": "volume", "op": ">", "value": random.randint(100000, 300000)},
            ],
        }
    )

    # 8. Multi-filter: RSI + Moving Average
    filters_list.append(
        {
            "name": "Multi: RSI + SMA",
            "filters": [
                {"type": "rsi", "op": ">", "period": 14, "value": random.uniform(50, 65)},
                {"type": "moving_average", "kind": "SMA", "period": 50, "op": ">", "target": "close"},
            ],
        }
    )

    # 9. Complex: Price + Volume + RSI + MA
    filters_list.append(
        {
            "name": "Complex: Price+Vol+RSI+MA",
            "filters": [
                {"type": "price", "op": "between", "min": 50, "max": 5000},
                {"type": "volume", "op": ">", "value": random.randint(100000, 200000)},
                {"type": "rsi", "op": ">", "period": 14, "value": random.uniform(50, 60)},
                {"type": "moving_average", "kind": "SMA", "period": 50, "op": ">", "target": "close"},
            ],
        }
    )

    # 10. Very restrictive (all filters must pass)
    filters_list.append(
        {
            "name": "Very Restrictive (All)",
            "filters": [
                {"type": "price", "op": "between", "min": 100, "max": 2000},
                {"type": "volume", "op": ">", "value": 300000},
                {"type": "rsi", "op": ">", "period": 14, "value": 60},
                {"type": "moving_average", "kind": "SMA", "period": 20, "op": ">", "target": "close"},
                {"type": "moving_average", "kind": "SMA", "period": 50, "op": ">", "target": "close"},
            ],
        }
    )

    return filters_list


# ---------------------------------------------------------------------------
# Chartink API Caller
# ---------------------------------------------------------------------------


def call_chartink_api(filters: list, verbose: bool = False) -> set:
    """Call Chartink API with given filters and return symbol set."""
    from scripts.screener_filter import ScreenerFilter
    from utils.filter_translator import FilterTranslator

    scan_clause = FilterTranslator.translate_to_scan_clause(filters)

    if verbose:
        print(f"  Chartink scan_clause: {scan_clause}")

    cf = ScreenerFilter()
    syms = cf.get_filtered_symbols(scan_clause=scan_clause)
    return set(syms.keys())


# ---------------------------------------------------------------------------
# Local Filter Runner
# ---------------------------------------------------------------------------


def run_local_filter(
    symbols_data: dict, filters: list, indicators=None, target_date: pd.Timestamp = None, verbose: bool = False
) -> tuple:
    """Run local compute_stock_prefilter on cached data.

    Args:
        symbols_data: Dict of symbol -> DataFrame
        filters: List of filter configurations
        indicators: Pre-computed indicators (optional)
        target_date: Specific date to filter on (optional, defaults to latest)
        verbose: Print verbose output

    Returns: (passing_symbols_set, actual_target_date)
    """
    from scripts.vectorbt_indicator_batch import compute_all_indicators
    from scripts.vectorbt_signal_generator import compute_stock_prefilter

    # Use provided indicators or compute new ones
    if indicators is None:
        if verbose:
            print(f"  Computing indicators for {len(symbols_data)} symbols...")
        # Create minimal strategy config for indicator computation
        strategy_config = {"stock_filters": filters, "swing_trading_gates": {}}
        indicators = compute_all_indicators(symbols_data, strategy_config)

    # Apply filters
    strategy_config = {"stock_filters": filters, "swing_trading_gates": {}}

    if verbose:
        print("  Applying local prefilter...")

    prefilter_matrix = compute_stock_prefilter(indicators, strategy_config)

    # Find target date (use provided or default to most recent trading day)
    if target_date is None:
        today = pd.Timestamp.today().normalize()
        for d in reversed(indicators.dates):
            if d <= today:
                target_date = d
                break

    if target_date is None:
        target_date = indicators.dates[-1]

    # Extract passing symbols on target date
    passing = set()
    for sym in indicators.symbols:
        if target_date in prefilter_matrix.index:
            if prefilter_matrix.loc[target_date, sym]:
                passing.add(sym)

    return passing, target_date


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------


def load_cached_data(verbose: bool = False) -> dict:
    """Load cached parquet data for all symbols."""
    from utils.data_cache import CACHE_DIR

    if verbose:
        print(f"Loading cached data from {CACHE_DIR}...")

    parquet_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
    cached_symbols = [f.replace(".parquet", "") for f in parquet_files]

    if verbose:
        print(f"  Found {len(cached_symbols)} cached files")

    symbols_data = {}
    failed = 0

    for sym in cached_symbols:
        try:
            fpath = os.path.join(CACHE_DIR, f"{sym}.parquet")
            df = pd.read_parquet(fpath)
            if df is not None and not df.empty and len(df) > 50:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                cols_needed = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in df.columns for c in cols_needed):
                    symbols_data[sym] = df[cols_needed]
                else:
                    failed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    if verbose:
        print(f"  Loaded {len(symbols_data)} symbols ({failed} failed)")

    return symbols_data


def find_real_latest_date(symbols_data: dict, min_coverage: float = 0.9) -> pd.Timestamp:
    """Find the latest date that has real data for at least min_coverage of symbols.

    Avoids forward-filled edge dates where only a handful of symbols have real data.
    """
    date_counts: dict = {}
    for df in symbols_data.values():
        last_date = df.index[-1]
        d = pd.Timestamp(last_date).normalize()
        date_counts[d] = date_counts.get(d, 0) + 1

    total = len(symbols_data)
    for d in sorted(date_counts, reverse=True):
        if date_counts[d] >= total * min_coverage:
            return d
    return max(date_counts) if date_counts else pd.Timestamp.today().normalize()


# ---------------------------------------------------------------------------
# Comparison Logic
# ---------------------------------------------------------------------------


def compare_results(chartink_syms: set, local_syms: set, symbols_data: dict, verbose: bool = False) -> dict:
    """Compare Chartink vs local filter results.

    Filters out:
    - Indices (^NSEI, etc.) from both sets
    - Symbols not in our cache from Chartink (universe mismatch)
    - ETFs and mutual funds (often have different ticker formats)

    Compares only on the overlapping universe to isolate indicator differences.
    """
    # Filter out indices
    chartink_syms = {s for s in chartink_syms if not s.startswith("^")}
    local_syms = {s for s in local_syms if not s.startswith("^")}

    # Filter Chartink to only symbols in our cache (remove universe mismatch)
    cache_symbols = set(symbols_data.keys())
    chartink_in_cache = chartink_syms & cache_symbols
    local_in_cache = local_syms & cache_symbols

    # Compare on overlapping universe
    matched = chartink_in_cache & local_in_cache
    only_chartink = chartink_in_cache - local_in_cache
    only_local = local_in_cache - chartink_in_cache

    # Total universe (both in cache)
    total_universe = len(chartink_in_cache | local_in_cache)

    # Match percentage on overlapping universe
    match_pct = (len(matched) / total_universe * 100) if total_universe > 0 else 100.0

    # Also calculate raw match (including universe mismatch)
    raw_matched = chartink_syms & local_syms
    raw_total = len(chartink_syms | local_syms)
    raw_match_pct = (len(raw_matched) / raw_total * 100) if raw_total > 0 else 100.0

    return {
        "chartink_count": len(chartink_syms),
        "chartink_in_cache": len(chartink_in_cache),
        "local_count": len(local_syms),
        "matched": len(matched),
        "only_chartink": len(only_chartink),
        "only_local": len(only_local),
        "match_pct": match_pct,
        "raw_match_pct": raw_match_pct,
        "universe_overlap": total_universe,
        "only_chartink_syms": sorted(only_chartink)[:20] if verbose else [],
        "only_local_syms": sorted(only_local)[:20] if verbose else [],
    }


# ---------------------------------------------------------------------------
# Logic Verification (VBT vs Manual TA-Lib)
# ---------------------------------------------------------------------------


def run_logic_verification(
    symbols_data: dict,
    filters_list: list,
    indicators=None,
    target_date: pd.Timestamp = None,
    verbose: bool = False,
) -> list:
    """Verify local filter logic by comparing VBT prefilter vs manual TA-Lib.

    Both methods use the same aligned data to ensure a fair comparison.
    Returns a list of result dicts with pass/fail for each filter.
    """
    import numpy as np
    import talib

    from scripts.vectorbt_indicator_batch import _align_to_common_dates, compute_all_indicators
    from scripts.vectorbt_signal_generator import compute_stock_prefilter

    if indicators is None:
        strategy_config = {"stock_filters": [], "swing_trading_gates": {}}
        indicators = compute_all_indicators(symbols_data, strategy_config)

    if target_date is None:
        target_date = indicators.dates[-1]

    # Align data the same way compute_all_indicators does
    common_dates, aligned = _align_to_common_dates(symbols_data)

    results = []
    for fc in filters_list:
        filter_name = fc["name"]
        filters = fc["filters"]

        cfg = {"stock_filters": filters, "swing_trading_gates": {}}
        prefilter = compute_stock_prefilter(indicators, cfg)
        vbt_pass = set(s for s in indicators.symbols if prefilter.loc[target_date, s])

        manual_pass = set()
        for sym in indicators.symbols:
            adf = aligned[sym]
            close = adf["Close"].values.astype(float)
            vol = adf["Volume"].values.astype(float)

            passes = True
            for f in filters:
                f_type = f["type"]
                if f_type == "price":
                    c = close[-1]
                    if np.isnan(c):
                        passes = False
                        continue
                    op = f["op"]
                    if op == ">":
                        passes = passes and (c > f["value"])
                    elif op == "<":
                        passes = passes and (c < f["value"])
                    elif op == "between":
                        passes = passes and (c > f["min"]) and (c < f["max"])
                elif f_type == "volume":
                    if np.isnan(vol[-1]):
                        passes = False
                    else:
                        passes = passes and (vol[-1] > f["value"])
                elif f_type == "rsi":
                    period = f.get("period", 14)
                    rsi = talib.RSI(close, timeperiod=period)
                    last = rsi[-1]
                    if np.isnan(last):
                        passes = False
                    elif f["op"] == "between":
                        passes = passes and (last >= f["min"]) and (last <= f["max"])
                    elif f["op"] == ">":
                        passes = passes and (last > f["value"])
                    elif f["op"] == "<":
                        passes = passes and (last < f["value"])
                elif f_type == "moving_average":
                    kind = f["kind"].lower()
                    if kind == "hma" or f["op"] == "monitor":
                        continue
                    period = f["period"]
                    sma = talib.SMA(close, timeperiod=period)
                    last = sma[-1]
                    if np.isnan(last):
                        passes = False
                    elif f["op"] == ">":
                        passes = passes and (close[-1] > last)
                    elif f["op"] == "<":
                        passes = passes and (close[-1] < last)
                elif f_type == "price_distance_sma":
                    period = f.get("period", 20)
                    op = f.get("op", "<=")
                    max_dist_pct = f.get("max_distance_pct", 5.0)
                    sma = talib.SMA(close, timeperiod=period)
                    last = sma[-1]
                    c = close[-1]
                    if np.isnan(last) or np.isnan(c):
                        passes = False
                    else:
                        dist_pct = abs(c - last) / last * 100.0
                        if op == "<=":
                            passes = passes and (dist_pct <= max_dist_pct)
                        elif op == "<":
                            passes = passes and (dist_pct < max_dist_pct)
                        elif op == ">=":
                            passes = passes and (dist_pct >= max_dist_pct)
                        elif op == ">":
                            passes = passes and (dist_pct > max_dist_pct)
                elif f_type == "volume_spike_lookup":
                    lookback = f["lookback_days"]
                    multiplier = f["multiplier"]
                    ma_period = f["ma_period"]
                    # Current date must have valid volume and vol_sma
                    if np.isnan(vol[-1]):
                        passes = False
                        continue
                    vol_sma = talib.SMA(vol, timeperiod=ma_period)
                    if np.isnan(vol_sma[-1]):
                        passes = False
                        continue
                    spike = False
                    for i in range(1, lookback + 1):
                        idx = len(vol) - 1 - i
                        if idx < 0 or np.isnan(vol[idx]) or np.isnan(vol_sma[idx]):
                            continue
                        if vol[idx] > vol_sma[idx] * multiplier:
                            spike = True
                            break
                    passes = passes and spike

            if passes:
                manual_pass.add(sym)

        match = vbt_pass == manual_pass
        only_vbt = len(vbt_pass - manual_pass)
        only_manual = len(manual_pass - vbt_pass)

        results.append(
            {
                "filter": filter_name,
                "vbt_count": len(vbt_pass),
                "manual_count": len(manual_pass),
                "match": match,
                "only_vbt": only_vbt,
                "only_manual": only_manual,
                "status": "PASS" if match else "FAIL",
            }
        )

        if verbose and not match:
            print(f"  Only VBT: {sorted(vbt_pass - manual_pass)[:10]}")
            print(f"  Only Manual: {sorted(manual_pass - vbt_pass)[:10]}")

    return results


# ---------------------------------------------------------------------------
# Main Test Runner
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Random filter comparison test")
    parser.add_argument("--filters", type=int, default=10, help="Number of random filters (default 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--min-match", type=float, default=95.0, help="Minimum match percentage (default 95)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between Chartink API calls (seconds)")
    parser.add_argument(
        "--historical", action="store_true", help="Test on a historical date (both systems have same data)"
    )
    parser.add_argument(
        "--logic-only",
        action="store_true",
        help="Skip Chartink API, verify local filter logic only (VBT vs manual TA-Lib)",
    )
    args = parser.parse_args()

    print(f"{'='*70}")
    print("RANDOM FILTER COMPARISON TEST: Local vs Chartink")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Filters to test: {args.filters}")
    print(f"Minimum match: {args.min_match}%")
    print(f"Random seed: {args.seed if args.seed else 'None (truly random)'}")
    print(f"{'='*70}\n")

    # Load cached data once
    print("[1/3] Loading cached data...")
    symbols_data = load_cached_data(verbose=args.verbose)

    if len(symbols_data) < 100:
        print(f"ERROR: Only {len(symbols_data)} symbols loaded. Need at least 100.")
        print("Run 'python scripts/sync_historical_data.py' first to populate cache.")
        sys.exit(1)

    print(f"  Loaded {len(symbols_data)} symbols from cache\n")

    # Check cache staleness by finding the real latest date across all symbols

    real_latest_date = find_real_latest_date(symbols_data)
    today = pd.Timestamp.today()
    days_stale = (today - real_latest_date).days

    print(f"  Real latest date: {real_latest_date.date()}")
    print(f"  Today: {today.date()}")
    print(f"  Cache age: {days_stale} days")

    if days_stale > 1:
        print(f"\n  WARNING: Cache is {days_stale} days old!")
        print(f"  Chartink uses today's data but local uses {real_latest_date.date()} data.")
        print("  This causes mismatches on indicator-based filters (RSI, SMA, Volume).")
        print("  Price filters should still match well (prices change slowly).")
        print("  For accurate comparison, run after market close when yfinance updates.\n")
    else:
        print("  Cache is fresh (≤1 day old)\n")

    # Truncate all symbols to real_latest_date to avoid forward-fill edge distortion
    for sym in list(symbols_data.keys()):
        df = symbols_data[sym]
        truncated = df.loc[:real_latest_date]
        if len(truncated) > 50:
            symbols_data[sym] = truncated
        else:
            del symbols_data[sym]

    print(f"  Truncated {len(symbols_data)} symbols to {real_latest_date.date()}\n")

    # Compute indicators once (using empty filters to get all indicators)
    print("[2/3] Computing indicators (one-time)...")
    from scripts.vectorbt_indicator_batch import compute_all_indicators

    strategy_config = {"stock_filters": [], "swing_trading_gates": {}}
    indicators = compute_all_indicators(symbols_data, strategy_config)
    print(f"  Computed indicators for {len(indicators.symbols)} symbols x {len(indicators.dates)} dates\n")

    # Generate random filters
    random_filters = generate_random_filters(seed=args.seed)

    if args.filters > 10:
        for i in range(10, args.filters):
            random_filters.append({"name": f"Random Mix {i+1}", "filters": _generate_truly_random_filters()})

    # --- LOGIC-ONLY MODE ---
    if args.logic_only:
        print("[3/3] Running logic verification (VBT vs manual TA-Lib)...\n")
        logic_results = run_logic_verification(
            symbols_data,
            random_filters[: args.filters],
            indicators=indicators,
            target_date=real_latest_date,
            verbose=args.verbose,
        )

        passed = sum(1 for r in logic_results if r["status"] == "PASS")
        failed = len(logic_results) - passed

        print(f"\n{'='*70}")
        print("LOGIC VERIFICATION SUMMARY")
        print(f"{'='*70}")
        print(f"{'Filter Name':<40} {'VBT':>6} {'Manual':>8} {'Status':>8}")
        print("-" * 65)
        for r in logic_results:
            print(f"{r['filter']:<40} {r['vbt_count']:>6} {r['manual_count']:>8} {r['status']:>8}")

        print(f"\nPassed: {passed}/{len(logic_results)}")
        print(f"Failed: {failed}/{len(logic_results)}")

        if passed == len(logic_results):
            print("\nSUCCESS: All filter logic verified (VBT matches manual TA-Lib)")
            return 0
        else:
            print(f"\nFAILURE: {failed} filters have logic mismatches")
            return 1

    # --- CHARTINK COMPARISON MODE ---
    print("[3/3] Running random filter comparisons...\n")

    results = []
    passed = 0
    failed = 0

    for i, filter_config in enumerate(random_filters[: args.filters], 1):
        filter_name = filter_config["name"]
        filters = filter_config["filters"]

        print(f"[{i}/{args.filters}] Testing: {filter_name}")

        # Call Chartink API
        try:
            chartink_syms = call_chartink_api(filters, verbose=args.verbose)
            print(f"  Chartink: {len(chartink_syms)} symbols")
        except Exception as e:
            print(f"  Chartink ERROR: {e}")
            results.append({"filter": filter_name, "status": "ERROR", "error": str(e)})
            failed += 1
            continue

        # Run local filter
        try:
            local_syms, target_date = run_local_filter(
                symbols_data, filters, indicators=indicators, target_date=real_latest_date, verbose=args.verbose
            )
            print(f"  Local:    {len(local_syms)} symbols (date: {target_date.date()})")
        except Exception as e:
            print(f"  Local ERROR: {e}")
            results.append({"filter": filter_name, "status": "ERROR", "error": str(e)})
            failed += 1
            continue

        # Compare
        comparison = compare_results(chartink_syms, local_syms, symbols_data, verbose=args.verbose)

        status = "PASS" if comparison["match_pct"] >= args.min_match else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(
            f"  Match:    {comparison['matched']}/{comparison['universe_overlap']} ({comparison['match_pct']:.1f}%) [{status}]"
        )
        print(f"  Raw:      {comparison['raw_match_pct']:.1f}% (before universe filter)")

        if comparison["match_pct"] < args.min_match and args.verbose:
            print(
                f"  Only in Chartink ({comparison['only_chartink']}): {', '.join(comparison['only_chartink_syms'][:10])}"
            )
            print(f"  Only in Local ({comparison['only_local']}): {', '.join(comparison['only_local_syms'][:10])}")

        results.append(
            {
                "filter": filter_name,
                "chartink_count": comparison["chartink_count"],
                "chartink_in_cache": comparison["chartink_in_cache"],
                "local_count": comparison["local_count"],
                "matched": comparison["matched"],
                "universe": comparison["universe_overlap"],
                "match_pct": comparison["match_pct"],
                "raw_match_pct": comparison["raw_match_pct"],
                "status": status,
            }
        )

        print()

        # Rate limiting
        if i < args.filters:
            time.sleep(args.delay)

    # Final Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total filters tested: {args.filters}")
    print(f"Passed (≥{args.min_match}%): {passed}")
    print(f"Failed (<{args.min_match}%): {failed}")
    print(f"Errors: {sum(1 for r in results if r.get('status') == 'ERROR')}")
    print(f"Pass rate: {passed}/{args.filters} ({passed/args.filters*100:.1f}%)\n")

    # Print results table
    print(
        f"{'Filter Name':<40} {'Chartink':>9} {'InCache':>9} {'Local':>9} {'Match':>9} {'%':>8} {'Raw%':>8} {'Status':>8}"
    )
    print("-" * 105)
    for r in results:
        if r.get("status") == "ERROR":
            print(f"{r['filter']:<40} {'ERROR':>9}")
        else:
            print(
                f"{r['filter']:<40} {r['chartink_count']:>9} {r['chartink_in_cache']:>9} {r['local_count']:>9} {r['matched']:>9} {r['match_pct']:>7.1f}% {r['raw_match_pct']:>7.1f}% {r['status']:>8}"
            )

    print(f"{'='*70}\n")

    # Exit code
    if passed == args.filters:
        print("SUCCESS: All filters passed the match threshold!")
        return 0
    else:
        print(f"FAILURE: {failed} filters did not meet the {args.min_match}% threshold.")
        return 1


def _generate_truly_random_filters() -> list:
    """Generate a truly random combination of filters."""
    filters = []

    # Randomly decide how many filters (2-5)
    n_filters = random.randint(2, 5)

    filter_types = ["price", "volume", "rsi", "moving_average", "volume_spike_lookup"]

    for _ in range(n_filters):
        f_type = random.choice(filter_types)

        if f_type == "price":
            op = random.choice([">", "<", "between"])
            if op == "between":
                filters.append(
                    {
                        "type": "price",
                        "op": "between",
                        "min": random.uniform(20, 500),
                        "max": random.uniform(1000, 10000),
                    }
                )
            else:
                filters.append({"type": "price", "op": op, "value": random.uniform(50, 2000)})

        elif f_type == "volume":
            filters.append({"type": "volume", "op": ">", "value": random.randint(50000, 500000)})

        elif f_type == "rsi":
            filters.append(
                {
                    "type": "rsi",
                    "op": random.choice([">", "<"]),
                    "period": random.choice([9, 14]),
                    "value": random.uniform(30, 70),
                }
            )

        elif f_type == "moving_average":
            filters.append(
                {
                    "type": "moving_average",
                    "kind": "SMA",
                    "period": random.choice([20, 50, 150, 200]),
                    "op": random.choice([">", "<"]),
                    "target": "close",
                }
            )

        elif f_type == "volume_spike_lookup":
            filters.append(
                {
                    "type": "volume_spike_lookup",
                    "lookback_days": random.randint(5, 15),
                    "multiplier": random.uniform(1.0, 2.5),
                    "ma_period": random.choice([20, 50]),
                }
            )

    return filters


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
