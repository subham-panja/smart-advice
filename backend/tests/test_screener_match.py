#!/usr/bin/env python3
"""
Screener vs Local Filter Comparison Test
==========================================

Compares screener API results with the local compute_stock_prefilter
to ensure they produce the exact same stock universe.

Uses the ROLLING CACHE (parquet files) — the same data the backtest uses.
Make sure your cache is up to date before running:
    python scripts/sync_historical_data.py

Usage:
    cd backend
    python tests/test_screener_match.py
    python tests/test_screener_match.py --strategy Swing_Trading --verbose
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("ScreenerMatch")


def call_screener_api(strategy_config: dict) -> set:
    """Call real screener API with the same scan clause the live pipeline uses."""
    from scripts.screener_filter import ScreenerFilter
    from utils.filter_translator import FilterTranslator

    filters = strategy_config["stock_filters"]
    scan_clause = FilterTranslator.translate_to_scan_clause(filters)
    print(f"\nScreener scan clause:\n  {scan_clause}\n")

    cf = ScreenerFilter()
    syms = cf.get_filtered_symbols(scan_clause=scan_clause)
    return set(syms.keys())


def load_cached_data() -> tuple[dict, pd.Timestamp]:
    """Load ALL cached parquet data — the same rolling cache the backtest uses.

    Returns (symbols_data, latest_date_in_cache).
    """
    from utils.data_cache import CACHE_DIR

    parquet_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
    cached_symbols = [f.replace(".parquet", "") for f in parquet_files]
    print(f"Found {len(cached_symbols)} cached parquet files in {CACHE_DIR}")

    symbols_data = {}
    failed = 0
    latest_date = pd.Timestamp("2000-01-01")

    for i, sym in enumerate(cached_symbols):
        try:
            fpath = os.path.join(CACHE_DIR, f"{sym}.parquet")
            df = pd.read_parquet(fpath)
            if df is not None and not df.empty and len(df) > 50:
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                cols_needed = ["Open", "High", "Low", "Close", "Volume"]
                if all(c in df.columns for c in cols_needed):
                    symbols_data[sym] = df[cols_needed]
                    sym_end = df.index[-1]
                    if sym_end > latest_date:
                        latest_date = sym_end
                else:
                    failed += 1
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i+1}/{len(cached_symbols)} ({len(symbols_data)} ok, {failed} failed)")

    print(f"  Loaded data for {len(symbols_data)} symbols from cache ({failed} failed/skipped)")
    print(f"  Cache latest date: {latest_date.date()}")

    today = pd.Timestamp.today().normalize()
    staleness = (today - latest_date.normalize()).days
    if staleness > 5:
        print(f"\n  WARNING: Cache is {staleness} days stale! Run 'python scripts/sync_historical_data.py' first.")
        print(
            f"  Screener API returns TODAY's results but cache ends {latest_date.date()} — comparison will be inaccurate.\n"
        )

    return symbols_data, latest_date


def run_local_filter(symbols_data: dict, strategy_config: dict) -> tuple[set, dict]:
    """Run local compute_stock_prefilter on the cached data."""
    from scripts.vectorbt_indicator_batch import compute_all_indicators
    from scripts.vectorbt_signal_generator import compute_stock_prefilter

    if len(symbols_data) < 50:
        raise RuntimeError(f"Too few symbols loaded: {len(symbols_data)}")

    print(f"\nComputing indicators via vectorbt for {len(symbols_data)} symbols...")
    indicators = compute_all_indicators(symbols_data, strategy_config)

    print("Applying local stock prefilter...")
    prefilter_matrix = compute_stock_prefilter(indicators, strategy_config)

    latest_date = indicators.dates[-1]
    print(f"Latest date in indicators: {latest_date.date()}")

    today = pd.Timestamp.today().normalize()
    target_date = None
    for d in reversed(indicators.dates):
        if d <= today:
            target_date = d
            break

    if target_date is None:
        target_date = latest_date

    print(f"Filtering on date: {target_date.date()}")

    passing = set()
    for sym in indicators.symbols:
        if target_date in prefilter_matrix.index:
            if prefilter_matrix.loc[target_date, sym]:
                passing.add(sym)

    return passing, {
        "indicators": indicators,
        "prefilter_matrix": prefilter_matrix,
        "target_date": target_date,
        "symbols_data": symbols_data,
    }


def diagnose_mismatches(
    screener_syms: set,
    local_syms: set,
    context: dict,
    strategy_config: dict,
):
    """For each mismatched symbol, diagnose WHY it differs."""
    indicators = context["indicators"]
    target_date = context["target_date"]
    symbols_data = context["symbols_data"]

    only_screener = sorted(screener_syms - local_syms)
    only_local = sorted(local_syms - screener_syms)

    filters = strategy_config.get("stock_filters", [])

    print(f"\n{'='*70}")
    print("DIAGNOSIS: Stocks in screener but NOT in local filter (false negatives)")
    print(f"{'='*70}")
    print(f"Count: {len(only_screener)}")

    for sym in only_screener[:40]:
        reasons = []
        if sym not in indicators.symbols:
            reasons.append("NOT IN CACHE")
        elif sym not in symbols_data:
            reasons.append("NO RAW DATA")
        else:
            df = symbols_data.get(sym)
            if df is not None:
                data_end = df.index[-1]
                if data_end.normalize() < target_date.normalize():
                    reasons.append(f"CACHE ENDS {data_end.date()} < target {target_date.date()}")

            for f in filters:
                f_type = f["type"]
                try:
                    if sym not in indicators.close.columns:
                        reasons.append("NOT IN INDICATORS")
                        break

                    if f_type == "price":
                        val = indicators.close.loc[target_date, sym]
                        op = f["op"]
                        if pd.isna(val):
                            reasons.append("PRICE=NaN")
                        elif op == "between":
                            if not (f["min"] < val < f["max"]):
                                reasons.append(f"PRICE={val:.2f} (need {f['min']}<{val:.2f}<{f['max']})")
                        elif op == ">" and val <= f["value"]:
                            reasons.append(f"PRICE={val:.2f} <= {f['value']}")
                    elif f_type == "volume":
                        val = indicators.volume.loc[target_date, sym]
                        if pd.isna(val):
                            reasons.append("VOL=NaN")
                        elif val <= f["value"]:
                            reasons.append(f"VOL={val:.0f} <= {f['value']}")
                    elif f_type == "rsi":
                        val = indicators.rsi_14.loc[target_date, sym]
                        op = f["op"]
                        if pd.isna(val):
                            reasons.append("RSI=NaN")
                        elif op == ">" and val <= f["value"]:
                            reasons.append(f"RSI={val:.1f} <= {f['value']}")
                    elif f_type == "moving_average":
                        period = f["period"]
                        op = f["op"]
                        sma_map = {
                            20: indicators.sma_20,
                            50: indicators.sma_50,
                            150: indicators.sma_150,
                            200: indicators.sma_200,
                        }
                        sma_df = sma_map.get(period)
                        if sma_df is not None and sym in sma_df.columns:
                            close_val = indicators.close.loc[target_date, sym]
                            sma_val = sma_df.loc[target_date, sym]
                            if pd.isna(sma_val):
                                reasons.append(f"SMA{period}=NaN")
                            elif op == ">" and close_val <= sma_val:
                                reasons.append(f"Close({close_val:.2f}) <= SMA{period}({sma_val:.2f})")
                        else:
                            reasons.append(f"SMA{period}=NOT COMPUTED")
                    elif f_type == "volume_spike_lookup":
                        lookback = f["lookback_days"]
                        multiplier = f["multiplier"]
                        ma_period = f["ma_period"]
                        import vectorbt as vbt

                        v = indicators.volume
                        vol_sma = vbt.talib("SMA").run(v, timeperiod=ma_period).real
                        vol_sma.columns = indicators.symbols
                        found_spike = False
                        for i in range(1, lookback + 1):
                            try:
                                past_vol = v.shift(i).loc[target_date, sym]
                                past_sma = vol_sma.shift(i).loc[target_date, sym]
                                if not pd.isna(past_vol) and not pd.isna(past_sma):
                                    if past_vol > past_sma * multiplier:
                                        found_spike = True
                                        break
                            except Exception:
                                pass
                        if not found_spike:
                            reasons.append(f"NO VOL SPIKE in last {lookback} days")
                    elif f_type == "market_cap":
                        try:
                            from scripts.data_fetcher import get_market_caps

                            mc = get_market_caps([sym])
                            cap = mc.get(sym, None)
                            if cap is None:
                                reasons.append("MCAP=MISSING")
                            elif cap <= f.get("value", 0):
                                reasons.append(f"MCAP={cap:.0f}Cr <= {f['value']}Cr")
                        except Exception:
                            reasons.append("MCAP=ERROR")
                except Exception as e:
                    reasons.append(f"CHECK_ERROR: {e}")

        status = " | ".join(reasons) if reasons else "UNKNOWN (all individual checks passed)"
        print(f"  {sym:20s} -> {status}")

    if len(only_screener) > 40:
        print(f"  ... and {len(only_screener) - 40} more")

    print(f"\n{'='*70}")
    print("DIAGNOSIS: Stocks in local filter but NOT in screener (false positives)")
    print(f"{'='*70}")
    print(f"Count: {len(only_local)}")

    for sym in only_local[:40]:
        if sym not in indicators.symbols:
            print(f"  {sym:20s} -> NOT IN INDICATORS (shouldn't happen)")
            continue

        close_val = indicators.close.loc[target_date, sym] if target_date in indicators.close.index else None
        vol_val = indicators.volume.loc[target_date, sym] if target_date in indicators.volume.index else None
        rsi_val = indicators.rsi_14.loc[target_date, sym] if target_date in indicators.rsi_14.index else None

        info_parts = []
        if close_val is not None and not pd.isna(close_val):
            info_parts.append(f"Close={close_val:.2f}")
        if vol_val is not None and not pd.isna(vol_val):
            info_parts.append(f"Vol={vol_val:.0f}")
        if rsi_val is not None and not pd.isna(rsi_val):
            info_parts.append(f"RSI={rsi_val:.1f}")

        for f in filters:
            if f["type"] == "market_cap":
                try:
                    from scripts.data_fetcher import get_market_caps

                    mc = get_market_caps([sym])
                    cap = mc.get(sym, None)
                    if cap is not None:
                        info_parts.append(f"MCAP={cap:.0f}Cr")
                    else:
                        info_parts.append("MCAP=MISSING")
                except Exception:
                    info_parts.append("MCAP=ERROR")

        info = " | ".join(info_parts) if info_parts else "no data"
        print(f"  {sym:20s} -> {info} (screener excluded this)")

    if len(only_local) > 40:
        print(f"  ... and {len(only_local) - 40} more")


def main():
    parser = argparse.ArgumentParser(description="Compare screener API vs local filter results")
    parser.add_argument("--strategy", type=str, default="Swing_Trading", help="Strategy name")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--skip-screener", action="store_true", help="Skip screener API call, use saved symbols")
    args = parser.parse_args()

    from utils.strategy_loader import StrategyLoader

    strategy = StrategyLoader.get_strategy_by_name(args.strategy)
    if not strategy:
        print(f"Error: Strategy '{args.strategy}' not found")
        sys.exit(1)

    print(f"{'='*70}")
    print("SCREENER vs LOCAL FILTER COMPARISON")
    print(f"Strategy: {strategy['name']}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    saved_screener_path = os.path.join(os.path.dirname(__file__), ".screener_cache.json")

    if args.skip_screener and os.path.exists(saved_screener_path):
        with open(saved_screener_path) as f:
            screener_syms = set(json.load(f))
        print(f"\n[1/2] Using saved screener symbols ({len(screener_syms)} symbols)")
    else:
        print("\n[1/2] Calling screener API (live)...")
        screener_syms = call_screener_api(strategy)
        print(f"  Screener returned: {len(screener_syms)} symbols")

        if screener_syms:
            with open(saved_screener_path, "w") as f:
                json.dump(sorted(screener_syms), f)

    if not screener_syms:
        print("  ERROR: Screener returned 0 symbols. Cannot compare.")
        sys.exit(1)

    print("\n[2/2] Loading cached data and running local filter...")
    symbols_data, cache_latest = load_cached_data()
    local_syms, context = run_local_filter(symbols_data, strategy)
    print(f"  Local filter returned: {len(local_syms)} symbols")

    matched = screener_syms & local_syms
    only_screener = screener_syms - local_syms
    only_local = local_syms - screener_syms
    total = len(screener_syms | local_syms)
    match_pct = (len(matched) / total * 100) if total > 0 else 0

    target_date = context["target_date"]

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Target date:       {target_date.date()}")
    print(f"  Cache latest:      {cache_latest.date()}")
    print(f"  Screener symbols:  {len(screener_syms)}")
    print(f"  Local symbols:     {len(local_syms)}")
    print(f"  Matched:           {len(matched)}")
    print(f"  Only in Screener:  {len(only_screener)}")
    print(f"  Only in Local:     {len(only_local)}")
    print(f"  Match %:           {match_pct:.1f}%")

    if match_pct == 100.0:
        print("\n  PERFECT MATCH! Local filter exactly matches screener API.")
    else:
        print("\n  MISMATCH DETECTED. Running diagnosis...")
        diagnose_mismatches(screener_syms, local_syms, context, strategy)

    print(f"\n{'='*70}")
    return match_pct


if __name__ == "__main__":
    pct = main()
    sys.exit(0 if pct == 100.0 else 1)
