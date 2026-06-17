#!/usr/bin/env python3
"""
Diagnose Filter Mismatches
===========================

Investigates why specific symbols pass in Chartink but not locally (or vice versa).
Compares RSI, SMA, and Volume calculations between TA-Lib and manual methods.

Usage:
    cd backend
    python tests/diagnose_filter_mismatch.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd


def load_sample_data(symbols: list) -> dict:
    """Load cached data for specific symbols."""
    from utils.data_cache import CACHE_DIR

    symbols_data = {}
    for sym in symbols:
        fpath = os.path.join(CACHE_DIR, f"{sym}.parquet")
        if os.path.exists(fpath):
            df = pd.read_parquet(fpath)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            symbols_data[sym] = df[["Open", "High", "Low", "Close", "Volume"]]

    return symbols_data


def rsi_talib(prices: pd.Series, period: int = 14) -> pd.Series:
    """TA-Lib RSI using Wilder's smoothing (same as vbt.talib('RSI'))."""
    import talib

    return pd.Series(talib.RSI(prices.values.astype(float), timeperiod=period), index=prices.index)


def rsi_wilder(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI using Wilder's smoothing (exponential)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_sma(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI using simple moving average."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def main():
    target_symbols = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "HINDUNILVR",
        "SBIN",
        "BHARTIARTL",
        "KOTAKBANK",
        "ITC",
    ]

    print("Loading data for test symbols...")
    symbols_data = load_sample_data(target_symbols)
    target_symbols = [s for s in target_symbols if s in symbols_data]
    print(f"Loaded {len(symbols_data)} symbols\n")

    # ── RSI COMPARISON ──
    print("=" * 80)
    print("RSI(14) CALCULATION COMPARISON — Latest Date")
    print("=" * 80)
    print(
        f"\n{'Symbol':<15} {'Close':>8} {'TA-Lib':>8} {'Wilder':>8} {'SMA':>8} {'Δ(T-W)':>8} {'Δ(T-S)':>8} {'Chartink?':>10}"
    )
    print("-" * 80)

    for sym in target_symbols:
        prices = symbols_data[sym]["Close"]
        close_val = prices.iloc[-1]

        rsi_tl = rsi_talib(prices, 14).iloc[-1]
        rsi_w = rsi_wilder(prices, 14).iloc[-1]
        rsi_s = rsi_sma(prices, 14).iloc[-1]

        diff_tw = rsi_tl - rsi_w
        diff_ts = rsi_tl - rsi_s

        # Check if RSI(14) > 55 would differ between methods
        chartink_note = ""
        if (rsi_tl > 55) != (rsi_w > 55):
            chartink_note = "FLIP!"
        elif (rsi_tl > 55) != (rsi_s > 55):
            chartink_note = "FLIP!"

        print(
            f"{sym:<15} {close_val:>8.2f} {rsi_tl:>8.2f} {rsi_w:>8.2f} {rsi_s:>8.2f} {diff_tw:>+8.3f} {diff_ts:>+8.3f} {chartink_note:>10}"
        )

    # ── RSI(9) COMPARISON ──
    print("\n" + "=" * 80)
    print("RSI(9) CALCULATION COMPARISON — Latest Date")
    print("=" * 80)
    print(f"\n{'Symbol':<15} {'Close':>8} {'TA-Lib':>8} {'Wilder':>8} {'SMA':>8} {'Δ(T-W)':>8} {'Δ(T-S)':>8}")
    print("-" * 80)

    for sym in target_symbols:
        prices = symbols_data[sym]["Close"]
        close_val = prices.iloc[-1]

        rsi_tl = rsi_talib(prices, 9).iloc[-1]
        rsi_w = rsi_wilder(prices, 9).iloc[-1]
        rsi_s = rsi_sma(prices, 9).iloc[-1]

        diff_tw = rsi_tl - rsi_w
        diff_ts = rsi_tl - rsi_s

        print(
            f"{sym:<15} {close_val:>8.2f} {rsi_tl:>8.2f} {rsi_w:>8.2f} {rsi_s:>8.2f} {diff_tw:>+8.3f} {diff_ts:>+8.3f}"
        )

    # ── SMA COMPARISON ──
    print("\n" + "=" * 80)
    print("SMA CALCULATION COMPARISON — Latest Date")
    print("=" * 80)

    for period in [20, 50, 150, 200]:
        print(f"\n--- SMA({period}) ---")
        print(f"{'Symbol':<15} {'Close':>8} {'SMA(TL)':>10} {'SMA(PD)':>10} {'Δ':>8} {'Flip?':>8}")
        print("-" * 65)

        flips = 0
        for sym in target_symbols:
            prices = symbols_data[sym]["Close"]
            close_val = prices.iloc[-1]

            import talib

            sma_tl = talib.SMA(prices.values.astype(float), timeperiod=period)
            sma_tl_val = sma_tl[-1]
            sma_pd = prices.rolling(period).mean().iloc[-1]

            diff = sma_tl_val - sma_pd

            flip = ""
            if (close_val > sma_tl_val) != (close_val > sma_pd):
                flip = "FLIP!"
                flips += 1

            print(f"{sym:<15} {close_val:>8.2f} {sma_tl_val:>10.4f} {sma_pd:>10.4f} {diff:>+8.4f} {flip:>8}")

        print(f"  Flips: {flips}/{len(target_symbols)} (close>SMA differs between TA-Lib and pandas)")

    # ── VOLUME COMPARISON ──
    print("\n" + "=" * 80)
    print("VOLUME DATA — Latest Date")
    print("=" * 80)
    print(f"\n{'Symbol':<15} {'Volume':>12} {'20d Avg':>12} {'Ratio':>8} {'>100k':>8} {'>300k':>8}")
    print("-" * 70)

    for sym in target_symbols:
        vol = symbols_data[sym]["Volume"]
        latest_vol = vol.iloc[-1]
        avg_20d = vol.rolling(20).mean().iloc[-1]
        ratio = latest_vol / avg_20d if avg_20d > 0 else 0

        print(
            f"{sym:<15} {latest_vol:>12,.0f} {avg_20d:>12,.0f} {ratio:>8.2f} {'Y' if latest_vol > 100000 else 'N':>8} {'Y' if latest_vol > 300000 else 'N':>8}"
        )

    # ── SUMMARY ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings:
1. TA-Lib RSI and Wilder's RSI should be nearly identical (both use Wilder's smoothing)
2. SMA-based RSI differs significantly from TA-Lib/Wilder's RSI
3. TA-Lib SMA and pandas SMA should be identical
4. If Chartink uses SMA-based RSI, that explains the RSI filter mismatches
5. Volume data differences come from different data sources (yfinance vs NSE)
6. Cache staleness (yesterday's data vs today's) affects ALL indicator-based filters
""")


if __name__ == "__main__":
    main()
