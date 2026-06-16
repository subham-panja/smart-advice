import json
import logging
import os
import time
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf

from config import MAX_RETRIES, NSE_CACHE_FILE, RATE_LIMIT_DELAY, REQUEST_DELAY

logger = logging.getLogger(__name__)

# Use a relative data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "historical")
os.makedirs(DATA_DIR, exist_ok=True)


def get_all_nse_symbols() -> Dict[str, str]:
    """Returns a dictionary of all NSE symbols strictly."""
    if not os.path.exists(NSE_CACHE_FILE):
        raise FileNotFoundError(f"NSE symbol cache missing: {NSE_CACHE_FILE}")

    try:
        with open(NSE_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load NSE symbols from {NSE_CACHE_FILE}: {e}")
        raise e


def refresh_nse_symbols() -> Dict[str, str]:
    """Fetch current NSE equity symbols and merge with existing cache.

    Rolling update: adds new symbols, keeps all existing ones, never deletes.
    Tries NSE India CSV archive first, falls back to yfinance ticker discovery.
    """
    import io

    import requests

    existing = {}
    if os.path.exists(NSE_CACHE_FILE):
        try:
            with open(NSE_CACHE_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    fresh_symbols = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    csv_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    try:
        resp = requests.get(csv_url, headers=headers, timeout=30)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            sym_col = None
            for col in df.columns:
                if "symbol" in col.lower():
                    sym_col = col
                    break
            if sym_col is None:
                sym_col = df.columns[0]

            for raw_sym in df[sym_col].dropna():
                sym = str(raw_sym).strip().upper()
                if sym and sym != "NAN" and not sym.startswith(" "):
                    fresh_symbols[sym] = sym
            logger.info(f"Fetched {len(fresh_symbols)} symbols from NSE CSV archive")
    except Exception as e:
        logger.warning(f"NSE CSV archive failed: {e}")

    if not fresh_symbols:
        logger.info("Falling back to yfinance NSE symbol discovery...")
        known_symbols = [
            "RELIANCE",
            "TCS",
            "HDFCBANK",
            "INFY",
            "ICICIBANK",
            "SBIN",
            "BHARTIARTL",
            "ITC",
            "LT",
            "AXISBANK",
            "HINDUNILVR",
            "KOTAKBANK",
            "MARUTI",
            "SUNPHARMA",
            "TITAN",
            "BAJFINANCE",
            "WIPRO",
            "HCLTECH",
            "ADANIPORTS",
            "JSWSTEEL",
            "TATASTEEL",
            "POWERGRID",
            "NTPC",
            "ONGC",
            "COALINDIA",
            "ULTRACEMCO",
            "ASIANPAINT",
            "NESTLEIND",
        ]
        for sym in known_symbols:
            try:
                t = yf.Ticker(f"{sym}.NS")
                info = t.fast_info
                if info and hasattr(info, "last_price") and info.last_price and info.last_price > 0:
                    fresh_symbols[sym] = sym
            except Exception:
                pass
        if fresh_symbols:
            logger.info(f"Verified {len(fresh_symbols)} symbols via yfinance")

    if not fresh_symbols:
        logger.warning("Could not fetch fresh NSE symbols. Using existing cache as-is.")
        return existing

    merged = dict(existing)
    new_count = 0
    for sym in fresh_symbols:
        if sym not in merged:
            merged[sym] = sym
            new_count += 1

    try:
        with open(NSE_CACHE_FILE, "w") as f:
            json.dump(merged, f, sort_keys=True, indent=2)
        logger.info(f"NSE symbols refreshed: {len(merged)} total ({new_count} new, {len(existing)} existing)")
    except Exception as e:
        logger.error(f"Failed to save NSE symbols to {NSE_CACHE_FILE}: {e}")

    return merged


def get_historical_data(symbol: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical OHLCV data with rolling incremental cache.

    On first call: fetches full history from yfinance and caches to disk.
    On subsequent calls: reads cache, fetches only missing recent days, appends.
    """
    cache_path = os.path.join(DATA_DIR, f"{symbol}_{period}_{interval}.csv")

    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if cached_df.index.tz is not None:
                cached_df.index = cached_df.index.tz_localize(None)
            if not cached_df.empty:
                last_cached = cached_df.index[-1]
                today = pd.Timestamp.today().normalize()

                if last_cached.normalize() >= today:
                    return cached_df

                next_day = last_cached.normalize() + pd.Timedelta(days=1)
                if np.busday_count(next_day.date(), (today + pd.Timedelta(days=1)).date()) == 0:
                    return cached_df

                fetch_start = last_cached.strftime("%Y-%m-%d")
                yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol
                time.sleep(REQUEST_DELAY)

                try:
                    new_df = yf.Ticker(yf_sym).history(start=fetch_start, interval=interval)
                    if not new_df.empty:
                        new_df = new_df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                        new_df.index = pd.to_datetime(new_df.index)
                        if new_df.index.tz is not None:
                            new_df.index = new_df.index.tz_localize(None)

                        new_rows = new_df[new_df.index > last_cached]
                        if not new_rows.empty:
                            updated = pd.concat([cached_df, new_rows])
                            updated.to_csv(cache_path)
                            logger.info(f"📈 {symbol}: appended {len(new_rows)} new day(s) to cache")
                            return updated
                except Exception as update_err:
                    logger.warning(f"Incremental update failed for {symbol}: {update_err}, using existing cache")

                return cached_df
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}. Fetching fresh.")

    time.sleep(REQUEST_DELAY)

    attempts = 0
    last_error = None

    while attempts <= MAX_RETRIES:
        try:
            yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol
            logger.info(f"🔄 Fetching data for {symbol} (Attempt {attempts + 1})...")

            df = yf.Ticker(yf_sym).history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No historical data returned for {symbol}")

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if df.empty:
                raise ValueError(f"Data for {symbol} became empty after dropping NaNs.")

            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.to_csv(cache_path)
            return df

        except Exception as e:
            last_error = e
            attempts += 1
            if attempts <= MAX_RETRIES:
                logger.warning(f"Fetch failed for {symbol}: {e}. Retrying in {RATE_LIMIT_DELAY}s...")
                time.sleep(RATE_LIMIT_DELAY)
            else:
                logger.error(f"Critical fetch failure for {symbol} after {attempts} attempts: {e}")
                raise last_error


def get_current_price(symbol: str) -> float:
    """Gets latest price strictly."""
    yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol
    ticker = yf.Ticker(yf_sym)
    price = ticker.info.get("regularMarketPrice") or ticker.info.get("previousClose")
    if price is None:
        raise ValueError(f"Could not retrieve current price for {symbol}")
    return float(price)


def get_benchmark_data(period: str = "1y") -> pd.DataFrame:
    """Strictly fetches benchmark (Nifty) data."""
    return get_historical_data("^NSEI", period=period)


MARKET_CAP_CACHE = os.path.join(DATA_DIR, "..", "market_cap_cache.json")


def get_market_caps(symbols: list = None, min_cap_cr: float = 0) -> dict:
    """Fetch and cache market caps (in Crores) for NSE symbols from yfinance.

    Returns dict {symbol: market_cap_cr}. Only fetches uncached symbols.
    Skips symbols that fail or have no market cap data.
    """
    cache = {}
    if os.path.exists(MARKET_CAP_CACHE):
        try:
            with open(MARKET_CAP_CACHE, "r") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if symbols is None:
        all_syms = get_all_nse_symbols()
        symbols = list(all_syms.keys()) if isinstance(all_syms, dict) else list(all_syms)

    uncached = [s for s in symbols if s not in cache]
    if not uncached:
        return cache

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_mc(sym):
        try:
            yf_sym = f"{sym}.NS"
            t = yf.Ticker(yf_sym)
            mc = t.fast_info.get("market_cap", None)
            if mc is None:
                mc = t.info.get("marketCap", None)
            if mc and mc > 0:
                return sym, mc / 10000000
        except Exception:
            pass
        return sym, None

    fetched = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch_mc, s): s for s in uncached}
        for future in as_completed(futures):
            sym, cap_cr = future.result()
            if cap_cr is not None:
                cache[sym] = round(cap_cr, 2)
                fetched += 1

    try:
        with open(MARKET_CAP_CACHE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

    return cache
