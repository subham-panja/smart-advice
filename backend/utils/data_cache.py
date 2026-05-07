"""
Data Cache Manager
==================

Handles caching of historical OHLCV data in parquet format with incremental/delta fetching.
Historical data never changes, so the cache never expires. When a longer period is requested,
only the missing data (delta) is fetched and merged with the existing cache.

Cache location: backend/data/historical/
Cache format: {symbol}.parquet (one file per symbol, grows over time)
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# Cache directory from config
CACHE_DIR = config.DATA_CACHE_CONFIG.get("cache_dir", os.path.join(config.BACKEND_DIR, "data", "historical"))
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_ENABLED = config.DATA_CACHE_CONFIG.get("enabled", True)


def _cache_path(symbol: str) -> str:
    """Generate cache file path for a symbol (one file per symbol, no period in key)."""
    return os.path.join(CACHE_DIR, f"{symbol}.parquet")


def _period_to_start_date(period: str, reference_date: pd.Timestamp = None) -> pd.Timestamp:
    """Convert a yfinance period string to a start date."""
    if reference_date is None:
        reference_date = pd.Timestamp.now(tz="Asia/Kolkata")

    if period == "max" or period == "ytd":
        return pd.Timestamp("1990-01-01", tz="Asia/Kolkata")

    period_map = {
        "1d": pd.Timedelta(days=1),
        "5d": pd.Timedelta(days=5),
        "1mo": pd.DateOffset(months=1),
        "3mo": pd.DateOffset(months=3),
        "6mo": pd.DateOffset(months=6),
        "1y": pd.DateOffset(years=1),
        "2y": pd.DateOffset(years=2),
        "3y": pd.DateOffset(years=3),
        "4y": pd.DateOffset(years=4),
        "5y": pd.DateOffset(years=5),
        "10y": pd.DateOffset(years=10),
        "20y": pd.DateOffset(years=20),
    }

    offset = period_map.get(period.lower())
    if offset is None:
        raise ValueError(f"Unknown period: {period}")

    return reference_date - offset


def _fetch_with_retry(
    yf_sym: str, start: pd.Timestamp = None, end: pd.Timestamp = None, period: str = None, interval: str = "1d"
):
    """Fetch from yfinance with retry logic."""
    attempts = 0
    last_error = None

    while attempts <= config.MAX_RETRIES:
        try:
            if period:
                return yf.Ticker(yf_sym).history(period=period, interval=interval)
            else:
                kwargs = {"interval": interval}
                if start:
                    kwargs["start"] = start.strftime("%Y-%m-%d")
                if end:
                    kwargs["end"] = end.strftime("%Y-%m-%d")
                return yf.Ticker(yf_sym).history(**kwargs)
        except Exception as e:
            last_error = e
            attempts += 1
            if attempts <= config.MAX_RETRIES:
                logger.warning(f"Fetch failed for {yf_sym}: {e}. Retrying in {config.RATE_LIMIT_DELAY}s...")
                time.sleep(config.RATE_LIMIT_DELAY)
            else:
                logger.error(f"Critical fetch failure for {yf_sym} after {attempts} attempts: {e}")
                raise last_error


def fetch_historical_data_cached(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch historical OHLCV data with incremental caching.

    If cache exists and covers the requested period, returns cached data (no API call).
    If cache exists but doesn't cover the full period, fetches only the delta and merges.
    If no cache exists, fetches the full period and caches it.
    """
    cache_path = _cache_path(symbol)
    yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol

    requested_start = _period_to_start_date(period)

    # Load existing cache if available
    cached_df = None
    if not force_refresh and os.path.exists(cache_path):
        try:
            cached_df = pd.read_parquet(cache_path)
            if cached_df.empty:
                cached_df = None
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}. Fetching fresh.")
            cached_df = None

    # Determine if we need to fetch more data
    if cached_df is not None:
        cached_oldest = cached_df.index[0]

        if cached_oldest <= requested_start:
            # Cache fully covers requested range — just trim and return
            logger.debug(f"Cache covers full range for {symbol} ({period})")
            df = cached_df
        else:
            # Need delta: fetch from requested_start to just before cached_oldest
            delta_start = requested_start
            delta_end = cached_oldest - pd.Timedelta(days=1)

            logger.info(
                f"Delta fetch for {symbol}: {delta_start.strftime('%Y-%m-%d')} to {delta_end.strftime('%Y-%m-%d')}"
            )
            time.sleep(config.REQUEST_DELAY)

            delta_df = _fetch_with_retry(
                yf_sym, start=delta_start, end=delta_end + pd.Timedelta(days=1), interval=interval
            )

            if delta_df is not None and not delta_df.empty:
                delta_df = delta_df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                delta_df.index = pd.to_datetime(delta_df.index)

                # Merge: new data + cached data, deduplicate keeping newest
                df = pd.concat([delta_df, cached_df])
                df = df[~df.index.duplicated(keep="last")]
                df = df.sort_index()
            else:
                # Delta fetch returned empty — use cached data as-is
                logger.warning(f"Delta fetch returned no data for {symbol}, using cached data")
                df = cached_df
    else:
        # No cache, full fetch using period (yfinance returns whatever data is available)
        logger.info(f"Fetching full data for {symbol} ({period})...")
        time.sleep(config.REQUEST_DELAY)

        df = _fetch_with_retry(yf_sym, period=period, interval=interval)

        if df is None or df.empty:
            raise ValueError(f"No historical data returned for {symbol}")

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.empty:
            raise ValueError(f"Data for {symbol} became empty after dropping NaNs.")

        df.index = pd.to_datetime(df.index)

    if df.empty:
        raise ValueError(f"No historical data returned for {symbol}")

    # Save full merged data to cache (always preserve maximum available history)
    try:
        df.to_parquet(cache_path, index=True)
        logger.debug(f"Cached {symbol} -> {cache_path} ({len(df)} rows)")
    except Exception as e:
        logger.warning(f"Failed to cache {symbol}: {e}")

    # Trim to requested period for return
    if requested_start > df.index[0]:
        df = df[df.index >= requested_start]

    return df


def fetch_multiple_symbols_cached(
    symbols: Dict[str, str],
    period: str = "5y",
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for multiple symbols with incremental caching."""
    data = {}
    total = len(symbols)
    logger.info(f"Fetching data for {total} symbols (period={period})...")

    with ThreadPoolExecutor(max_workers=config.DATA_FETCH_THREADS) as executor:
        future_to_sym = {
            executor.submit(fetch_historical_data_cached, sym, period=period): sym for sym in symbols.keys()
        }

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


def clear_cache(symbols: List[str] = None) -> int:
    """Clear cached parquet files. If symbols provided, clear only those."""
    if not os.path.exists(CACHE_DIR):
        return 0

    count = 0
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".parquet"):
            continue

        fpath = os.path.join(CACHE_DIR, fname)
        if symbols is None:
            os.remove(fpath)
            count += 1
        else:
            sym = fname.replace(".parquet", "")
            if sym in symbols:
                os.remove(fpath)
                count += 1

    logger.info(f"Cleared {count} cache files from {CACHE_DIR}")
    return count


def get_cache_stats() -> Dict:
    """Get cache statistics."""
    if not os.path.exists(CACHE_DIR):
        return {"total_files": 0, "total_size_mb": 0, "symbols": []}

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
    total_size = sum(os.path.getsize(os.path.join(CACHE_DIR, f)) for f in files)
    symbols = [f.replace(".parquet", "") for f in files]

    return {
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "symbols": sorted(set(symbols)),
    }


def migrate_old_cache_files() -> int:
    """Migrate old {symbol}_{PERIOD}.parquet files into unified {symbol}.parquet files.

    Old format: RELIANCE_3Y.parquet, RELIANCE_5Y.parquet, etc.
    New format: RELIANCE.parquet (contains the maximum period ever fetched)

    Returns the number of old files processed.
    """
    if not os.path.exists(CACHE_DIR):
        return 0

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]

    # Group by symbol
    symbol_files = {}
    for f in files:
        parts = f.rsplit("_", 1)
        if len(parts) == 2:
            period_part = parts[1].replace(".parquet", "")
            # Check if it looks like a period (e.g., 1Y, 2Y, 3Y, 5Y, 10Y, 20Y, 1Mo)
            if period_part[:-1].isdigit() and period_part[-1] in ("Y", "M", "D"):
                symbol = parts[0]
                symbol_files.setdefault(symbol, []).append(f)

    migrated = 0
    for symbol, old_files in symbol_files.items():
        new_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")

        # Skip if already migrated (new format exists)
        if os.path.exists(new_path):
            # Delete old files anyway
            for f in old_files:
                os.remove(os.path.join(CACHE_DIR, f))
                migrated += 1
            continue

        # Merge all old files for this symbol, keeping the one with the most data
        best_df = None
        for f in old_files:
            try:
                df = pd.read_parquet(os.path.join(CACHE_DIR, f))
                if best_df is None or len(df) > len(best_df):
                    best_df = df
            except Exception:
                continue

        if best_df is not None and not best_df.empty:
            best_df.to_parquet(new_path, index=True)
            logger.info(f"Migrated {len(old_files)} old files for {symbol} -> {new_path} ({len(best_df)} rows)")

        # Remove old files
        for f in old_files:
            os.remove(os.path.join(CACHE_DIR, f))
            migrated += 1

    logger.info(f"Migration complete: {migrated} old files processed")
    return migrated
