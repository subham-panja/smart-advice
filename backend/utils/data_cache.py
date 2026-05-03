"""
Data Cache Manager
==================

Handles caching of historical OHLCV data in parquet format with staleness checking.
Used by portfolio backtesting to avoid redundant API downloads.

Cache location: backend/data/historical/
Cache format: {symbol}_{period}.parquet
Staleness: configurable in strategy JSON data_config.staleness_hours (default 24h)
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# Cache directory from config
CACHE_DIR = config.DATA_CACHE_CONFIG.get("cache_dir", os.path.join(config.BACKEND_DIR, "data", "historical"))
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_ENABLED = config.DATA_CACHE_CONFIG.get("enabled", True)
DEFAULT_STALENESS = config.DATA_CACHE_CONFIG.get("staleness_hours", 24)


def _cache_path(symbol: str, period: str) -> str:
    """Generate cache file path for a symbol + period combination."""
    clean_period = period.replace("y", "Y").replace("m", "M")
    return os.path.join(CACHE_DIR, f"{symbol}_{clean_period}.parquet")


def _is_fresh(cache_path: str, staleness_hours: int = 24) -> bool:
    """Check if cached data is still fresh (not older than staleness_hours)."""
    if not os.path.exists(cache_path):
        return False
    mtime = os.path.getmtime(cache_path)
    age_hours = (time.time() - mtime) / 3600
    return age_hours < staleness_hours


def fetch_historical_data_cached(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    fresh: bool = False,
    staleness_hours: int = None,
) -> pd.DataFrame:
    """Fetch historical OHLCV data with parquet caching and staleness check."""
    cache_path = _cache_path(symbol, period)

    # Try to load from cache
    if not fresh and _is_fresh(cache_path, staleness_hours):
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                logger.debug(f"Cache hit for {symbol} ({period})")
                return df
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}. Fetching fresh.")

    # Fetch fresh data
    logger.info(f"Fetching fresh data for {symbol} ({period})...")
    time.sleep(config.REQUEST_DELAY)

    attempts = 0
    last_error = None

    while attempts <= config.MAX_RETRIES:
        try:
            yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol
            df = yf.Ticker(yf_sym).history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No historical data returned for {symbol}")

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if df.empty:
                raise ValueError(f"Data for {symbol} became empty after dropping NaNs.")

            df.index = pd.to_datetime(df.index)

            # Save to parquet cache
            try:
                df.to_parquet(cache_path, index=True)
                logger.debug(f"Cached {symbol} ({period}) -> {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache {symbol}: {e}")

            return df

        except Exception as e:
            last_error = e
            attempts += 1
            if attempts <= config.MAX_RETRIES:
                logger.warning(f"Fetch failed for {symbol}: {e}. Retrying in {config.RATE_LIMIT_DELAY}s...")
                time.sleep(config.RATE_LIMIT_DELAY)
            else:
                logger.error(f"Critical fetch failure for {symbol} after {attempts} attempts: {e}")
                raise last_error


def fetch_multiple_symbols_cached(
    symbols: Dict[str, str],
    period: str = "5y",
    staleness_hours: int = None,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for multiple symbols with caching."""
    data = {}
    total = len(symbols)
    logger.info(f"Fetching data for {total} symbols (period={period})...")

    with ThreadPoolExecutor(max_workers=config.DATA_FETCH_THREADS) as executor:
        future_to_sym = {
            executor.submit(
                fetch_historical_data_cached, sym, period=period, staleness_hours=staleness_hours or DEFAULT_STALENESS
            ): sym
            for sym in symbols.keys()
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


def clear_cache(older_than_hours: Optional[int] = None) -> int:
    """Clear cached parquet files."""
    if not os.path.exists(CACHE_DIR):
        return 0

    count = 0
    now = time.time()

    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".parquet"):
            continue

        fpath = os.path.join(CACHE_DIR, fname)
        if older_than_hours is None:
            os.remove(fpath)
            count += 1
        else:
            age_hours = (now - os.path.getmtime(fpath)) / 3600
            if age_hours > older_than_hours:
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

    symbols = []
    for f in files:
        parts = f.rsplit("_", 1)
        if len(parts) == 2:
            symbols.append(parts[0])

    return {
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "symbols": sorted(set(symbols)),
    }
