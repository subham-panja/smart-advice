"""
Data Cache Manager
==================

Handles caching of historical OHLCV data in parquet format.
Each symbol has one cache file (e.g. RELIANCE.parquet) that gets updated daily.
If cache exists and is recent (within last trading day), data is returned from cache.
Otherwise, fresh data is fetched and the cache file is updated.

Cache location: backend/data/historical/
Cache format: {symbol}.parquet
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

import config
from utils.trading_clock import is_replay, trading_now

logger = logging.getLogger(__name__)

CACHE_DIR = config.DATA_CACHE_CONFIG.get("cache_dir", os.path.join(config.BACKEND_DIR, "data", "historical"))
os.makedirs(CACHE_DIR, exist_ok=True)

DELISTED_CACHE_FILE = os.path.join(config.BACKEND_DIR, "data", "delisted_symbols.json")
_delisted_symbols = set()
if os.path.exists(DELISTED_CACHE_FILE):
    try:
        with open(DELISTED_CACHE_FILE, "r") as f:
            _delisted_symbols = set(json.load(f))
    except Exception:
        pass


def mark_symbol_delisted(symbol: str):
    """Mark a symbol as delisted/inactive to prevent repeated network queries."""
    _delisted_symbols.add(symbol)
    try:
        with open(DELISTED_CACHE_FILE, "w") as f:
            json.dump(sorted(list(_delisted_symbols)), f, indent=2)
    except Exception:
        pass


CACHE_ENABLED = config.DATA_CACHE_CONFIG.get("enabled", True)

_PERIOD_APPROX_ROWS = {
    "1d": 1,
    "5d": 5,
    "1mo": 22,
    "3mo": 65,
    "6mo": 130,
    "1y": 252,
    "2y": 504,
    "5y": 1260,
    "10y": 2520,
    "max": 5000,
}


def _min_rows_for_period(period: str) -> int:
    """Return minimum trading-day rows expected for a yfinance period string.

    Handles standard periods (1y, 2y, 5y, etc.) and day-based strings (60d, 30d).
    Applies 85% tolerance for holidays, IPOs, and market closures.
    """
    if period in _PERIOD_APPROX_ROWS:
        return int(_PERIOD_APPROX_ROWS[period] * 0.85)
    # Handle yfinance-style day/week/month strings like "60d", "2wk", "3mo"
    import re as _re

    m = _re.match(r"^(\d+)([dwmo])$", period)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        day_map = {"d": 1, "w": 5, "m": 22, "o": 22}  # approximate trading days
        return int(n * day_map.get(unit, 1) * 0.7)
    return 0  # unknown period → accept any cached data


def _cache_path(symbol: str, interval: str = "1d") -> str:
    """Return cache path for a symbol (separated by interval if not daily)."""
    if interval and interval != "1d":
        return os.path.join(CACHE_DIR, f"{symbol}_{interval}.parquet")
    return os.path.join(CACHE_DIR, f"{symbol}.parquet")


_last_trading_day_cache = {"date": None, "fetched_at": None}


def _get_last_trading_day() -> Optional[date]:
    """Determine the actual last trading day using NIFTY 50 index data.

    Handles weekends AND Indian market holidays correctly:
    - Normal weekend (Sat/Sun): returns Friday
    - Holiday on Friday: returns Thursday
    - Holiday week: returns the most recent day NIFTY actually traded

    Falls back to weekday-based estimation if ^NSEI cache is unavailable or stale.
    """
    today = trading_now().date()
    # Reuse cached result within the same session
    if _last_trading_day_cache["date"] and _last_trading_day_cache["fetched_at"] == today:
        return _last_trading_day_cache["date"]

    nsei_path = _cache_path("^NSEI")
    if os.path.exists(nsei_path):
        try:
            df = pd.read_parquet(nsei_path)
            if not df.empty:
                last_date = df.index[-1]
                if hasattr(last_date, "date"):
                    last_date = last_date.date()
                elif isinstance(last_date, str):
                    last_date = pd.to_datetime(last_date).date()

                # Verify that ^NSEI cache itself is fresh relative to today
                max_gap = 4 if today.weekday() >= 5 else 2
                if (today - last_date).days <= max_gap:
                    _last_trading_day_cache["date"] = last_date
                    _last_trading_day_cache["fetched_at"] = today
                    return last_date
        except Exception:
            pass

    # If ^NSEI cache is missing or stale, fetch fresh ^NSEI from yfinance
    try:
        df = yf.Ticker("^NSEI").history(period="5d", interval="1d")
        if df is not None and not df.empty:
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            df.to_parquet(nsei_path, index=True)
            last_date = df.index[-1].date()
            _last_trading_day_cache["date"] = last_date
            _last_trading_day_cache["fetched_at"] = today
            return last_date
    except Exception as e:
        logger.warning(f"Failed to fetch ^NSEI to determine last trading day: {e}")

    # Fallback: estimate last trading day from weekday
    d = today
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _is_cache_recent(df: pd.DataFrame, max_age_days: int = 1) -> bool:
    """Check if cached data is recent enough.

    Uses actual last trading day from NIFTY 50 when available, which correctly
    handles weekends AND Indian market holidays. On a holiday Friday, Thursday's
    data is fresh. On a normal weekend, Friday's data is fresh.

    Falls back to weekday-based gap counting if NIFTY data is unavailable.
    """
    if df.empty:
        return False
    try:
        last_date = df.index[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()
        elif isinstance(last_date, str):
            last_date = pd.to_datetime(last_date).date()

        today = trading_now().date()

        # Hard safety check: If not in replay mode and cache is > 4 calendar days old, it's stale!
        if not is_replay():
            gap_days = (today - last_date).days
            if gap_days > 4:
                return False

        # Use actual last trading day from NIFTY 50 (handles holidays correctly)
        last_td = _get_last_trading_day()
        if last_td and last_td != today:
            return last_date >= last_td

        # Fallback: count trading days (weekdays only)
        trading_days_gap = 0
        d = last_date + timedelta(days=1)
        while d <= today:
            if d.weekday() < 5:
                trading_days_gap += 1
            d += timedelta(days=1)

        return trading_days_gap <= max_age_days
    except Exception:
        return False


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


def _sync_live_price(df: pd.DataFrame, yf_sym: str, symbol: str) -> pd.DataFrame:
    """Sync live market last_price from fast_info if today's history row has NaN Close."""
    if is_replay():
        return df.dropna(subset=["Close"])
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        t = yf.Ticker(yf_sym)
        fast_info = getattr(t, "fast_info", None)
        if fast_info:
            lp = getattr(fast_info, "last_price", None)
            if lp and not pd.isna(lp) and float(lp) > 0:
                today_date = trading_now().date()
                today_ts = pd.Timestamp(today_date)

                df_clean = df.dropna(how="all")
                today_mask = df_clean.index.date == today_date
                if today_mask.any():
                    idx_loc = df_clean.index[today_mask][0]
                    df_clean.loc[idx_loc, "Close"] = round(float(lp), 2)
                    op = getattr(fast_info, "open", None)
                    hi = getattr(fast_info, "day_high", None)
                    lo = getattr(fast_info, "day_low", None)
                    if op and not pd.isna(op):
                        df_clean.loc[idx_loc, "Open"] = round(float(op), 2)
                    if hi and not pd.isna(hi):
                        df_clean.loc[idx_loc, "High"] = round(float(hi), 2)
                    if lo and not pd.isna(lo):
                        df_clean.loc[idx_loc, "Low"] = round(float(lo), 2)
                else:
                    op = getattr(fast_info, "open", lp) or lp
                    hi = getattr(fast_info, "day_high", lp) or lp
                    lo = getattr(fast_info, "day_low", lp) or lp
                    new_row = pd.DataFrame(
                        [
                            {
                                "Open": round(float(op), 2),
                                "High": round(float(hi), 2),
                                "Low": round(float(lo), 2),
                                "Close": round(float(lp), 2),
                                "Volume": getattr(fast_info, "last_volume", 0) or 0,
                            }
                        ],
                        index=[today_ts],
                    )
                    df_clean = pd.concat([df_clean, new_row])
                df = df_clean
    except Exception as e:
        logger.debug(f"Live price sync failed for {symbol}: {e}")
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[cols].dropna(subset=["Close"])


def fetch_historical_data_cached(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
    min_rows: Optional[int] = None,
    sync_live: bool = False,
) -> pd.DataFrame:
    """Fetch historical OHLCV data with single-file caching."""
    if symbol in _delisted_symbols:
        return pd.DataFrame()

    cache_path = _cache_path(symbol, interval)

    yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol
    needed = min_rows if min_rows else _min_rows_for_period(period)

    # Check if cache exists and is usable
    if not force_refresh and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if not df.empty:
                is_recent = _is_cache_recent(df, max_age_days=1)
                has_enough_rows = len(df) >= needed if period != "max" else is_recent

                # If cache is recent and non-empty, use existing cache directly
                if is_recent and (has_enough_rows or len(df) >= 1):
                    # Sync live price for current day if cache row lacks today's close and explicitly requested
                    if sync_live and not is_replay() and df.index[-1].date() < trading_now().date():
                        df = _sync_live_price(df, yf_sym, symbol)
                    logger.debug(f"Cache hit for {symbol}: {len(df)} rows, recent data")
                    return df

                if is_recent and not has_enough_rows:
                    logger.info(
                        f"Cache for {symbol} has {len(df)} rows, need {needed} for {period}. Re-fetching longer period..."
                    )
                elif has_enough_rows:
                    logger.info(f"Cache for {symbol} has {len(df)} rows but outdated. Re-fetching...")
                else:
                    logger.info(f"Cache for {symbol} has {len(df)} rows, need {needed} for {period}. Re-fetching...")
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}. Fetching fresh.")

    # Fetch fresh data
    logger.info(f"Fetching {symbol} ({period})...")
    time.sleep(config.REQUEST_DELAY)

    try:
        df_fresh = _fetch_with_retry(yf_sym, period=period, interval=interval)
    except Exception as e:
        err_msg = str(e).lower()
        if "delisted" in err_msg or "404" in err_msg or "no data found" in err_msg:
            mark_symbol_delisted(symbol)
        # Fallback to existing cache if available rather than crashing
        if os.path.exists(cache_path):
            try:
                cached_df = pd.read_parquet(cache_path)
                if not cached_df.empty:
                    logger.warning(
                        f"Fetch failed for {symbol} ({e}), falling back to cached data ({len(cached_df)} rows)"
                    )
                    return cached_df
            except Exception:
                pass
        raise e

    if df_fresh is None or df_fresh.empty:
        mark_symbol_delisted(symbol)
        if os.path.exists(cache_path):
            try:
                cached_df = pd.read_parquet(cache_path)
                if not cached_df.empty:
                    return cached_df
            except Exception:
                pass
        raise ValueError(f"No historical data returned for {symbol}")

    df = _sync_live_price(df_fresh, yf_sym, symbol)
    if df.empty:
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        raise ValueError(f"Data for {symbol} became empty after live price sync.")

    df.index = pd.to_datetime(df.index)

    # Update cache with new data
    try:
        df.to_parquet(cache_path, index=True)
        logger.debug(f"Cached {symbol} -> {cache_path} ({len(df)} rows)")
    except Exception as e:
        logger.warning(f"Failed to cache {symbol}: {e}")

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
                    logger.debug(f"Insufficient data for {sym}: {len(df) if df is not None else 0} bars")
            except Exception as e:
                logger.debug(f"Failed to fetch {sym}: {e}")

    logger.info(f"Successfully fetched data for {len(data)}/{total} symbols")
    return data


def _extract_symbol(filename: str) -> str:
    """Extract symbol from parquet filename.

    Handles both old format (RELIANCE.parquet) and
    date-stamped format (RELIANCE_2026-06-16.parquet).
    """
    base = filename.replace(".parquet", "")
    match = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}$", base)
    if match:
        return match.group(1)
    return base


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
            sym = _extract_symbol(fname)
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

    # Count unique symbols (handle both old date-stamped and new format)
    symbols = set()
    recent_count = 0
    today = date.today()

    for f in files:
        sym = _extract_symbol(f)
        symbols.add(sym)

        # Check if file is recent
        fpath = os.path.join(CACHE_DIR, f)
        try:
            df = pd.read_parquet(fpath)
            if not df.empty:
                last_date = df.index[-1]
                if hasattr(last_date, "date"):
                    last_date = last_date.date()
                elif isinstance(last_date, str):
                    last_date = pd.to_datetime(last_date).date()

                if (today - last_date).days <= 3:  # ~1 trading day = up to 3 calendar days (weekend)
                    recent_count += 1
        except Exception:
            pass

    return {
        "total_files": len(files),
        "unique_symbols": len(symbols),
        "recent_cached": recent_count,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "symbols": sorted(symbols),
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


def migrate_date_stamped_files() -> int:
    """Migrate date-stamped files ({symbol}_{YYYY-MM-DD}.parquet) to single file format ({symbol}.parquet).

    For each symbol with multiple date-stamped files:
    1. Find the most recent file
    2. Save it as {symbol}.parquet
    3. Delete all date-stamped files for that symbol

    Returns the number of files deleted.
    """
    if not os.path.exists(CACHE_DIR):
        return 0

    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]

    # Group date-stamped files by symbol
    symbol_files = {}
    for f in files:
        base = f.replace(".parquet", "")
        match = re.match(r"^(.+?)_\d{4}-\d{2}-\d{2}$", base)
        if match:
            symbol = match.group(1)
            symbol_files.setdefault(symbol, []).append(f)

    deleted = 0
    for symbol, dated_files in symbol_files.items():
        new_path = os.path.join(CACHE_DIR, f"{symbol}.parquet")

        # Find the most recent date-stamped file
        most_recent_file = None
        most_recent_date = None

        for f in dated_files:
            match = re.search(r"_(\d{4}-\d{2}-\d{2})\.parquet$", f)
            if match:
                file_date = match.group(1)
                if most_recent_date is None or file_date > most_recent_date:
                    most_recent_date = file_date
                    most_recent_file = f

        if most_recent_file:
            # Copy most recent file to new format
            old_path = os.path.join(CACHE_DIR, most_recent_file)
            try:
                df = pd.read_parquet(old_path)
                if not df.empty:
                    df.to_parquet(new_path, index=True)
                    logger.info(f"Migrated {symbol}: {most_recent_file} -> {symbol}.parquet ({len(df)} rows)")
            except Exception as e:
                logger.warning(f"Failed to migrate {symbol}: {e}")
                continue

        # Delete all date-stamped files for this symbol
        for f in dated_files:
            try:
                os.remove(os.path.join(CACHE_DIR, f))
                deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {f}: {e}")

    logger.info(f"Date-stamped migration complete: {deleted} files deleted")
    return deleted


def consolidate_cache() -> Dict:
    """Run all cache consolidation operations.

    Returns summary of operations performed.
    """
    results = {
        "period_migration": 0,
        "date_stamped_migration": 0,
    }

    # First migrate old period-based files (RELIANCE_5Y.parquet)
    results["period_migration"] = migrate_old_cache_files()

    # Then migrate date-stamped files (RELIANCE_2026-06-18.parquet)
    results["date_stamped_migration"] = migrate_date_stamped_files()

    # Get final stats
    stats = get_cache_stats()
    results["final_files"] = stats["total_files"]
    results["final_symbols"] = stats["unique_symbols"]
    results["final_size_mb"] = stats["total_size_mb"]

    logger.info(f"Cache consolidation complete: {results}")
    return results
