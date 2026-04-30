import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from config import DATA_FETCH_THREADS, NSE_CACHE_FILE

logger = logging.getLogger(__name__)


def get_all_nse_symbols() -> Dict[str, str]:
    """Returns a dictionary of all NSE symbols."""
    if os.path.exists(NSE_CACHE_FILE):
        try:
            with open(NSE_CACHE_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {"RELIANCE": "Reliance Industries", "TCS": "TCS"}


def get_historical_data(symbol: str, period: str = "1y", interval: str = "1d", fresh: bool = False) -> pd.DataFrame:
    """Fetches historical data with smart cache invalidation for execution accuracy."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_{period}_{interval}.csv")

    # Smart Cache Check
    if not fresh and os.path.exists(cache_path):
        try:
            mtime = os.path.getmtime(cache_path)
            age_seconds = time.time() - mtime

            # Define TTL based on the data period
            # For backtesting (long periods), cache is valid for 24 hours
            if "y" in period or "mo" in period:
                cache_ttl = 86400  # 24 hours
            else:
                # During market hours, short periods expire in 2 minutes
                now = datetime.now()
                is_market_hours = (
                    now.weekday() < 5
                    and (now.hour > 9 or (now.hour == 9 and now.minute >= 15))
                    and (now.hour < 15 or (now.hour == 15 and now.minute <= 30))
                )
                cache_ttl = 120 if is_market_hours else 3600

            if age_seconds < cache_ttl:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                if not df.empty:
                    return df
        except:
            pass

    try:
        yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol
        logger.info(f"🔄 Fetching fresh data for {symbol}...")

        # We always fetch a bit more than requested to ensure indicator stability
        df = yf.Ticker(yf_sym).history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()

        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)

        # Save to cache but only if it's a significant amount of data
        if not df.empty:
            df.to_csv(cache_path)

        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()


def get_current_price(symbol: str) -> Optional[float]:
    """Gets the latest market price."""
    try:
        df = get_historical_data(symbol, period="5d")
        return float(df["Close"].iloc[-1]) if not df.empty else None
    except:
        return None


def get_current_price_batch(symbols: List[str]) -> Dict[str, Optional[float]]:
    """Fetches prices for multiple symbols in parallel."""
    results = {}
    with ThreadPoolExecutor(max_workers=DATA_FETCH_THREADS) as executor:
        future_to_sym = {executor.submit(get_current_price, s): s for s in symbols}
        for f in as_completed(future_to_sym):
            results[future_to_sym[f]] = f.result()
    return results


def get_filtered_nse_symbols(**kwargs):
    return get_all_nse_symbols()


def get_benchmark_data(period: str = "1y") -> pd.DataFrame:
    return get_historical_data("^NSEI", period=period)
