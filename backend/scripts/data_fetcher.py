import json
import logging
import os
import time
from typing import Dict

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


def get_historical_data(symbol: str, period: str = "2y", interval: str = "1d", fresh: bool = False) -> pd.DataFrame:
    """Fetches historical OHLCV data strictly from yfinance with caching and retries."""
    # Strict inter-request delay to prevent rate limiting
    time.sleep(REQUEST_DELAY)

    cache_path = os.path.join(DATA_DIR, f"{symbol}_{period}_{interval}.csv")

    if not fresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"Cache read error for {symbol}: {e}. Retrying with fresh fetch.")

    attempts = 0
    last_error = None

    while attempts <= MAX_RETRIES:
        try:
            yf_sym = f"{symbol}.NS" if not symbol.startswith("^") else symbol
            logger.info(f"🔄 Fetching fresh data for {symbol} (Attempt {attempts + 1})...")

            df = yf.Ticker(yf_sym).history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No historical data returned for {symbol}")

            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if df.empty:
                raise ValueError(f"Data for {symbol} became empty after dropping NaNs.")

            df.index = pd.to_datetime(df.index)
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
