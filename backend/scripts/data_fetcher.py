import pandas as pd
import json
import os
import time
import logging
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import NSE_CACHE_FILE, DATA_FETCH_THREADS, HISTORICAL_DATA_PERIOD

logger = logging.getLogger(__name__)

def get_all_nse_symbols() -> Dict[str, str]:
    """Returns a dictionary of all NSE symbols."""
    if os.path.exists(NSE_CACHE_FILE):
        try:
            with open(NSE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except: pass
    return {"RELIANCE": "Reliance Industries", "TCS": "TCS"}

def get_historical_data(symbol: str, period: str = '1y', interval: str = '1d', fresh: bool = False) -> pd.DataFrame:
    """Fetches historical data with local CSV caching."""
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}_{period}_{interval}.csv")

    if not fresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty and (time.time() - os.path.getmtime(cache_path)) < 43200:
                return df
        except: pass

    try:
        yf_sym = f"{symbol}.NS" if not symbol.startswith('^') else symbol
        # Use a single ticker download to avoid MultiIndex issues
        df = yf.Ticker(yf_sym).history(period=period, interval=interval)
        if df.empty: return pd.DataFrame()
        
        # history() returns standard columns. Let's ensure they are named correctly.
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        df.index = pd.to_datetime(df.index)
        df.to_csv(cache_path)
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def get_current_price(symbol: str) -> Optional[float]:
    """Gets the latest market price."""
    try:
        df = get_historical_data(symbol, period='5d')
        return float(df['Close'].iloc[-1]) if not df.empty else None
    except: return None

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

def get_benchmark_data(period: str = '1y') -> pd.DataFrame:
    return get_historical_data('^NSEI', period=period)
