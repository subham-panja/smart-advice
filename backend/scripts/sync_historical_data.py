import logging
import os
import sys
import time
from datetime import datetime

# Add the parent directory to sys.path to allow imports from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_FETCH_THREADS
from scripts.data_fetcher import get_all_nse_symbols, get_historical_data

DEFAULT_HISTORICAL_PERIOD = "5y"

logger = logging.getLogger(__name__)


def sync_all_historical_data():
    """
    Syncs historical data for all NSE symbols.
    If data exists in cache, it will fetch only the delta.
    If not, it will fetch the full DEFAULT_HISTORICAL_PERIOD.
    """
    start_time = datetime.now()
    logger.info(f"Starting historical data sync for all NSE symbols (Period: {DEFAULT_HISTORICAL_PERIOD})")

    # 1. Get all symbols
    symbols_dict = get_all_nse_symbols()
    if not symbols_dict:
        logger.error("Could not fetch NSE symbols. Exiting.")
        return

    symbols = list(symbols_dict.keys())
    total_symbols = len(symbols)
    logger.info(f"Found {total_symbols} symbols to sync.")

    # 2. Process symbols in batches or using threads
    # Note: We use threads carefully to avoid rate limiting from yfinance
    success_count = 0
    fail_count = 0

    # Use fewer threads for yfinance to be safe
    max_workers = min(DATA_FETCH_THREADS, 5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(get_historical_data, symbol, period=DEFAULT_HISTORICAL_PERIOD): symbol for symbol in symbols
        }

        for i, future in enumerate(as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if not data.empty:
                    success_count += 1
                    if i % 10 == 0 or i == total_symbols - 1:
                        logger.info(f"Progress: {i+1}/{total_symbols} | Synced {symbol}")
                else:
                    fail_count += 1
                    logger.warning(f"Failed to sync {symbol}: Empty data returned")
            except Exception as e:
                fail_count += 1
                logger.error(f"Error syncing {symbol}: {e}")

            # Add a small delay to avoid hitting rate limits too hard
            time.sleep(0.5)

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 50)
    logger.info(f"Sync Completed at {end_time}")
    logger.info(f"Total Duration: {duration}")
    logger.info(f"Successfully Synced: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info("=" * 50)


if __name__ == "__main__":
    sync_all_historical_data()
