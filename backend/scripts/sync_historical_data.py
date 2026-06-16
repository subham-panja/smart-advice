import logging
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concurrent.futures import ThreadPoolExecutor, as_completed

from config import DATA_FETCH_THREADS
from scripts.data_fetcher import get_all_nse_symbols, refresh_nse_symbols
from utils.data_cache import fetch_historical_data_cached, get_cache_stats

DEFAULT_HISTORICAL_PERIOD = "5y"

logger = logging.getLogger(__name__)


def sync_all_historical_data():
    """Sync historical data for all NSE symbols into the parquet cache.

    Uses the same parquet cache (data_cache.py) that the backtest and
    orchestrator read from. Rolling incremental: keeps all old data,
    fetches only missing recent days, appends.
    """
    start_time = datetime.now()
    logger.info(f"Starting historical data sync for all NSE symbols (Period: {DEFAULT_HISTORICAL_PERIOD})")

    logger.info("Refreshing NSE symbol list...")
    refresh_nse_symbols()

    symbols_dict = get_all_nse_symbols()
    if not symbols_dict:
        logger.error("Could not fetch NSE symbols. Exiting.")
        return

    symbols = list(symbols_dict.keys())
    total_symbols = len(symbols)
    logger.info(f"Found {total_symbols} symbols to sync.")

    success_count = 0
    fail_count = 0

    max_workers = min(DATA_FETCH_THREADS, 5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(fetch_historical_data_cached, symbol, period=DEFAULT_HISTORICAL_PERIOD): symbol
            for symbol in symbols
        }

        for i, future in enumerate(as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data is not None and not data.empty:
                    success_count += 1
                    if i % 50 == 0 or i == total_symbols - 1:
                        logger.info(f"Progress: {i+1}/{total_symbols} | Synced {symbol} ({len(data)} bars)")
                else:
                    fail_count += 1
                    logger.warning(f"Failed to sync {symbol}: Empty data returned")
            except Exception as e:
                fail_count += 1
                logger.error(f"Error syncing {symbol}: {e}")

    end_time = datetime.now()
    duration = end_time - start_time

    stats = get_cache_stats()

    logger.info("=" * 50)
    logger.info(f"Sync Completed at {end_time}")
    logger.info(f"Total Duration: {duration}")
    logger.info(f"Successfully Synced: {success_count}")
    logger.info(f"Failed: {fail_count}")
    logger.info(f"Cache: {stats['total_files']} files, {stats['total_size_mb']:.1f} MB")
    logger.info("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sync_all_historical_data()
