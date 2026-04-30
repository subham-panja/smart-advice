#!/usr/bin/env python3
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import config
from scripts.data_fetcher import get_benchmark_data, get_historical_data
from scripts.market_regime_detection import MarketRegimeDetection
from utils.cache_manager import get_cache_manager
from utils.logger import setup_logging
from utils.parallel_worker import analyze_stock_worker, init_worker
from utils.persistence_handler import PersistenceHandler
from utils.stock_scanner import StockScanner

logger = None


class AutomatedStockAnalysis:
    """Orchestrates the two-phase stock analysis pipeline."""

    def __init__(self, verbose=False, fresh=False):
        global logger
        logger = setup_logging(verbose=verbose)
        self.persistence = PersistenceHandler()
        self.verbose = verbose
        self.fresh = fresh
        self.start_time = datetime.now()

    def run(self):
        logger.info("Initializing scan...")
        get_cache_manager().clean_corrupted_cache_files()
        self.persistence.clear_old_data(7)

        # Macro Check
        if config.ANALYSIS_CONFIG.get("market_regime_detection", True):
            if not MarketRegimeDetection().get_simple_regime_check()["passed"]:
                logger.warning("Market regime is BEARISH. Skipping analysis.")
                return

        # Phase 1: Fetch Data
        symbols = StockScanner.get_symbols()
        symbols_list = list(symbols.keys())
        logger.info(f"Phase 1: Fetching data for {len(symbols_list)} stocks...")

        fetched = {}
        with ThreadPoolExecutor(max_workers=config.DATA_FETCH_THREADS) as ex:
            f_map = {
                ex.submit(get_historical_data, s, config.HISTORICAL_DATA_PERIOD, fresh=self.fresh): s
                for s in symbols_list
            }
            for f in as_completed(f_map):
                s = f_map[f]
                try:
                    df = f.result()
                    if not df.empty:
                        fetched[s] = df
                except Exception:
                    pass

        logger.info(f"Phase 1 Complete. Fetched {len(fetched)} stocks.")
        if not fetched:
            return

        # Phase 2: Parallel Analysis
        logger.info(f"Phase 2: Analyzing {len(fetched)} stocks...")

        bench = get_benchmark_data(config.HISTORICAL_DATA_PERIOD)
        cfg = {
            "ANALYSIS_CONFIG": dict(config.ANALYSIS_CONFIG),
            "STRATEGY_CONFIG": dict(config.STRATEGY_CONFIG),
            "BENCHMARK_DATA": bench.to_dict() if not bench.empty else {},
            "BENCHMARK_INDEX": bench.index.strftime("%Y-%m-%d %H:%M:%S").tolist() if not bench.empty else [],
        }

        items = [
            (s, symbols.get(s, s), df.to_dict(), df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(), cfg)
            for s, df in fetched.items()
        ]

        results = []
        if config.USE_MULTIPROCESSING_PIPELINE:
            logger.info(f"Phase 2: Analyzing {len(fetched)} stocks using {config.NUM_WORKER_PROCESSES} processes...")
            with multiprocessing.get_context("spawn").Pool(config.NUM_WORKER_PROCESSES, init_worker) as pool:
                for i, res in enumerate(pool.imap_unordered(analyze_stock_worker, items)):
                    results.append(res)
                    if not self.verbose:
                        print(f"\rProgress: {((i+1)/len(items))*100:.1f}%", end="", flush=True)
        else:
            logger.info(f"Phase 2: Analyzing {len(fetched)} stocks serially (Multiprocessing Disabled)...")
            init_worker()
            for i, item in enumerate(items):
                res = analyze_stock_worker(item)
                results.append(res)
                if not self.verbose:
                    print(f"\rProgress: {((i+1)/len(items))*100:.1f}%", end="", flush=True)

        if not self.verbose:
            print()

        # Phase 3: Persistence
        recos = 0
        for r in results:
            if r.get("success"):
                self.persistence.save_recommendation(r["result"])
                self.persistence.save_backtest_results(r["result"])
                if r["recommended"]:
                    recos += 1

        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Pipeline Finished: {duration/60:.1f}m | {recos} Recommendations found.")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (overrides config)")
    args = parser.parse_args()

    # Priority: Command line flag > Config file
    verbose = args.verbose or config.VERBOSE_LOGGING

    try:
        AutomatedStockAnalysis(verbose=verbose).run()
        return 0
    except Exception as e:
        print(f"Critical Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
