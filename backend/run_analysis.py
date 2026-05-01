#!/usr/bin/env python3
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict

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

    def run(self, strategy_config: Dict[str, Any]):
        strat_name = strategy_config["name"]
        logger.info(f"🚀 Starting analysis for Strategy: {strat_name}")

        get_cache_manager().clean_corrupted_cache_files()

        # Macro Check
        if strategy_config["analysis_config"]["market_regime_detection"]:
            if not MarketRegimeDetection().get_simple_regime_check(strategy_config["market_regime_config"])["passed"]:
                logger.warning(f"[{strat_name}] Market regime is BEARISH. Skipping strategy.")
                return

        # Phase 1: Fetch Data
        thresholds = strategy_config.get("recommendation_thresholds", {})
        backtest_period = thresholds.get("backtest_period", config.HISTORICAL_DATA_PERIOD)

        symbols = StockScanner.get_symbols(strategy_config=strategy_config)
        symbols_list = list(symbols.keys())
        logger.info(f"Phase 1: Fetching data ({backtest_period}) for {len(symbols_list)} stocks...")

        fetched = {}
        with ThreadPoolExecutor(max_workers=config.DATA_FETCH_THREADS) as ex:
            f_map = {ex.submit(get_historical_data, s, backtest_period, fresh=self.fresh): s for s in symbols_list}
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

        # Build comprehensive config for worker
        worker_cfg = dict(strategy_config)
        worker_cfg.update(
            {
                "BENCHMARK_DATA": bench.to_dict() if not bench.empty else {},
                "BENCHMARK_INDEX": bench.index.strftime("%Y-%m-%d %H:%M:%S").tolist() if not bench.empty else [],
                # Legacy keys for backward compatibility with analyzer.py if needed, but we'll update analyzer.py
                "STRATEGY_NAME": strat_name,
                "ANALYSIS_CONFIG": strategy_config["analysis_config"],
                "STRATEGY_CONFIG": strategy_config["strategy_config"],
                "RECOMMENDATION_THRESHOLDS": strategy_config["recommendation_thresholds"],
            }
        )

        items = [
            (s, symbols[s], df.to_dict(), df.index.strftime("%Y-%m-%d %H:%M:%S").tolist(), worker_cfg)
            for s, df in fetched.items()
        ]

        results = []
        if config.USE_MULTIPROCESSING_PIPELINE:
            logger.info(f"Phase 2: Analyzing {len(fetched)} stocks using {config.NUM_WORKER_PROCESSES} processes...")
            with multiprocessing.get_context("spawn").Pool(
                config.NUM_WORKER_PROCESSES, init_worker, (self.verbose,)
            ) as pool:
                for i, res in enumerate(pool.imap_unordered(analyze_stock_worker, items)):
                    if res["success"]:
                        self.persistence.save_recommendation(res["result"])
                        self.persistence.save_backtest_results(res["result"])
                        results.append(res["result"])

                    if not self.verbose:
                        print(f"\rProgress: {((i+1)/len(items))*100:.1f}%", end="", flush=True)

            # Save Audit Log
            audit_data = [r.get("audit_log") for r in results if r.get("audit_log")]
            os.makedirs("logs", exist_ok=True)
            with open("logs/audit_log.json", "w") as f:
                json.dump(audit_data, f, indent=4)

            logger.info(f"Phase 2 Complete. Fetched {len(fetched)} stocks. Audit log saved.")
            return results
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

        # Summary
        recos = sum(1 for r in results if r["success"] and r["recommended"])
        duration = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"[{strat_name}] Pipeline Finished: {duration/60:.1f}m | {recos} Recommendations found.")
