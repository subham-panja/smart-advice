import logging
import os
from typing import Any, Dict, Tuple

import pandas as pd

from scripts.analyzer import StockAnalyzer

logger = logging.getLogger(__name__)
_worker_analyzer = None
_persistence = None


def init_worker(verbose=False):
    """Initializer called once per worker process."""
    global _worker_analyzer, _persistence
    from utils.logger import setup_logging
    from utils.persistence_handler import PersistenceHandler

    setup_logging(verbose=verbose)
    _worker_analyzer = StockAnalyzer()
    _persistence = PersistenceHandler()

    # Use a localized logger for the worker initialization message
    local_logger = logging.getLogger("WorkerInit")
    local_logger.info(f"Worker {os.getpid()} initialized with Persistence (verbose={verbose})")


def analyze_stock_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function that runs the analysis pipeline on pre-fetched data."""
    global _worker_analyzer, _persistence
    symbol, name, hist_dict, idx, cfg = args

    try:
        hist = pd.DataFrame.from_dict(hist_dict) if hist_dict else pd.DataFrame()
        if not hist.empty:
            hist.index = pd.to_datetime(idx)

        bench_dict, bench_idx = cfg.get("BENCHMARK_DATA", {}), cfg.get("BENCHMARK_INDEX", [])
        bench = pd.DataFrame.from_dict(bench_dict) if bench_dict else pd.DataFrame()
        if not bench.empty:
            bench.index = pd.to_datetime(bench_idx)

        res = _worker_analyzer.analyze_stock_with_data(symbol, name, hist, cfg, index_data=bench)

        # Direct persistence from worker using pre-initialized handler
        if res.get("is_recommended"):
            _persistence.save_recommendation(res)
            _persistence.save_backtest_results(res)

        return {
            "success": "error" not in res,
            "symbol": symbol,
            "result": res,
            "recommended": res.get("is_recommended", False),
        }
    except Exception as e:
        logger.error(f"Worker error {symbol}: {e}")
        return {"success": False, "symbol": symbol, "error": str(e)}
