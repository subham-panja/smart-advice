"""
Parallel Worker Module
File: utils/parallel_worker.py

Top-level module containing worker functions for multiprocessing.
Functions must be importable at the module level for pickle serialization.
"""

import os

# Load config early to apply OpenMP/threading limits before importing data science libraries
import config

import logging
import pandas as pd
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Per-process global state (initialized once per worker process)
_worker_analyzer = None
_worker_app = None


def init_worker():
    """
    Initializer called once per worker process.
    Creates a StockAnalyzer instance local to this process.
    """
    global _worker_analyzer, _worker_app
    from app import create_app
    from scripts.analyzer import StockAnalyzer

    _worker_app = create_app()
    _worker_analyzer = StockAnalyzer()
    logger.info(f"Worker process {os.getpid()} initialized")


def analyze_stock_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function that runs in a child process.
    Receives pre-fetched data and runs the full analysis pipeline.

    Args:
        args: Tuple of (symbol, company_name, historical_data_dict, index_data, app_config_dict)
              - historical_data_dict: DataFrame converted to dict via df.to_dict()
              - index_data: DataFrame index as list of ISO strings
              - app_config_dict: Serializable subset of app.config

    Returns:
        Analysis result dictionary
    """
    global _worker_analyzer, _worker_app

    symbol, company_name, hist_data_dict, index_strings, app_config_dict = args

    try:
        # Reconstruct DataFrame from dict
        if hist_data_dict and index_strings:
            historical_data = pd.DataFrame.from_dict(hist_data_dict)
            historical_data.index = pd.to_datetime(index_strings)
            historical_data.index.name = 'Date'
        else:
            historical_data = pd.DataFrame()

        with _worker_app.app_context():
            # Build app_config from the serialized config
            app_config = dict(_worker_app.config)
            app_config.update(app_config_dict)

            result = _worker_analyzer.analyze_stock_with_data(
                symbol, company_name, historical_data, app_config
            )

            return {
                'success': True,
                'symbol': symbol,
                'result': result,
                'recommended': result.get('is_recommended', False)
            }

    except Exception as e:
        logger.error(f"Worker error analyzing {symbol}: {e}")
        return {
            'success': False,
            'symbol': symbol,
            'error': str(e),
            'recommended': False
        }
