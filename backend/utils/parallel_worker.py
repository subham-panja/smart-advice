import os
import logging
import pandas as pd
from typing import Dict, Any, Tuple
from scripts.analyzer import StockAnalyzer

logger = logging.getLogger(__name__)
_worker_analyzer = None

def init_worker():
    """Initializer called once per worker process."""
    global _worker_analyzer
    _worker_analyzer = StockAnalyzer()
    logger.info(f"Worker {os.getpid()} initialized")

def analyze_stock_worker(args: Tuple) -> Dict[str, Any]:
    """Worker function that runs the analysis pipeline on pre-fetched data."""
    global _worker_analyzer
    symbol, name, hist_dict, idx, cfg = args
    
    try:
        hist = pd.DataFrame.from_dict(hist_dict) if hist_dict else pd.DataFrame()
        if not hist.empty: hist.index = pd.to_datetime(idx)
        
        bench_dict, bench_idx = cfg.get('BENCHMARK_DATA', {}), cfg.get('BENCHMARK_INDEX', [])
        bench = pd.DataFrame.from_dict(bench_dict) if bench_dict else pd.DataFrame()
        if not bench.empty: bench.index = pd.to_datetime(bench_idx)

        res = _worker_analyzer.analyze_stock_with_data(symbol, name, hist, cfg, index_data=bench)
        return {'success': 'error' not in res, 'symbol': symbol, 'result': res, 'recommended': res.get('is_recommended', False)}
    except Exception as e:
        logger.error(f"Worker error {symbol}: {e}")
        return {'success': False, 'symbol': symbol, 'error': str(e)}
