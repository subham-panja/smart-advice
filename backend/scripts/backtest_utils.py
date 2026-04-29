import pandas as pd
import logging
from typing import Dict, Any
from scripts.backtesting_runner import BacktestingRunner

logger = logging.getLogger(__name__)

class BacktestUtils:
    """Class for backtesting orchestration and metrics."""
    
    def __init__(self):
        self.backtest_runner = BacktestingRunner()

    def perform_backtesting(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Runs backtesting and logs key metrics."""
        try:
            res = self.backtest_runner.run(symbol, df)
            if res.get('status') == 'completed':
                m = res.get('combined_metrics', {})
                logger.info(f"Backtest {symbol}: CAGR {m.get('avg_cagr', 0)}%, Win {m.get('avg_win_rate', 0)}%")
            return res
        except Exception as e:
            logger.error(f"Backtest {symbol} error: {e}")
            return {'symbol': symbol, 'status': 'error', 'error': str(e)}

    def calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates aggregate metrics across multiple backtest runs."""
        if not results: return {}
        try:
            rs = results.values()
            return {
                'average_cagr': round(sum(r.get('cagr', 0) for r in rs) / len(rs), 2),
                'average_win_rate': round(sum(r.get('win_rate', 0) for r in rs) / len(rs), 2),
                'total_trades': sum(r.get('total_trades', 0) for r in rs)
            }
        except: return {}
