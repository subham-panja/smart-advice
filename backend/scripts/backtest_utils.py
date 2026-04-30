import logging
from typing import Any, Dict

import pandas as pd

from scripts.backtesting_runner import BacktestingRunner

logger = logging.getLogger(__name__)


class BacktestUtils:
    """Class for backtesting orchestration and metrics."""

    def __init__(self):
        self.backtest_runner = BacktestingRunner()

    def perform_backtesting(self, symbol: str, df: pd.DataFrame, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Runs backtesting strictly from app_config, defaulting to dynamic if benchmarks are missing."""
        strategy_classes = app_config["analysis_config"].get("backtest_strategies", [])

        try:
            res = self.backtest_runner.run(symbol, df, strategy_classes=strategy_classes, app_config=app_config)
            if res["status"] == "completed":
                m = res["combined_metrics"]
                logger.info(f"Backtest {symbol}: CAGR {m['avg_cagr']}%, Win {m['avg_win_rate']}%")
            return res
        except Exception as e:
            logger.error(f"Backtest {symbol} error: {e}")
            raise e

    def calculate_overall_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates aggregate metrics across multiple backtest runs with no fallbacks."""
        if not results:
            raise ValueError("No results provided for aggregate metrics")

        rs = results.values()
        return {
            "average_cagr": round(sum(r["cagr"] for r in rs) / len(rs), 2),
            "average_win_rate": round(sum(r["win_rate"] for r in rs) / len(rs), 2),
            "total_trades": sum(r["total_trades"] for r in rs),
        }
