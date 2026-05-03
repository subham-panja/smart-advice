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
        # Check if individual stock backtesting is enabled
        if not app_config["analysis_config"].get("individual_stock_backtest", True):
            return {"status": "skipped", "combined_metrics": {"avg_cagr": 0.0, "avg_win_rate": 0.0}}

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

    def perform_backtrader_backtest(self, symbol: str, df: pd.DataFrame, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest using backtrader engine (alternative to built-in backtesting).

        This is disabled by default but implemented for future use.
        To enable, set analysis_config.backtest_engine = 'backtrader' in strategy config.
        """
        backtest_engine = app_config["analysis_config"].get("backtest_engine", "builtin")

        if backtest_engine != "backtrader":
            return {"status": "skipped", "reason": "Backtrader engine not enabled"}

        try:
            import backtrader as bt

            # Create cerebro engine
            cerebro = bt.Cerebro()

            # Add data feed
            data = bt.feeds.PandasData(
                dataname=df,
                datetime="Date",
                open="Open",
                high="High",
                low="Low",
                close="Close",
                volume="Volume",
                openinterest=-1,
            )
            cerebro.adddata(data)

            # Add strategy (use default SMA crossover for now)
            cerebro.addstrategy(bt.strategies.SmaCross)

            # Set cash and commission
            cerebro.broker.setcash(app_config.get("trading_config", {}).get("initial_capital", 100000))
            cerebro.broker.setcommission(
                commission=app_config.get("trading_config", {}).get("brokerage_charges", 0.002)
            )

            # Run backtest
            initial_value = cerebro.broker.getvalue()
            cerebro.run()
            final_value = cerebro.broker.getvalue()

            # Calculate metrics
            total_return = ((final_value - initial_value) / initial_value) * 100

            return {
                "status": "completed",
                "engine": "backtrader",
                "combined_metrics": {
                    "avg_cagr": total_return,  # Simplified - should calculate proper CAGR
                    "avg_win_rate": 0.0,  # TODO: Calculate from trades
                    "total_return_pct": total_return,
                    "final_value": final_value,
                },
                "trades": [],  # TODO: Extract trades from results
            }
        except ImportError:
            logger.warning("backtrader not installed - skipping backtrader backtest")
            return {"status": "skipped", "reason": "backtrader not available"}
        except Exception as e:
            logger.error(f"Backtrader backtest error for {symbol}: {e}")
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
