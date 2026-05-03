import importlib
import logging
from typing import Any, Dict, List

import pandas as pd

# Defaults come from strategy config now
from scripts.backtesting import BacktestingEngine
from scripts.strategies.base_strategy import BacktraderStrategy

logger = logging.getLogger(__name__)


class BacktestingRunner:
    """Evaluates multiple strategies or the core dynamic strategy and calculates performance metrics."""

    def __init__(self, initial_cash: float = None, commission: float = None):
        self.initial_cash = initial_cash if initial_cash is not None else 100000.0
        self.commission = commission if commission is not None else 0.0020

    def run(
        self, symbol: str, df: pd.DataFrame, strategy_classes: List[str], app_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        thresholds = app_config.get("RECOMMENDATION_THRESHOLDS", {})
        min_days = thresholds.get("min_data_required_days", 252)

        if len(df) < min_days:
            logger.warning(f"Backtest skipped for {symbol}: Insufficient data ({len(df)} < {min_days} days)")
            return {"symbol": symbol, "status": "insufficient_data"}

        results = {}
        # If no strategy_classes provided, we perform a DYNAMIC backtest of the main strategy
        execution_list = strategy_classes if strategy_classes else ["DYNAMIC"]

        for name in execution_list:
            try:
                engine = BacktestingEngine(self.initial_cash, self.commission)
                bt_res = engine.run_backtest(
                    self._create_strategy(name, app_config), df, params={"symbol": symbol, "strat_params": app_config}
                )
                results[name] = self._calc_metrics(bt_res, df)
            except Exception as e:
                logger.error(f"Backtest {name} error for {symbol}: {e}")
                raise e

        combined = self._combine(results)
        # Pull trades from DYNAMIC result or first strategy
        first_res = results.get("DYNAMIC", next(iter(results.values())))
        return {
            "symbol": symbol,
            "status": "completed",
            "strategy_results": results,
            "combined_metrics": combined,
            "trades": first_res.get("trades", []),
        }

    def _create_strategy(self, name: str, app_config: Dict[str, Any]):
        if name == "DYNAMIC":
            # This is the "User's Strategy" - dynamic and based on JSON
            class DynamicBTStrategy(BacktraderStrategy):
                def __init__(self, *args, **kwargs):
                    BacktraderStrategy.__init__(self, *args, **kwargs)
                    from scripts.swing_trading_signals import SwingTradingSignalAnalyzer

                    self.analyzer = SwingTradingSignalAnalyzer()

                def _execute_strategy_logic(self, data, symbol="UNKNOWN"):
                    # Use the full strat_params (which is app_config) for the analyzer
                    res = self.analyzer.analyze_swing_opportunity(symbol, data, strategy_config=self.strat_params)
                    return 1 if res.get("recommendation") == "BUY" else 0

                def notify_order(self, order):
                    if order.status in [order.Completed]:
                        self.last_executed_size = abs(order.executed.size)

            return DynamicBTStrategy

        mapping = {
            "MA_Crossover_50_200": "scripts.strategies.ma_crossover_50_200",
            "RSI_Overbought_Oversold": "scripts.strategies.rsi_overbought_oversold",
            "MACD_Signal_Crossover": "scripts.strategies.macd_signal_crossover",
            "Bollinger_Band_Breakout": "scripts.strategies.bollinger_band_breakout",
        }

        # Check if it's a known standard strategy
        if name not in mapping:
            raise ValueError(f"Unknown strategy: {name}. Must be in {list(mapping.keys())} or 'DYNAMIC'")

        class BTStrategy(BacktraderStrategy):
            def _execute_strategy_logic(self, data, symbol="UNKNOWN"):
                mod = importlib.import_module(mapping[name])
                # Backtesting-specific param lookup
                strat_params = app_config["STRATEGY_CONFIG"][name]
                return getattr(mod, name)(strat_params)._execute_strategy_logic(data, symbol=symbol)

        return BTStrategy

    def _calc_metrics(self, bt: dict, df: pd.DataFrame) -> dict:
        t = bt.get("trade_analysis", {})

        # Safe extraction with defaults for no-trade scenarios

        won = t.get("won", {}).get("total", 0)
        lost = t.get("lost", {}).get("total", 0)
        total = t.get("total", {}).get("total", 0)

        p_won = t.get("won", {}).get("pnl", {}).get("total", 0.0)
        p_lost = abs(t.get("lost", {}).get("pnl", {}).get("total", 0.0))

        wr = (won / total * 100) if total > 0 else 0
        pf = (p_won / p_lost) if p_lost > 0 else (999.0 if p_won > 0 else 0.0)

        # Safe expectancy calculation
        avg_won = p_won / won if won > 0 else 0
        avg_lost = p_lost / lost if lost > 0 else 0
        exp = ((wr / 100 * avg_won) - ((1 - wr / 100) * avg_lost)) if total > 0 else 0.0

        # Dynamic Year Calculation for CAGR
        try:
            # Ensure the index is datetime if it isn't already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            days = (df.index[-1] - df.index[0]).days
            years = days / 365.25
        except Exception:
            years = len(df) / 252.0  # Fallback to trading year estimate

        cagr = ((bt["final_portfolio_value"] / bt["initial_cash"]) ** (1 / years) - 1) * 100 if years > 0.1 else 0.0

        return {
            "cagr": round(cagr, 2),
            "win_rate": round(wr, 2),
            "expectancy": round(exp, 2),
            "profit_factor": round(min(pf, 999.0), 2),
            "total_trades": total,
            "trades": bt.get("trades", []),
        }

    def _combine(self, results: dict) -> dict:
        res = [r for r in results.values()]
        if not res:
            raise ValueError("No backtest results to combine")
        return {
            "avg_cagr": round(sum(r["cagr"] for r in res) / len(res), 2),
            "avg_win_rate": round(sum(r["win_rate"] for r in res) / len(res), 2),
            "avg_expectancy": round(sum(r["expectancy"] for r in res) / len(res), 2),
            "avg_profit_factor": round(sum(r["profit_factor"] for r in res) / len(res), 2),
            "total_trades": sum(r["total_trades"] for r in res),
        }
