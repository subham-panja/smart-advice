import logging
from typing import Any, Dict, Type

import backtrader as bt
import pandas as pd

from config import RISK_MANAGEMENT

logger = logging.getLogger(__name__)


class TradeList(bt.Analyzer):
    """Custom analyzer to collect detailed list of trades."""

    def __init__(self):
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append(
                {
                    "symbol": self.strategy.p.symbol,
                    "entry_date": bt.num2date(trade.dtopen),
                    "entry_price": trade.priceover,
                    "exit_date": bt.num2date(trade.dtclose),
                    "exit_price": trade.priceout,
                    "quantity": trade.size,
                    "pnl": trade.pnl,
                    "pnl_pct": (trade.pnl / (trade.priceover * abs(trade.size))) * 100 if trade.priceover else 0,
                }
            )

    def get_analysis(self):
        return self.trades


class BacktestingEngine:
    """Core backtesting engine using Backtrader with strict configuration."""

    def __init__(self, initial_cash: float, commission: float):
        self.initial_cash = initial_cash
        self.commission = commission

    def run_backtest(
        self, strategy_class: Type[bt.Strategy], df: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Runs a backtest strictly using provided parameters."""
        try:
            cerebro = bt.Cerebro()
            cerebro.broker.set_cash(self.initial_cash)
            cerebro.broker.setcommission(commission=self.commission)

            # Position sizing strictly from config
            pos_pct = RISK_MANAGEMENT["position_sizing"]["max_position_pct"] * 100
            cerebro.addsizer(bt.sizers.PercentSizer, percents=pos_pct)

            cerebro.adddata(bt.feeds.PandasData(dataname=df))
            cerebro.addstrategy(strategy_class, **params)

            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
            cerebro.addanalyzer(TradeList, _name="tradelist")

            results = cerebro.run()
            strat = results[0]

            final_val = cerebro.broker.getvalue()
            return {
                "initial_cash": self.initial_cash,
                "final_portfolio_value": final_val,
                "roi": ((final_val - self.initial_cash) / self.initial_cash) * 100,
                "trade_analysis": strat.analyzers.trades.get_analysis(),
                "drawdown": strat.analyzers.drawdown.get_analysis(),
                "trades": strat.analyzers.tradelist.get_analysis(),
            }
        except Exception as e:
            logger.error(f"Critical backtest failure: {e}")
            raise e
