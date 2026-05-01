import logging
from typing import Any, Dict, Type

import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)


class TradeList(bt.Analyzer):
    """Custom analyzer to collect detailed list of trades."""

    def __init__(self):
        self.trades = []

    def notify_trade(self, trade):
        if trade.isclosed:
            try:
                # Use strategy's captured size and direct price access
                entry_price = trade.price
                exit_price = trade.data.close[0]

                # Get size from strategy if available, fallback to deriving from PnL
                size = getattr(self.strategy, "last_executed_size", 0)
                pnl = trade.pnlcomm

                if size == 0 and abs(exit_price - entry_price) > 0.01:
                    size = abs(pnl / (exit_price - entry_price))

                pnl_pct = (pnl / (entry_price * size)) * 100 if entry_price > 0 and size > 0 else 0

                self.trades.append(
                    {
                        "symbol": self.strategy.p.symbol,
                        "entry_date": bt.num2date(trade.dtopen).strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": round(entry_price, 2),
                        "exit_date": bt.num2date(trade.dtclose).strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_price": round(exit_price, 2),
                        "quantity": round(size, 0),
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "is_profitable": pnl > 0,
                    }
                )
            except Exception as e:
                logger.error(f"Error extracting trade details for {self.strategy.p.symbol}: {e}")

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

            # Default sizer (Strategy will override with explicit sizes)
            cerebro.addsizer(bt.sizers.FixedSize, stake=1)

            cerebro.adddata(bt.feeds.PandasData(dataname=df))
            cerebro.addstrategy(strategy_class, tradehistory=True, **params)

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
