import logging
from typing import Any, Dict, Type

import backtrader as bt
import pandas as pd

logger = logging.getLogger(__name__)


class RiskPercentSizer(bt.Sizer):
    """
    Sizer that calculates quantity based on risk % of total portfolio value.
    Example: Risk 1% of capital. If Stop Loss is 5% away, size = 20% of capital.
    """

    params = (
        ("risk_pct", 1.0),  # Risk 1% of total value per trade
        ("min_shares", 1),
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        if not isbuy:
            return self.broker.getposition(data).size

        # Get total portfolio value (Cash + Open Positions) for compounding
        total_value = self.broker.get_value()

        # We need the stop loss price from the strategy to calculate risk distance
        # Strategy should store it in self.current_stop_loss
        stop_loss = getattr(self.strategy, "backtest_stop_loss", None)
        entry_price = data.close[0]

        if stop_loss is None or stop_loss >= entry_price:
            # Fallback to simple % of capital if no SL provided
            return int((total_value * 0.10) / entry_price)

        risk_per_share = entry_price - stop_loss
        total_risk_allowed = total_value * (self.params.risk_pct / 100.0)

        size = int(total_risk_allowed / risk_per_share)
        return max(size, self.params.min_shares)


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

            # Risk-Based Position Sizing (Compounding)
            risk_pct = params.get("strat_params", {}).get("risk_management", {}).get("risk_per_trade_pct", 1.0)
            cerebro.addsizer(RiskPercentSizer, risk_pct=risk_pct)

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
