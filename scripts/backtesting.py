# Backtesting Engine
# File: scripts/backtesting.py

import backtrader as bt
import pandas as pd
from typing import Type, Dict, Any
from utils.logger import setup_logging

logger = setup_logging()

class BacktestingEngine:
    """
    A simple backtesting engine using Backtrader.
    """
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.001):
        """
        Initialize the backtesting engine with initial cash and commission.

        Args:
            initial_cash: Starting cash for the backtesting
            commission: Broker commission per trade (e.g., 0.001 for 0.1%)
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.cerebro = bt.Cerebro()
        self.cerebro.broker.set_cash(self.initial_cash)
        self.cerebro.broker.setcommission(commission=self.commission)
        
    def run_backtest(self, strategy_class: Type[bt.Strategy], data: pd.DataFrame, strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run backtesting on the given strategy and data.

        Args:
            strategy_class: Strategy class to backtest
            data: Stock data as a DataFrame
            strategy_params: Parameters for the strategy initialization

        Returns:
            Dictionary containing backtest results such as final portfolio value and profit/loss
        """
        # Convert DataFrame to Backtrader data feed
        data_feed = bt.feeds.PandasData(dataname=data)
        self.cerebro.adddata(data_feed)

        # Add strategy with parameters
        self.cerebro.addstrategy(strategy_class, **(strategy_params or {}))

        # Run backtest
        logger.info(f"Starting backtest with initial cash: {self.initial_cash}")
        initial_portfolio_value = self.cerebro.broker.getvalue()
        self.cerebro.run()
        final_portfolio_value = self.cerebro.broker.getvalue()
        
        # Calculate results
        profit_loss = final_portfolio_value - initial_portfolio_value
        roi = (profit_loss / initial_portfolio_value) * 100
        
        # Log results
        logger.info(f"Backtest complete - Final Portfolio Value: {final_portfolio_value}")
        logger.info(f"Profit/Loss: {profit_loss}, ROI: {roi:.2f}%")
        
        return {
            'initial_cash': self.initial_cash,
            'final_portfolio_value': final_portfolio_value,
            'profit_loss': profit_loss,
            'roi': roi
        }
