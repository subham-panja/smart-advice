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
        try:
            # Ensure the DataFrame has proper datetime index for backtrader
            data_copy = data.copy()
            
            # Convert index to datetime if it's not already
            if not isinstance(data_copy.index, pd.DatetimeIndex):
                try:
                    data_copy.index = pd.to_datetime(data_copy.index)
                except Exception as e:
                    logger.error(f"Failed to convert index to datetime: {e}")
                    # Create a date range if conversion fails
                    data_copy.index = pd.date_range(start='2020-01-01', periods=len(data_copy), freq='D')
            
            # Ensure required columns exist with proper names
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in data_copy.columns:
                    if col == 'Volume' and 'volume' in data_copy.columns:
                        data_copy['Volume'] = data_copy['volume']
                    elif col == 'Volume':
                        data_copy['Volume'] = 0  # Default volume if missing
                    else:
                        # Try lowercase version
                        if col.lower() in data_copy.columns:
                            data_copy[col] = data_copy[col.lower()]
                        else:
                            logger.warning(f"Missing required column: {col}")
                            return {
                                'initial_cash': self.initial_cash,
                                'final_portfolio_value': self.initial_cash,
                                'profit_loss': 0,
                                'roi': 0,
                                'error': f'Missing required column: {col}'
                            }
            
            # Create a fresh cerebro instance to avoid data/strategy accumulation
            self.cerebro = bt.Cerebro()
            self.cerebro.broker.set_cash(self.initial_cash)
            self.cerebro.broker.setcommission(commission=self.commission)
            
            # Convert DataFrame to Backtrader data feed
            data_feed = bt.feeds.PandasData(dataname=data_copy)
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
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {
                'initial_cash': self.initial_cash,
                'final_portfolio_value': self.initial_cash,
                'profit_loss': 0,
                'roi': 0,
                'error': str(e)
            }
