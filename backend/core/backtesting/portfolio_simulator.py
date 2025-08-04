"""
Portfolio Simulator Module
==========================

Simulates trading activities within a simulated portfolio environment.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.logger import setup_logging

logger = setup_logging()

class PortfolioSimulator:
    """
    Simulates a trading portfolio based on defined strategies and parameters.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """Initialize the portfolio simulator."""
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """Reset the portfolio to initial state."""
        self.cash = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
    
    def simulate_trading(self, data: pd.DataFrame, strategy_signals: pd.Series) -> Dict[str, Any]:
        """
        Simulate trading based on historical data and strategy signals.
        
        Args:
            data: Historical OHLCV data
            strategy_signals: Series with BUY/SELL/HOLD signals
            
        Returns:
            Dictionary with simulation results
        """
        try:
            if data.empty or strategy_signals.empty:
                return {'error': 'Empty data or signals'}
            
            logger.info("Starting trading simulation")
            self.reset()
            
            # Align data and signals
            aligned_data = data.align(strategy_signals, join='inner', axis=0)
            if aligned_data[0].empty:
                return {'error': 'No aligned data between prices and signals'}
            
            data_aligned, signals_aligned = aligned_data
            
            # Execute trades based on signals
            for i, (date, row) in enumerate(data_aligned.iterrows()):
                current_price = row['Close']
                signal = signals_aligned.iloc[i] if i < len(signals_aligned) else 'HOLD'
                
                # Calculate current portfolio value
                portfolio_value = self.cash + (self.position * current_price)
                self.portfolio_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'position': self.position,
                    'price': current_price
                })
                
                # Execute trades
                if signal == 'BUY' and self.position == 0 and self.cash > current_price:
                    # Enter long position
                    shares_to_buy = int(self.cash * 0.95 / current_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        self.cash -= cost
                        self.position = shares_to_buy
                        
                        self.trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': cost
                        })
                        
                elif signal == 'SELL' and self.position > 0:
                    # Exit long position
                    proceeds = self.position * current_price
                    self.cash += proceeds
                    
                    self.trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': self.position,
                        'value': proceeds
                    })
                    
                    self.position = 0
            
            # Calculate final portfolio value
            final_price = data_aligned['Close'].iloc[-1]
            final_portfolio_value = self.cash + (self.position * final_price)
            
            logger.info("Trading simulation completed")
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': final_portfolio_value,
                'trades': self.trades,
                'portfolio_history': self.portfolio_values
            }
            
        except Exception as e:
            logger.error(f"Error simulating trading: {e}")
            return {'error': str(e)}
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current simulation state.
        
        Returns:
            Dictionary containing simulation summary
        """
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'current_position': self.position,
            'total_trades': len(self.trades),
            'portfolio_points': len(self.portfolio_values)
        }

