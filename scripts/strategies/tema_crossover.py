"""
TEMA (Triple Exponential Moving Average) Crossover Strategy
File: scripts/strategies/tema_crossover.py

This strategy uses TEMA crossover to identify trend changes.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class TEMA_Crossover(BaseStrategy):
    """
    TEMA Crossover Strategy.
    
    Buy Signal: Fast TEMA crosses above slow TEMA
    Sell Signal: Fast TEMA crosses below slow TEMA
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 12)
        self.slow_period = self.get_parameter('slow_period', 26)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the TEMA crossover strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.slow_period + 1):
            return -1
            
        try:
            # Calculate TEMA using TA-Lib
            close_prices = data['Close'].values
            
            tema_fast = ta.TEMA(close_prices, timeperiod=self.fast_period)
            tema_slow = ta.TEMA(close_prices, timeperiod=self.slow_period)
            
            # Check if we have valid values
            if (pd.isna(tema_fast[-1]) or pd.isna(tema_slow[-1]) or 
                pd.isna(tema_fast[-2]) or pd.isna(tema_slow[-2])):
                self.log_signal(-1, "Insufficient data for TEMA calculation", data)
                return -1
            
            # Check for bullish crossover
            if (tema_fast[-2] <= tema_slow[-2] and tema_fast[-1] > tema_slow[-1]):
                reason = f"Bullish TEMA Cross: Fast ({tema_fast[-1]:.2f}) crosses above Slow ({tema_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Check for bearish crossover
            elif (tema_fast[-2] >= tema_slow[-2] and tema_fast[-1] < tema_slow[-1]):
                reason = f"Bearish TEMA Cross: Fast ({tema_fast[-1]:.2f}) crosses below Slow ({tema_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current trend
            elif tema_fast[-1] > tema_slow[-1]:
                reason = f"Bullish TEMA trend: Fast ({tema_fast[-1]:.2f}) > Slow ({tema_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Bearish TEMA trend: Fast ({tema_fast[-1]:.2f}) < Slow ({tema_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in TEMA calculation: {str(e)}", data)
            return -1
