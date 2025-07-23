"""
Moving Average Crossover Strategy (50-200)
File: scripts/strategies/ma_crossover_50_200.py

This strategy implements the classic golden cross (50-day MA crosses above 200-day MA)
and death cross (50-day MA crosses below 200-day MA) trading signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class MA_Crossover_50_200(BaseStrategy):
    """
    Moving Average Crossover Strategy using 50-day and 200-day Simple Moving Averages.
    
    Buy Signal: 50-day MA crosses above 200-day MA (Golden Cross)
    Sell Signal: 50-day MA crosses below 200-day MA (Death Cross)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 50)
        self.slow_period = self.get_parameter('slow_period', 200)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the MA crossover strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal (golden cross), -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.slow_period):
            return -1
            
        try:
            # Calculate moving averages using TA-Lib
            close_prices = data['Close'].values
            
            # Calculate SMAs
            sma_fast = ta.SMA(close_prices, timeperiod=self.fast_period)
            sma_slow = ta.SMA(close_prices, timeperiod=self.slow_period)
            
            # Check if we have valid values for the latest periods
            if (pd.isna(sma_fast[-1]) or pd.isna(sma_slow[-1]) or 
                pd.isna(sma_fast[-2]) or pd.isna(sma_slow[-2])):
                self.log_signal(-1, "Insufficient data for MA calculation", data)
                return -1
            
            # Check for golden cross (bullish signal)
            # Fast MA was below slow MA and now crosses above
            if (sma_fast[-2] <= sma_slow[-2] and sma_fast[-1] > sma_slow[-1]):
                reason = f"Golden Cross: {self.fast_period}-day MA ({sma_fast[-1]:.2f}) crosses above {self.slow_period}-day MA ({sma_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Check for death cross or if fast MA is below slow MA
            elif (sma_fast[-2] >= sma_slow[-2] and sma_fast[-1] < sma_slow[-1]):
                reason = f"Death Cross: {self.fast_period}-day MA ({sma_fast[-1]:.2f}) crosses below {self.slow_period}-day MA ({sma_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current trend - if fast MA is above slow MA, it's bullish
            elif sma_fast[-1] > sma_slow[-1]:
                reason = f"Bullish trend: {self.fast_period}-day MA ({sma_fast[-1]:.2f}) above {self.slow_period}-day MA ({sma_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Fast MA is below slow MA - bearish
            else:
                reason = f"Bearish trend: {self.fast_period}-day MA ({sma_fast[-1]:.2f}) below {self.slow_period}-day MA ({sma_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in MA crossover calculation: {str(e)}", data)
            return -1
