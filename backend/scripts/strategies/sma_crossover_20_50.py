"""
SMA Crossover Strategy (20-50)
File: scripts/strategies/sma_crossover_20_50.py

This strategy implements the SMA crossover using 20-day and 50-day Simple Moving Averages.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class SMA_Crossover_20_50(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy using 20-day and 50-day SMAs.
    
    Buy Signal: 20-day SMA crosses above 50-day SMA
    Sell Signal: 20-day SMA crosses below 50-day SMA
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 20)
        self.slow_period = self.get_parameter('slow_period', 50)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core SMA crossover strategy logic.
        Called by base class run_strategy method after volume filtering.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
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
            
            # Check if we have valid values
            if (pd.isna(sma_fast[-1]) or pd.isna(sma_slow[-1]) or 
                pd.isna(sma_fast[-2]) or pd.isna(sma_slow[-2])):
                self.log_signal(-1, "Insufficient data for SMA calculation", data)
                return -1
            
            # Check for bullish crossover
            if (sma_fast[-2] <= sma_slow[-2] and sma_fast[-1] > sma_slow[-1]):
                reason = f"Bullish SMA Cross: {self.fast_period}-day SMA ({sma_fast[-1]:.2f}) crosses above {self.slow_period}-day SMA ({sma_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Check for bearish crossover
            elif (sma_fast[-2] >= sma_slow[-2] and sma_fast[-1] < sma_slow[-1]):
                reason = f"Bearish SMA Cross: {self.fast_period}-day SMA ({sma_fast[-1]:.2f}) crosses below {self.slow_period}-day SMA ({sma_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current trend
            elif sma_fast[-1] > sma_slow[-1]:
                reason = f"Bullish trend: {self.fast_period}-day SMA ({sma_fast[-1]:.2f}) above {self.slow_period}-day SMA ({sma_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Bearish trend: {self.fast_period}-day SMA ({sma_fast[-1]:.2f}) below {self.slow_period}-day SMA ({sma_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in SMA crossover calculation: {str(e)}", data)
            return -1
