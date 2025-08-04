"""
EMA Crossover Strategy (12/26)
File: scripts/strategies/ema_crossover_12_26.py

This strategy implements the EMA crossover using 12-day and 26-day Exponential Moving Averages.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class EMA_Crossover_12_26(BaseStrategy):
    """
    EMA Crossover Strategy using 12-day and 26-day Exponential Moving Averages.
    
    Buy Signal: 12-day EMA crosses above 26-day EMA
    Sell Signal: 12-day EMA crosses below 26-day EMA
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 12)
        self.slow_period = self.get_parameter('slow_period', 26)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the EMA crossover strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.slow_period):
            return -1
            
        try:
            # Calculate EMAs using TA-Lib
            close_prices = data['Close'].values
            
            # Calculate EMAs
            ema_fast = ta.EMA(close_prices, timeperiod=self.fast_period)
            ema_slow = ta.EMA(close_prices, timeperiod=self.slow_period)
            
            # Check if we have valid values for the latest periods
            if (pd.isna(ema_fast[-1]) or pd.isna(ema_slow[-1]) or 
                pd.isna(ema_fast[-2]) or pd.isna(ema_slow[-2])):
                self.log_signal(-1, "Insufficient data for EMA calculation", data)
                return -1
            
            # Check for bullish crossover
            # Fast EMA was below slow EMA and now crosses above
            if (ema_fast[-2] <= ema_slow[-2] and ema_fast[-1] > ema_slow[-1]):
                reason = f"Bullish EMA crossover: {self.fast_period}-day EMA ({ema_fast[-1]:.2f}) crosses above {self.slow_period}-day EMA ({ema_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Check for bearish crossover
            elif (ema_fast[-2] >= ema_slow[-2] and ema_fast[-1] < ema_slow[-1]):
                reason = f"Bearish EMA crossover: {self.fast_period}-day EMA ({ema_fast[-1]:.2f}) crosses below {self.slow_period}-day EMA ({ema_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current trend - if fast EMA is above slow EMA, it's bullish
            elif ema_fast[-1] > ema_slow[-1]:
                reason = f"Bullish EMA trend: {self.fast_period}-day EMA ({ema_fast[-1]:.2f}) above {self.slow_period}-day EMA ({ema_slow[-1]:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Fast EMA is below slow EMA - bearish
            else:
                reason = f"Bearish EMA trend: {self.fast_period}-day EMA ({ema_fast[-1]:.2f}) below {self.slow_period}-day EMA ({ema_slow[-1]:.2f})"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in EMA crossover calculation: {str(e)}", data)
            return -1
