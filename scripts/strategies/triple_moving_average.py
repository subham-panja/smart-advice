"""
Triple Moving Average Strategy
File: scripts/strategies/triple_moving_average.py

This strategy uses three moving averages to identify trend alignment and strength.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Triple_Moving_Average(BaseStrategy):
    """
    Triple Moving Average Strategy.
    
    Buy Signal: All three MAs aligned bullishly (fast > medium > slow)
    Sell Signal: All three MAs aligned bearishly (fast < medium < slow)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 9)
        self.medium_period = self.get_parameter('medium_period', 21)
        self.slow_period = self.get_parameter('slow_period', 50)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Triple Moving Average strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.slow_period + 1):
            return -1
            
        try:
            # Calculate moving averages using TA-Lib
            close_prices = data['Close'].values
            
            ma_fast = ta.SMA(close_prices, timeperiod=self.fast_period)
            ma_medium = ta.SMA(close_prices, timeperiod=self.medium_period)
            ma_slow = ta.SMA(close_prices, timeperiod=self.slow_period)
            
            # Check if we have valid values
            if (pd.isna(ma_fast[-1]) or pd.isna(ma_medium[-1]) or pd.isna(ma_slow[-1]) or
                pd.isna(ma_fast[-2]) or pd.isna(ma_medium[-2]) or pd.isna(ma_slow[-2])):
                self.log_signal(-1, "Insufficient data for Triple MA calculation", data)
                return -1
            
            current_price = close_prices[-1]
            current_fast = ma_fast[-1]
            current_medium = ma_medium[-1]
            current_slow = ma_slow[-1]
            
            previous_fast = ma_fast[-2]
            previous_medium = ma_medium[-2]
            previous_slow = ma_slow[-2]
            
            # Perfect bullish alignment: Price > Fast > Medium > Slow
            if (current_price > current_fast > current_medium > current_slow):
                reason = f"Perfect bullish alignment: Price {current_price:.2f} > Fast {current_fast:.2f} > Med {current_medium:.2f} > Slow {current_slow:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Perfect bearish alignment: Price < Fast < Medium < Slow
            elif (current_price < current_fast < current_medium < current_slow):
                reason = f"Perfect bearish alignment: Price {current_price:.2f} < Fast {current_fast:.2f} < Med {current_medium:.2f} < Slow {current_slow:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Bullish crossover: Fast crosses above Medium while both above Slow
            elif (previous_fast <= previous_medium and current_fast > current_medium and
                  current_medium > current_slow and current_slow > current_slow):
                reason = f"Bullish MA crossover: Fast {current_fast:.2f} crosses above Medium {current_medium:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Bearish crossover: Fast crosses below Medium
            elif (previous_fast >= previous_medium and current_fast < current_medium):
                reason = f"Bearish MA crossover: Fast {current_fast:.2f} crosses below Medium {current_medium:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong bullish: Fast and Medium both above Slow, and Fast > Medium
            elif (current_fast > current_medium > current_slow and current_price > current_fast):
                reason = f"Strong bullish trend: Fast {current_fast:.2f} > Med {current_medium:.2f} > Slow {current_slow:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong bearish: Fast and Medium both below Slow, and Fast < Medium
            elif (current_fast < current_medium < current_slow and current_price < current_fast):
                reason = f"Strong bearish trend: Fast {current_fast:.2f} < Med {current_medium:.2f} < Slow {current_slow:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Partial bullish alignment
            elif (current_fast > current_medium and current_medium > current_slow):
                reason = f"Partial bullish alignment: Fast {current_fast:.2f} > Med {current_medium:.2f} > Slow {current_slow:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Check if price is above all MAs (bullish)
            elif (current_price > current_fast and current_price > current_medium and current_price > current_slow):
                reason = f"Price above all MAs: {current_price:.2f} > all averages"
                self.log_signal(1, reason, data)
                return 1
            
            # Check if price is below all MAs (bearish)
            elif (current_price < current_fast and current_price < current_medium and current_price < current_slow):
                reason = f"Price below all MAs: {current_price:.2f} < all averages"
                self.log_signal(-1, reason, data)
                return -1
            
            # Mixed signals - check majority
            elif (current_fast > current_slow):
                reason = f"Mixed signals, fast trend positive: Fast {current_fast:.2f} > Slow {current_slow:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Mixed/negative signals: Fast {current_fast:.2f}, Med {current_medium:.2f}, Slow {current_slow:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Triple MA calculation: {str(e)}", data)
            return -1
