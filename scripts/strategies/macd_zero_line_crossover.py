"""
MACD Zero Line Crossover Strategy
File: scripts/strategies/macd_zero_line_crossover.py

This strategy uses MACD zero line crossovers to identify trend changes.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class MACD_Zero_Line_Crossover(BaseStrategy):
    """
    MACD Zero Line Crossover Strategy.
    
    Buy Signal: MACD line crosses above zero
    Sell Signal: MACD line crosses below zero
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 12)
        self.slow_period = self.get_parameter('slow_period', 26)
        self.signal_period = self.get_parameter('signal_period', 9)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the MACD zero line crossover strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.slow_period + self.signal_period):
            return -1
            
        try:
            # Calculate MACD using TA-Lib
            close_prices = data['Close'].values
            
            macd, macd_signal, macd_histogram = ta.MACD(
                close_prices, 
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
            
            # Check if we have valid values
            if pd.isna(macd[-1]) or pd.isna(macd[-2]):
                self.log_signal(-1, "Insufficient data for MACD calculation", data)
                return -1
            
            current_macd = macd[-1]
            previous_macd = macd[-2]
            
            # Buy signal: MACD crosses above zero
            if previous_macd <= 0 and current_macd > 0:
                reason = f"MACD crosses above zero: {current_macd:.4f} from {previous_macd:.4f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: MACD crosses below zero
            elif previous_macd >= 0 and current_macd < 0:
                reason = f"MACD crosses below zero: {current_macd:.4f} from {previous_macd:.4f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong buy signal: MACD is well above zero
            elif current_macd > 0.02:  # Threshold can be adjusted
                reason = f"MACD strongly positive: {current_macd:.4f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong sell signal: MACD is well below zero
            elif current_macd < -0.02:  # Threshold can be adjusted
                reason = f"MACD strongly negative: {current_macd:.4f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check MACD trend when near zero
            elif current_macd > 0:
                reason = f"MACD above zero: {current_macd:.4f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"MACD below zero: {current_macd:.4f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in MACD zero line crossover calculation: {str(e)}", data)
            return -1
