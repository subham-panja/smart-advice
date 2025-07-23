"""
RSI Overbought/Oversold Strategy
File: scripts/strategies/rsi_overbought_oversold.py

This strategy uses the Relative Strength Index (RSI) to identify overbought and oversold conditions.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class RSI_Overbought_Oversold(BaseStrategy):
    """
    RSI Overbought/Oversold Strategy.
    
    Buy Signal: RSI crosses above oversold level (typically 30)
    Sell Signal: RSI crosses below overbought level (typically 70)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.rsi_period = self.get_parameter('rsi_period', 14)
        self.oversold_level = self.get_parameter('oversold_level', 30)
        self.overbought_level = self.get_parameter('overbought_level', 70)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core RSI overbought/oversold strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.rsi_period + 1):
            return -1
            
        try:
            # Calculate RSI using TA-Lib
            close_prices = data['Close'].values
            rsi = ta.RSI(close_prices, timeperiod=self.rsi_period)
            
            # Check if we have valid RSI values
            if pd.isna(rsi[-1]) or pd.isna(rsi[-2]):
                self.log_signal(-1, "Insufficient data for RSI calculation", data)
                return -1
            
            current_rsi = rsi[-1]
            previous_rsi = rsi[-2]
            
            # Buy signal: RSI crosses above oversold level
            if previous_rsi <= self.oversold_level and current_rsi > self.oversold_level:
                reason = f"RSI recovery from oversold: {current_rsi:.2f} crosses above {self.oversold_level}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: RSI crosses below overbought level
            elif previous_rsi >= self.overbought_level and current_rsi < self.overbought_level:
                reason = f"RSI decline from overbought: {current_rsi:.2f} crosses below {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check if currently in oversold region (potential buy)
            elif current_rsi < self.oversold_level:
                reason = f"RSI oversold: {current_rsi:.2f} below {self.oversold_level}"
                self.log_signal(1, reason, data)
                return 1
            
            # Check if currently in overbought region (potential sell)
            elif current_rsi > self.overbought_level:
                reason = f"RSI overbought: {current_rsi:.2f} above {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # RSI in neutral zone - be more conservative
            elif current_rsi >= 60:  # Only bullish if RSI > 60 (stronger signal)
                reason = f"RSI bullish: {current_rsi:.2f} above 60"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_rsi <= 40:  # Only bearish if RSI < 40 (stronger signal)
                reason = f"RSI bearish: {current_rsi:.2f} below 40"
                self.log_signal(-1, reason, data)
                return -1
            
            # RSI in neutral zone (40-60) - no clear signal
            else:
                reason = f"RSI neutral: {current_rsi:.2f} (no clear signal)"
                self.log_signal(-1, reason, data)  # Conservative: no signal = bearish
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in RSI calculation: {str(e)}", data)
            return -1
