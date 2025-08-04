"""
Williams %R Overbought/Oversold Strategy
File: scripts/strategies/williams_percent_r_strategy.py

This strategy uses the Williams %R indicator to identify overbought and oversold conditions.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Williams_Percent_R_Overbought_Oversold(BaseStrategy):
    """
    Williams %R Overbought/Oversold Strategy.
    
    Buy Signal: Williams %R crosses above oversold level (typically -80)
    Sell Signal: Williams %R crosses below overbought level (typically -20)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 14)
        self.oversold_level = self.get_parameter('oversold_level', -80)
        self.overbought_level = self.get_parameter('overbought_level', -20)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Williams %R overbought/oversold strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.period + 1):
            return -1
            
        try:
            # Calculate Williams %R using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            will_r = ta.WILLR(high_prices, low_prices, close_prices, timeperiod=self.period)
            
            # Check if we have valid Williams %R values
            if pd.isna(will_r[-1]) or pd.isna(will_r[-2]):
                self.log_signal(-1, "Insufficient data for Williams %R calculation", data)
                return -1
            
            current_will_r = will_r[-1]
            previous_will_r = will_r[-2]
            
            # Buy signal: Williams %R crosses above oversold level
            if previous_will_r <= self.oversold_level and current_will_r > self.oversold_level:
                reason = f"Williams %R recovery from oversold: {current_will_r:.2f} crosses above {self.oversold_level}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Williams %R crosses below overbought level
            elif previous_will_r >= self.overbought_level and current_will_r < self.overbought_level:
                reason = f"Williams %R decline from overbought: {current_will_r:.2f} crosses below {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check if currently in oversold region (potential buy)
            elif current_will_r < self.oversold_level:
                reason = f"Williams %R oversold: {current_will_r:.2f} below {self.oversold_level}"
                self.log_signal(1, reason, data)
                return 1
            
            # Check if currently in overbought region (potential sell)
            elif current_will_r > self.overbought_level:
                reason = f"Williams %R overbought: {current_will_r:.2f} above {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Williams %R in neutral zone
            elif current_will_r >= -50:
                reason = f"Williams %R neutral-bullish: {current_will_r:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Williams %R bearish: {current_will_r:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Williams %R calculation: {str(e)}", data)
            return -1
