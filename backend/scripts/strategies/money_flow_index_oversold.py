"""
Money Flow Index Oversold Strategy
File: scripts/strategies/money_flow_index_oversold.py

This strategy uses the Money Flow Index to identify oversold conditions for buy signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Money_Flow_Index_Oversold(BaseStrategy):
    """
    Money Flow Index Oversold Strategy.
    
    Buy Signal: MFI crosses above oversold level (typically 20)
    Sell Signal: MFI crosses below overbought level (typically 80)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 14)
        self.oversold_level = self.get_parameter('oversold_level', 20)
        self.overbought_level = self.get_parameter('overbought_level', 80)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Money Flow Index oversold strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.period + 1):
            return -1
            
        try:
            # Calculate Money Flow Index using TA-Lib
            high_prices = data['High'].values.astype(float)
            low_prices = data['Low'].values.astype(float)
            close_prices = data['Close'].values.astype(float)
            volume = data['Volume'].values.astype(float)
            
            mfi = ta.MFI(high_prices, low_prices, close_prices, volume, timeperiod=self.period)
            
            # Check if we have valid values
            if pd.isna(mfi[-1]) or pd.isna(mfi[-2]):
                self.log_signal(-1, "Insufficient data for MFI calculation", data)
                return -1
            
            current_mfi = mfi[-1]
            previous_mfi = mfi[-2]
            
            # Buy signal: MFI crosses above oversold level
            if previous_mfi <= self.oversold_level and current_mfi > self.oversold_level:
                reason = f"MFI bullish crossover: {current_mfi:.2f} crosses above {self.oversold_level}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: MFI is deeply oversold
            elif current_mfi < 10:
                reason = f"MFI deeply oversold: {current_mfi:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: MFI crosses below overbought level
            elif previous_mfi >= self.overbought_level and current_mfi < self.overbought_level:
                reason = f"MFI bearish crossover: {current_mfi:.2f} crosses below {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong sell signal: MFI is deeply overbought
            elif current_mfi > 90:
                reason = f"MFI deeply overbought: {current_mfi:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current level and trend
            elif current_mfi < self.oversold_level:
                reason = f"MFI oversold: {current_mfi:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_mfi > self.overbought_level:
                reason = f"MFI overbought: {current_mfi:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # MFI in neutral zone - check trend
            elif current_mfi > 50 and current_mfi > previous_mfi:
                reason = f"MFI bullish: {current_mfi:.2f}, rising"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_mfi < 50 and current_mfi < previous_mfi:
                reason = f"MFI bearish: {current_mfi:.2f}, falling"
                self.log_signal(-1, reason, data)
                return -1
            
            # Default based on current level
            elif current_mfi > 50:
                reason = f"MFI above midline: {current_mfi:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"MFI below midline: {current_mfi:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in MFI calculation: {str(e)}", data)
            return -1
