"""
Parabolic SAR Reversal Strategy
File: scripts/strategies/parabolic_sar_reversal.py

This strategy uses Parabolic SAR reversals to identify trend changes.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Parabolic_SAR_Reversal(BaseStrategy):
    """
    Parabolic SAR Reversal Strategy.
    
    Buy Signal: Price crosses above Parabolic SAR (trend reversal to upside)
    Sell Signal: Price crosses below Parabolic SAR (trend reversal to downside)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.acceleration = self.get_parameter('acceleration', 0.02)
        self.maximum = self.get_parameter('maximum', 0.2)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Parabolic SAR reversal strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=10):
            return -1
            
        try:
            # Calculate Parabolic SAR using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            sar = ta.SAR(high_prices, low_prices, 
                        acceleration=self.acceleration, maximum=self.maximum)
            
            # Check if we have valid values
            if pd.isna(sar[-1]) or pd.isna(sar[-2]) or len(sar) < 2:
                self.log_signal(-1, "Insufficient data for Parabolic SAR calculation", data)
                return -1
            
            current_price = close_prices[-1]
            previous_price = close_prices[-2]
            current_sar = sar[-1]
            previous_sar = sar[-2]
            
            # Buy signal: Price crosses above SAR (bullish reversal)
            if previous_price <= previous_sar and current_price > current_sar:
                reason = f"Bullish SAR reversal: Price {current_price:.2f} crosses above SAR {current_sar:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Price crosses below SAR (bearish reversal)
            elif previous_price >= previous_sar and current_price < current_sar:
                reason = f"Bearish SAR reversal: Price {current_price:.2f} crosses below SAR {current_sar:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong buy signal: Price well above SAR (strong uptrend)
            elif current_price > current_sar and (current_price - current_sar) / current_price > 0.05:
                reason = f"Strong uptrend: Price {current_price:.2f} well above SAR {current_sar:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong sell signal: Price well below SAR (strong downtrend)
            elif current_price < current_sar and (current_sar - current_price) / current_price > 0.05:
                reason = f"Strong downtrend: Price {current_price:.2f} well below SAR {current_sar:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Current trend based on position relative to SAR
            elif current_price > current_sar:
                reason = f"Uptrend: Price {current_price:.2f} above SAR {current_sar:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Downtrend: Price {current_price:.2f} below SAR {current_sar:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Parabolic SAR calculation: {str(e)}", data)
            return -1
