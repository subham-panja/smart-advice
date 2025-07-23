"""
Bollinger Bands Breakout Strategy
File: scripts/strategies/bollinger_band_breakout.py

This strategy uses Bollinger Bands to identify breakout opportunities.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Bollinger_Band_Breakout(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy.
    
    Buy Signal: Price breaks above upper Bollinger Band
    Sell Signal: Price breaks below lower Bollinger Band
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 20)
        self.std_dev = self.get_parameter('std_dev', 2.0)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Bollinger Bands breakout strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.period + 1):
            return -1
            
        try:
            # Calculate Bollinger Bands using TA-Lib
            close_prices = data['Close'].values
            upper_band, middle_band, lower_band = ta.BBANDS(
                close_prices,
                timeperiod=self.period,
                nbdevup=self.std_dev,
                nbdevdn=self.std_dev,
                matype=0  # Simple Moving Average
            )
            
            # Check if we have valid Bollinger Band values
            if (pd.isna(upper_band[-1]) or pd.isna(middle_band[-1]) or 
                pd.isna(lower_band[-1]) or pd.isna(upper_band[-2]) or 
                pd.isna(middle_band[-2]) or pd.isna(lower_band[-2])):
                self.log_signal(-1, "Insufficient data for Bollinger Bands calculation", data)
                return -1
            
            current_close = close_prices[-1]
            previous_close = close_prices[-2]
            current_upper = upper_band[-1]
            current_middle = middle_band[-1]
            current_lower = lower_band[-1]
            previous_upper = upper_band[-2]
            previous_lower = lower_band[-2]
            
            # Buy signal: Price breaks above upper Bollinger Band
            if previous_close <= previous_upper and current_close > current_upper:
                reason = f"Bollinger upward breakout: Price ({current_close:.2f}) breaks above upper band ({current_upper:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Price breaks below lower Bollinger Band
            elif previous_close >= previous_lower and current_close < current_lower:
                reason = f"Bollinger Bands downward breakout: Price ({current_close:.2f}) breaks below lower band ({current_lower:.2f})"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check position relative to middle band
            elif current_close > current_middle:
                # Above middle band - bullish bias
                distance_to_upper = (current_upper - current_close) / (current_upper - current_middle)
                if distance_to_upper > 0.5:  # Not too close to upper band
                    reason = f"Above middle band: Price ({current_close:.2f}) above middle ({current_middle:.2f})"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Near upper band: Price ({current_close:.2f}) close to upper band ({current_upper:.2f})"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Below middle band - bearish bias
            elif current_close < current_middle:
                distance_to_lower = (current_close - current_lower) / (current_middle - current_lower)
                if distance_to_lower > 0.5:  # Not too close to lower band
                    reason = f"Below middle band: Price ({current_close:.2f}) below middle ({current_middle:.2f})"
                    self.log_signal(-1, reason, data)
                    return -1
                else:
                    # Near lower band - potential reversal opportunity
                    reason = f"Near lower band: Price ({current_close:.2f}) close to lower band ({current_lower:.2f})"
                    self.log_signal(1, reason, data)
                    return 1
            
            # At middle band - neutral
            else:
                reason = f"At middle band: Price ({current_close:.2f}) at middle ({current_middle:.2f})"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Bollinger Bands calculation: {str(e)}", data)
            return -1
