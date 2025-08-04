"""
Keltner Channels Breakout Strategy
File: scripts/strategies/keltner_channels_breakout.py

This strategy uses Keltner Channels to identify breakout signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Keltner_Channels_Breakout(BaseStrategy):
    """
    Keltner Channels Breakout Strategy.
    
    Buy Signal: Price breaks above upper Keltner Channel
    Sell Signal: Price breaks below lower Keltner Channel
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 20)
        self.atr_multiplier = self.get_parameter('atr_multiplier', 2.0)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core Keltner Channels breakout strategy logic.
        Called by base class run_strategy method after volume filtering.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.period + 1):
            return -1
            
        try:
            # Calculate components for Keltner Channels
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            # Calculate EMA (middle line)
            ema = ta.EMA(close_prices, timeperiod=self.period)
            
            # Calculate ATR
            atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=self.period)
            
            # Check if we have valid values
            if pd.isna(ema[-1]) or pd.isna(atr[-1]):
                self.log_signal(-1, "Insufficient data for Keltner Channels calculation", data)
                return -1
            
            # Calculate Keltner Channels
            upper_channel = ema + (self.atr_multiplier * atr)
            lower_channel = ema - (self.atr_multiplier * atr)
            
            current_price = close_prices[-1]
            previous_price = close_prices[-2] if len(close_prices) > 1 else current_price
            
            current_upper = upper_channel[-1]
            current_lower = lower_channel[-1]
            current_middle = ema[-1]
            
            previous_upper = upper_channel[-2] if len(upper_channel) > 1 else current_upper
            previous_lower = lower_channel[-2] if len(lower_channel) > 1 else current_lower
            
            # Buy signal: Price breaks above upper channel
            if previous_price <= previous_upper and current_price > current_upper:
                reason = f"Bullish breakout: Price {current_price:.2f} breaks above upper channel {current_upper:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: Price is above upper channel
            elif current_price > current_upper:
                reason = f"Above upper channel: Price {current_price:.2f} > {current_upper:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Price breaks below lower channel
            elif previous_price >= previous_lower and current_price < current_lower:
                reason = f"Bearish breakdown: Price {current_price:.2f} breaks below lower channel {current_lower:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong sell signal: Price is below lower channel
            elif current_price < current_lower:
                reason = f"Below lower channel: Price {current_price:.2f} < {current_lower:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Price within channels - check position relative to middle
            elif current_price > current_middle:
                reason = f"Above middle line: Price {current_price:.2f} > EMA {current_middle:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Below middle line: Price {current_price:.2f} < EMA {current_middle:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Keltner Channels calculation: {str(e)}", data)
            return -1
