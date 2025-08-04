"""
Aroon Oscillator Strategy
File: scripts/strategies/aroon_oscillator.py

This strategy uses the Aroon Oscillator to identify trend strength and direction.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Aroon_Oscillator(BaseStrategy):
    """
    Aroon Oscillator Strategy.
    
    Buy Signal: Aroon oscillator is positive and rising
    Sell Signal: Aroon oscillator is negative and falling
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 14)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Aroon Oscillator strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.period + 1):
            return -1
            
        try:
            # Calculate Aroon using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            aroon_down, aroon_up = ta.AROON(high_prices, low_prices, timeperiod=self.period)
            
            # Check if we have valid values
            if pd.isna(aroon_up[-1]) or pd.isna(aroon_down[-1]):
                self.log_signal(-1, "Insufficient data for Aroon calculation", data)
                return -1
            
            # Calculate Aroon Oscillator (Aroon Up - Aroon Down)
            aroon_oscillator = aroon_up - aroon_down
            
            current_oscillator = aroon_oscillator[-1]
            previous_oscillator = aroon_oscillator[-2] if len(aroon_oscillator) > 1 else current_oscillator
            
            # Buy signal: Aroon oscillator crosses above zero
            if previous_oscillator <= 0 and current_oscillator > 0:
                reason = f"Aroon oscillator turns positive: {current_oscillator:.2f} from {previous_oscillator:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: Aroon oscillator is strongly positive
            elif current_oscillator > 50:
                reason = f"Strong Aroon bullish trend: {current_oscillator:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Aroon oscillator crosses below zero
            elif previous_oscillator >= 0 and current_oscillator < 0:
                reason = f"Aroon oscillator turns negative: {current_oscillator:.2f} from {previous_oscillator:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong sell signal: Aroon oscillator is strongly negative
            elif current_oscillator < -50:
                reason = f"Strong Aroon bearish trend: {current_oscillator:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check trend
            elif current_oscillator > 0 and current_oscillator > previous_oscillator:
                reason = f"Rising positive Aroon: {current_oscillator:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_oscillator < 0 and current_oscillator < previous_oscillator:
                reason = f"Falling negative Aroon: {current_oscillator:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Default based on current value
            elif current_oscillator > 0:
                reason = f"Positive Aroon oscillator: {current_oscillator:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Negative Aroon oscillator: {current_oscillator:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Aroon oscillator calculation: {str(e)}", data)
            return -1
