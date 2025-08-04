"""
Chaikin Oscillator Strategy
File: scripts/strategies/chaikin_oscillator.py

This strategy uses the Chaikin Oscillator to identify momentum changes.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Chaikin_Oscillator(BaseStrategy):
    """
    Chaikin Oscillator Strategy.
    
    Buy Signal: Chaikin Oscillator crosses above zero
    Sell Signal: Chaikin Oscillator crosses below zero
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 3)
        self.slow_period = self.get_parameter('slow_period', 10)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Chaikin Oscillator strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=max(self.fast_period, self.slow_period) + 1):
            return -1
            
        try:
            # Calculate Chaikin Oscillator using TA-Lib
            high_prices = data['High'].values.astype(float)
            low_prices = data['Low'].values.astype(float)
            close_prices = data['Close'].values.astype(float)
            volume = data['Volume'].values.astype(float)
            
            chaikin_osc = ta.ADOSC(high_prices, low_prices, close_prices, volume,
                                  fastperiod=self.fast_period, slowperiod=self.slow_period)
            
            # Check if we have valid values
            if pd.isna(chaikin_osc[-1]) or pd.isna(chaikin_osc[-2]):
                self.log_signal(-1, "Insufficient data for Chaikin Oscillator calculation", data)
                return -1
            
            current_chaikin = chaikin_osc[-1]
            previous_chaikin = chaikin_osc[-2]
            
            # Buy signal: Chaikin Oscillator crosses above zero
            if previous_chaikin <= 0 and current_chaikin > 0:
                reason = f"Chaikin bullish crossover: {current_chaikin:.0f} crosses above zero"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Chaikin Oscillator crosses below zero
            elif previous_chaikin >= 0 and current_chaikin < 0:
                reason = f"Chaikin bearish crossover: {current_chaikin:.0f} crosses below zero"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong buy signal: Strongly positive and increasing
            elif current_chaikin > 100000 and current_chaikin > previous_chaikin:
                reason = f"Strong Chaikin momentum: {current_chaikin:.0f}, increasing"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong sell signal: Strongly negative and decreasing
            elif current_chaikin < -100000 and current_chaikin < previous_chaikin:
                reason = f"Strong negative Chaikin: {current_chaikin:.0f}, decreasing"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check trend and momentum
            elif current_chaikin > 0 and current_chaikin > previous_chaikin:
                reason = f"Positive Chaikin momentum: {current_chaikin:.0f}, rising"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_chaikin < 0 and current_chaikin < previous_chaikin:
                reason = f"Negative Chaikin momentum: {current_chaikin:.0f}, falling"
                self.log_signal(-1, reason, data)
                return -1
            
            # Default based on current position
            elif current_chaikin > 0:
                reason = f"Positive Chaikin: {current_chaikin:.0f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Negative Chaikin: {current_chaikin:.0f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Chaikin Oscillator calculation: {str(e)}", data)
            return -1
