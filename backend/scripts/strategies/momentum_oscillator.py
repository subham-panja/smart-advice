"""
Momentum Oscillator Strategy
File: scripts/strategies/momentum_oscillator.py

This strategy uses momentum oscillator to identify momentum-based buying and selling signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Momentum_Oscillator(BaseStrategy):
    """
    Momentum Oscillator Strategy.
    
    Buy Signal: Momentum is positive and increasing
    Sell Signal: Momentum is negative or decreasing
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 10)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Momentum Oscillator strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.period + 1):
            return -1
            
        try:
            # Calculate Momentum using TA-Lib
            close_prices = data['Close'].values
            momentum = ta.MOM(close_prices, timeperiod=self.period)
            
            # Check if we have valid momentum values
            if pd.isna(momentum[-1]) or pd.isna(momentum[-2]):
                self.log_signal(-1, "Insufficient data for Momentum calculation", data)
                return -1
            
            current_momentum = momentum[-1]
            previous_momentum = momentum[-2]
            
            # Calculate momentum change
            momentum_change = current_momentum - previous_momentum
            
            # Buy signal: Positive momentum that's increasing
            if current_momentum > 0 and momentum_change > 0:
                reason = f"Strong positive momentum: {current_momentum:.2f}, increasing by {momentum_change:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Moderate buy signal: Momentum turning positive
            elif current_momentum > 0 and previous_momentum <= 0:
                reason = f"Momentum turning positive: {current_momentum:.2f} from {previous_momentum:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Weak buy signal: Positive momentum but decreasing
            elif current_momentum > 0 and momentum_change <= 0:
                reason = f"Weakening positive momentum: {current_momentum:.2f}, change {momentum_change:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Sell signal: Negative momentum
            elif current_momentum < 0:
                reason = f"Negative momentum: {current_momentum:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Neutral momentum
            else:
                reason = f"Neutral momentum: {current_momentum:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Momentum calculation: {str(e)}", data)
            return -1
