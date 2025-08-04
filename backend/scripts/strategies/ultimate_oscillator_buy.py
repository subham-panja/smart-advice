"""
Ultimate Oscillator Buy Strategy
File: scripts/strategies/ultimate_oscillator_buy.py

This strategy uses the Ultimate Oscillator to identify buy signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Ultimate_Oscillator_Buy(BaseStrategy):
    """
    Ultimate Oscillator Buy Strategy.
    
    Buy Signal: Ultimate Oscillator crosses above 30 (from oversold)
    Sell Signal: Ultimate Oscillator crosses below 70 (from overbought)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period1 = self.get_parameter('period1', 7)
        self.period2 = self.get_parameter('period2', 14)
        self.period3 = self.get_parameter('period3', 28)
        self.oversold_level = self.get_parameter('oversold_level', 30)
        self.overbought_level = self.get_parameter('overbought_level', 70)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Ultimate Oscillator buy strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=max(self.period1, self.period2, self.period3) + 1):
            return -1
            
        try:
            # Calculate Ultimate Oscillator using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            ultosc = ta.ULTOSC(
                high_prices, low_prices, close_prices,
                timeperiod1=self.period1,
                timeperiod2=self.period2,
                timeperiod3=self.period3
            )
            
            # Check if we have valid values
            if pd.isna(ultosc[-1]) or pd.isna(ultosc[-2]):
                self.log_signal(-1, "Insufficient data for Ultimate Oscillator calculation", data)
                return -1
            
            current_ultosc = ultosc[-1]
            previous_ultosc = ultosc[-2]
            
            # Buy signal: Ultimate Oscillator crosses above oversold level
            if previous_ultosc <= self.oversold_level and current_ultosc > self.oversold_level:
                reason = f"Ultimate Oscillator bullish crossover: {current_ultosc:.2f} crosses above {self.oversold_level}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: Ultimate Oscillator is deeply oversold
            elif current_ultosc < 20:
                reason = f"Ultimate Oscillator deeply oversold: {current_ultosc:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Ultimate Oscillator crosses below overbought level
            elif previous_ultosc >= self.overbought_level and current_ultosc < self.overbought_level:
                reason = f"Ultimate Oscillator bearish crossover: {current_ultosc:.2f} crosses below {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong sell signal: Ultimate Oscillator is deeply overbought
            elif current_ultosc > 80:
                reason = f"Ultimate Oscillator deeply overbought: {current_ultosc:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current level and trend
            elif current_ultosc > 50 and current_ultosc > previous_ultosc:
                reason = f"Ultimate Oscillator bullish: {current_ultosc:.2f}, rising"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_ultosc < 50 and current_ultosc < previous_ultosc:
                reason = f"Ultimate Oscillator bearish: {current_ultosc:.2f}, falling"
                self.log_signal(-1, reason, data)
                return -1
            
            # Default based on current level
            elif current_ultosc > 50:
                reason = f"Ultimate Oscillator above midline: {current_ultosc:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Ultimate Oscillator below midline: {current_ultosc:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Ultimate Oscillator calculation: {str(e)}", data)
            return -1
