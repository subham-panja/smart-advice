"""
ADX Trend Strength Strategy
File: scripts/strategies/adx_trend_strength.py

This strategy uses the Average Directional Index (ADX) to identify trend strength.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class ADX_Trend_Strength(BaseStrategy):
    """
    ADX Trend Strength Strategy.
    
    Buy Signal: ADX above threshold with +DI > -DI (strong uptrend)
    Sell Signal: ADX above threshold with -DI > +DI (strong downtrend)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.adx_period = self.get_parameter('adx_period', 14)
        self.adx_threshold = self.get_parameter('adx_threshold', 25)
        self.strong_trend_threshold = self.get_parameter('strong_trend_threshold', 30)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the ADX trend strength strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.adx_period):
            return -1
            
        try:
            # Calculate ADX and DI indicators using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            # Calculate ADX, +DI, and -DI
            adx = ta.ADX(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            plus_di = ta.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            minus_di = ta.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=self.adx_period)
            
            # Check if we have valid values for the latest periods
            if (pd.isna(adx[-1]) or pd.isna(plus_di[-1]) or pd.isna(minus_di[-1])):
                self.log_signal(-1, "Insufficient data for ADX calculation", data)
                return -1
            
            current_adx = adx[-1]
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]
            
            # Check for strong uptrend
            if (current_adx > self.adx_threshold and 
                current_plus_di > current_minus_di):
                
                if current_adx > self.strong_trend_threshold:
                    reason = f"Strong uptrend: ADX ({current_adx:.2f}) > {self.strong_trend_threshold}, +DI ({current_plus_di:.2f}) > -DI ({current_minus_di:.2f})"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Moderate uptrend: ADX ({current_adx:.2f}) > {self.adx_threshold}, +DI ({current_plus_di:.2f}) > -DI ({current_minus_di:.2f})"
                    self.log_signal(1, reason, data)
                    return 1
            
            # Check for strong downtrend
            elif (current_adx > self.adx_threshold and 
                  current_minus_di > current_plus_di):
                reason = f"Strong downtrend: ADX ({current_adx:.2f}) > {self.adx_threshold}, -DI ({current_minus_di:.2f}) > +DI ({current_plus_di:.2f})"
                self.log_signal(-1, reason, data)
                return -1
            
            # Weak trend or sideways movement
            else:
                reason = f"Weak trend: ADX ({current_adx:.2f}) <= {self.adx_threshold}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in ADX calculation: {str(e)}", data)
            return -1
