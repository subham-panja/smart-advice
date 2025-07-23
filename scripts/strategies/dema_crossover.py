"""
DEMA (Double Exponential Moving Average) Crossover Strategy
File: scripts/strategies/dema_crossover.py

This strategy uses the DEMA crossover to identify buy/sell signals.
DEMA is designed to reduce lag compared to traditional EMA.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class DEMA_Crossover(BaseStrategy):
    """
    DEMA (Double Exponential Moving Average) Crossover Strategy.
    
    Buy Signal: Fast DEMA crosses above Slow DEMA
    Sell Signal: Fast DEMA crosses below Slow DEMA
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter('fast_period', 12)
        self.slow_period = self.get_parameter('slow_period', 26)
        
    def calculate_dema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Calculate Double Exponential Moving Average (DEMA).
        DEMA = 2 * EMA(period) - EMA(EMA(period))
        """
        try:
            # Use TA-Lib DEMA if available
            return pd.Series(ta.DEMA(data.values, timeperiod=period), index=data.index)
        except:
            # Fallback manual calculation
            ema1 = data.ewm(span=period).mean()
            ema2 = ema1.ewm(span=period).mean()
            return 2 * ema1 - ema2
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core DEMA crossover strategy logic.
        Called by base class run_strategy method after volume filtering.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        min_periods = max(self.fast_period, self.slow_period) + 5
        if not self.validate_data(data, min_periods=min_periods):
            return -1
            
        try:
            # Calculate DEMAs
            close_prices = data['Close']
            fast_dema = self.calculate_dema(close_prices, self.fast_period)
            slow_dema = self.calculate_dema(close_prices, self.slow_period)
            
            # Check if we have valid DEMA values
            if pd.isna(fast_dema.iloc[-1]) or pd.isna(slow_dema.iloc[-1]):
                self.log_signal(-1, "Insufficient data for DEMA calculation", data)
                return -1
            
            if pd.isna(fast_dema.iloc[-2]) or pd.isna(slow_dema.iloc[-2]):
                self.log_signal(-1, "Insufficient historical DEMA data", data)
                return -1
            
            current_fast = fast_dema.iloc[-1]
            current_slow = slow_dema.iloc[-1]
            previous_fast = fast_dema.iloc[-2]
            previous_slow = slow_dema.iloc[-2]
            
            # Buy signal: Fast DEMA crosses above Slow DEMA
            if previous_fast <= previous_slow and current_fast > current_slow:
                reason = f"DEMA bullish crossover: Fast({current_fast:.2f}) crosses above Slow({current_slow:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong bullish signal: Fast DEMA significantly above Slow DEMA and rising
            elif current_fast > current_slow * 1.01 and current_fast > previous_fast:
                reason = f"DEMA strong bullish: Fast({current_fast:.2f}) >> Slow({current_slow:.2f}) and rising"
                self.log_signal(1, reason, data)
                return 1
            
            # Moderate bullish signal: Fast DEMA above Slow DEMA
            elif current_fast > current_slow:
                reason = f"DEMA bullish: Fast({current_fast:.2f}) > Slow({current_slow:.2f})"
                self.log_signal(1, reason, data)
                return 1
            
            # Bearish condition: Fast DEMA below Slow DEMA
            else:
                reason = f"DEMA bearish: Fast({current_fast:.2f}) < Slow({current_slow:.2f})"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in DEMA calculation: {str(e)}", data)
            return -1
