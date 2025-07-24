"""
Commodity Channel Index (CCI) Strategy
File: scripts/strategies/commodity_channel_index.py

This strategy uses the Commodity Channel Index to identify overbought/oversold conditions
and potential reversal points. CCI measures the relationship between price and its moving average.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy


class Commodity_Channel_Index(BaseStrategy):
    """
    Strategy based on Commodity Channel Index (CCI).
    
    CCI signals:
    - CCI > +100: Overbought condition (potential sell)
    - CCI < -100: Oversold condition (potential buy)
    - CCI crossing above -100: Buy signal
    - CCI crossing below +100: Sell signal
    - Divergences between CCI and price action
    """
    
    def __init__(self, params=None):
        """
        Initialize the CCI strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - period: CCI calculation period (default: 20)
                   - oversold_level: Oversold threshold (default: -100)
                   - overbought_level: Overbought threshold (default: 100)
                   - extreme_oversold: Extreme oversold level (default: -200)
                   - extreme_overbought: Extreme overbought level (default: 200)
        """
        super().__init__(params)
        self.period = self.get_parameter('period', 20)
        self.oversold_level = self.get_parameter('oversold_level', -100)
        self.overbought_level = self.get_parameter('overbought_level', 100)
        self.extreme_oversold = self.get_parameter('extreme_oversold', -200)
        self.extreme_overbought = self.get_parameter('extreme_overbought', 200)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the CCI strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=self.period + 10):
            self.log_signal(-1, "Insufficient data for CCI analysis", data)
            return -1
        
        try:
            # Calculate CCI using TA-Lib
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            cci = ta.CCI(high, low, close, timeperiod=self.period)
            
            if len(cci) < 3 or np.isnan(cci[-1]) or np.isnan(cci[-2]):
                self.log_signal(-1, "Insufficient CCI data", data)
                return -1
            
            current_cci = cci[-1]
            prev_cci = cci[-2]
            prev2_cci = cci[-3] if len(cci) > 2 else prev_cci
            
            # Check for extreme conditions first
            if current_cci < self.extreme_oversold:
                # Extremely oversold - strong buy signal
                self.log_signal(1, f"Extreme oversold CCI: {current_cci:.2f} < {self.extreme_oversold}", data)
                return 1
            
            if current_cci > self.extreme_overbought:
                # Extremely overbought - avoid buying
                self.log_signal(-1, f"Extreme overbought CCI: {current_cci:.2f} > {self.extreme_overbought}", data)
                return -1
            
            # Check for crossing signals
            # Buy signal: CCI crossing above oversold level
            if prev_cci <= self.oversold_level and current_cci > self.oversold_level:
                self.log_signal(1, f"CCI bullish crossover: {prev_cci:.2f} -> {current_cci:.2f} above {self.oversold_level}", data)
                return 1
            
            # Sell signal: CCI crossing below overbought level
            if prev_cci >= self.overbought_level and current_cci < self.overbought_level:
                self.log_signal(-1, f"CCI bearish crossover: {prev_cci:.2f} -> {current_cci:.2f} below {self.overbought_level}", data)
                return -1
            
            # Check for oversold bounce
            if current_cci < self.oversold_level and current_cci > prev_cci:
                # CCI is oversold but starting to turn up
                self.log_signal(1, f"CCI oversold bounce: {current_cci:.2f} turning up from oversold", data)
                return 1
            
            # Check for momentum
            if current_cci > prev_cci > prev2_cci and current_cci > -50:
                # Positive momentum and not too negative
                self.log_signal(1, f"CCI positive momentum: {prev2_cci:.2f} -> {prev_cci:.2f} -> {current_cci:.2f}", data)
                return 1
            
            # Check for overbought conditions
            if current_cci > self.overbought_level:
                self.log_signal(-1, f"CCI overbought: {current_cci:.2f} > {self.overbought_level}", data)
                return -1
            
            # Check for negative momentum
            if current_cci < prev_cci < prev2_cci:
                self.log_signal(-1, f"CCI negative momentum: {prev2_cci:.2f} -> {prev_cci:.2f} -> {current_cci:.2f}", data)
                return -1
            
            # Neutral/hold signal
            self.log_signal(-1, f"CCI neutral: {current_cci:.2f} (no clear signal)", data)
            return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in CCI analysis: {str(e)}", data)
            return -1
