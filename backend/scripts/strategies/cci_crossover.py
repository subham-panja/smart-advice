"""
CCI (Commodity Channel Index) Crossover Strategy
File: scripts/strategies/cci_crossover.py

This strategy uses CCI crossovers to identify overbought/oversold conditions.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class CCI_Crossover(BaseStrategy):
    """
    CCI Crossover Strategy.
    
    Buy Signal: CCI crosses above -100 (from oversold)
    Sell Signal: CCI crosses below +100 (from overbought)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 20)
        self.overbought_level = self.get_parameter('overbought_level', 100)
        self.oversold_level = self.get_parameter('oversold_level', -100)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core CCI crossover strategy logic.
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
            # Calculate CCI using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            cci = ta.CCI(high_prices, low_prices, close_prices, timeperiod=self.period)
            
            # Check if we have valid values
            if pd.isna(cci[-1]) or pd.isna(cci[-2]):
                self.log_signal(-1, "Insufficient data for CCI calculation", data)
                return -1
            
            current_cci = cci[-1]
            previous_cci = cci[-2]
            
            # Buy signal: CCI crosses above oversold level
            if previous_cci <= self.oversold_level and current_cci > self.oversold_level:
                reason = f"CCI bullish crossover: {current_cci:.2f} crosses above {self.oversold_level}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: CCI crosses below overbought level
            elif previous_cci >= self.overbought_level and current_cci < self.overbought_level:
                reason = f"CCI bearish crossover: {current_cci:.2f} crosses below {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong buy signal: CCI is deeply oversold
            elif current_cci < -200:
                reason = f"CCI deeply oversold: {current_cci:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong sell signal: CCI is deeply overbought
            elif current_cci > 200:
                reason = f"CCI deeply overbought: {current_cci:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check CCI trend and position
            elif current_cci > 0 and current_cci > previous_cci:
                reason = f"CCI positive and rising: {current_cci:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_cci < 0 and current_cci < previous_cci:
                reason = f"CCI negative and falling: {current_cci:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Neutral zone with trend
            elif current_cci > 0:
                reason = f"CCI positive: {current_cci:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"CCI negative: {current_cci:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in CCI calculation: {str(e)}", data)
            return -1
