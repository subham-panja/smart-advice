"""
Rate of Change (ROC) Strategy
File: scripts/strategies/roc_rate_of_change.py

This strategy uses Rate of Change indicator to identify momentum shifts.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class ROC_Rate_of_Change(BaseStrategy):
    """
    Rate of Change (ROC) Strategy.
    
    Buy Signal: ROC crosses above zero or shows strong positive momentum
    Sell Signal: ROC crosses below zero or shows negative momentum
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 10)
        self.threshold = self.get_parameter('threshold', 0)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core Rate of Change strategy logic.
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
            # Calculate Rate of Change using TA-Lib
            close_prices = data['Close'].values
            roc = ta.ROC(close_prices, timeperiod=self.period)
            
            # Check if we have valid ROC values
            if pd.isna(roc[-1]) or pd.isna(roc[-2]):
                self.log_signal(-1, "Insufficient data for ROC calculation", data)
                return -1
            
            current_roc = roc[-1]
            previous_roc = roc[-2]
            
            # Buy signal: ROC crosses above threshold
            if previous_roc <= self.threshold and current_roc > self.threshold:
                reason = f"ROC crosses above {self.threshold}: {current_roc:.2f}% from {previous_roc:.2f}%"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: ROC is significantly positive
            elif current_roc > 5.0:  # 5% positive ROC
                reason = f"Strong positive ROC: {current_roc:.2f}%"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: ROC crosses below threshold
            elif previous_roc >= self.threshold and current_roc < self.threshold:
                reason = f"ROC crosses below {self.threshold}: {current_roc:.2f}% from {previous_roc:.2f}%"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong sell signal: ROC is significantly negative
            elif current_roc < -5.0:  # -5% negative ROC
                reason = f"Strong negative ROC: {current_roc:.2f}%"
                self.log_signal(-1, reason, data)
                return -1
            
            # Moderate signals based on ROC value
            elif current_roc > 0:
                reason = f"Positive ROC: {current_roc:.2f}%"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Negative/neutral ROC: {current_roc:.2f}%"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in ROC calculation: {str(e)}", data)
            return -1
