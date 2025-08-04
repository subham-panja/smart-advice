"""
Volume Price Trend Strategy
File: scripts/strategies/volume_price_trend.py

This strategy uses the Volume Price Trend indicator to determine buying and selling signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Volume_Price_Trend(BaseStrategy):
    """
    Volume Price Trend Strategy.
    
    Buy Signal: Positive volume price trend
    Sell Signal: Negative volume price trend
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Volume Price Trend strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=2):
            return -1
            
        try:
            # Calculate Volume Price Trend manually
            close_prices = data['Close'].values
            volume = data['Volume'].values
            
            # Calculate VPT: VPT = Previous VPT + Volume * ((Close - Previous Close) / Previous Close)
            vpt = np.zeros(len(close_prices))
            vpt[0] = 0  # Initial VPT value
            
            for i in range(1, len(close_prices)):
                if close_prices[i-1] != 0:
                    price_change_pct = (close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                    vpt[i] = vpt[i-1] + volume[i] * price_change_pct
                else:
                    vpt[i] = vpt[i-1]
            
            # Check if we have valid VPT values
            if len(vpt) < 2 or pd.isna(vpt[-1]):
                self.log_signal(-1, "Insufficient data for VPT calculation", data)
                return -1
            
            current_vpt = vpt[-1]
            previous_vpt = vpt[-2]
            
            # Buy signal: VPT is rising
            if current_vpt > previous_vpt and current_vpt > 0:
                reason = f"Rising VPT: {current_vpt:.2f} > {previous_vpt:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: VPT is falling
            elif current_vpt < previous_vpt:
                reason = f"Falling VPT: {current_vpt:.2f} < {previous_vpt:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # VPT is positive and stable
            elif current_vpt > 0:
                reason = f"Positive VPT: {current_vpt:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Negative VPT: {current_vpt:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in VPT calculation: {str(e)}", data)
            return -1
