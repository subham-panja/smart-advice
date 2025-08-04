"""
On Balance Volume (OBV) Strategy
File: scripts/strategies/on_balance_volume.py

This strategy uses the On Balance Volume indicator to identify volume-based buying and selling signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class On_Balance_Volume(BaseStrategy):
    """
    On Balance Volume (OBV) Strategy.
    
    Buy Signal: OBV is rising (confirms price trend)
    Sell Signal: OBV is falling (divergence or weakness)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback_period = self.get_parameter('lookback_period', 3)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the On Balance Volume strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.lookback_period + 1):
            return -1
            
        try:
            # Calculate On Balance Volume using TA-Lib
            close_prices = data['Close'].values.astype(float)
            volume = data['Volume'].values.astype(float)
            
            obv = ta.OBV(close_prices, volume)
            
            # Check if we have valid OBV values
            if pd.isna(obv[-1]) or len(obv) < self.lookback_period + 1:
                self.log_signal(-1, "Insufficient data for OBV calculation", data)
                return -1
            
            # Calculate OBV trend over lookback period
            current_obv = obv[-1]
            previous_obv = obv[-(self.lookback_period + 1)]
            obv_trend = current_obv - previous_obv
            
            # Calculate price trend over the same period
            current_price = close_prices[-1]
            previous_price = close_prices[-(self.lookback_period + 1)]
            price_trend = current_price - previous_price
            
            # Buy signal: OBV and price both rising (confirmation)
            if obv_trend > 0 and price_trend > 0:
                reason = f"OBV confirming price rise: OBV trend {obv_trend:.0f}, Price trend {price_trend:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: OBV rising faster than price (accumulation)
            elif obv_trend > 0 and price_trend <= 0:
                reason = f"OBV shows accumulation: OBV rising {obv_trend:.0f} while price flat/down {price_trend:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: OBV falling while price rising (divergence)
            elif obv_trend < 0 and price_trend > 0:
                reason = f"OBV divergence: Price rising {price_trend:.2f} but OBV falling {obv_trend:.0f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Sell signal: Both OBV and price falling
            elif obv_trend < 0 and price_trend < 0:
                reason = f"OBV confirming price decline: OBV trend {obv_trend:.0f}, Price trend {price_trend:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Neutral case: check recent OBV direction
            else:
                recent_obv_change = obv[-1] - obv[-2]
                if recent_obv_change > 0:
                    reason = f"Recent OBV rise: {recent_obv_change:.0f}"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Recent OBV decline: {recent_obv_change:.0f}"
                    self.log_signal(-1, reason, data)
                    return -1
                    
        except Exception as e:
            self.log_signal(-1, f"Error in OBV calculation: {str(e)}", data)
            return -1
