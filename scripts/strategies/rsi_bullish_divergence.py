"""
RSI Bullish Divergence Strategy
File: scripts/strategies/rsi_bullish_divergence.py

This strategy identifies bullish divergence between price and RSI.
"""

import pandas as pd
import numpy as np
import talib as ta
from scipy.signal import argrelextrema
from .base_strategy import BaseStrategy

class RSI_Bullish_Divergence(BaseStrategy):
    """
    RSI Bullish Divergence Strategy.
    
    Buy Signal: Price makes lower lows while RSI makes higher lows (bullish divergence)
    Sell Signal: No divergence detected or bearish conditions
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.rsi_period = self.get_parameter('rsi_period', 14)
        self.lookback_period = self.get_parameter('lookback_period', 20)
        self.min_distance = self.get_parameter('min_distance', 5)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the RSI bullish divergence strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=max(self.rsi_period, self.lookback_period) + 10):
            return -1
            
        try:
            # Calculate RSI
            close_prices = data['Close'].values
            rsi = ta.RSI(close_prices, timeperiod=self.rsi_period)
            
            # Check if we have valid RSI values
            if pd.isna(rsi[-1]) or len(rsi) < self.lookback_period + 10:
                self.log_signal(-1, "Insufficient data for RSI divergence calculation", data)
                return -1
            
            # Get recent data for analysis
            recent_close = close_prices[-self.lookback_period:]
            recent_rsi = rsi[-self.lookback_period:]
            
            # Find local minima (lows) in both price and RSI
            price_lows = argrelextrema(recent_close, np.less, order=self.min_distance)[0]
            rsi_lows = argrelextrema(recent_rsi, np.less, order=self.min_distance)[0]
            
            # Need at least 2 lows for divergence analysis
            if len(price_lows) < 2 or len(rsi_lows) < 2:
                # Check if RSI is in oversold territory
                current_rsi = rsi[-1]
                if current_rsi < 30:
                    reason = f"RSI oversold: {current_rsi:.2f} (no divergence pattern yet)"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = "Insufficient data for divergence analysis"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Get the two most recent lows
            last_price_low_idx = price_lows[-1]
            second_last_price_low_idx = price_lows[-2] if len(price_lows) >= 2 else price_lows[-1]
            
            last_rsi_low_idx = rsi_lows[-1]
            second_last_rsi_low_idx = rsi_lows[-2] if len(rsi_lows) >= 2 else rsi_lows[-1]
            
            # Check for bullish divergence
            # Price: lower low, RSI: higher low
            price_lower_low = recent_close[last_price_low_idx] < recent_close[second_last_price_low_idx]
            rsi_higher_low = recent_rsi[last_rsi_low_idx] > recent_rsi[second_last_rsi_low_idx]
            
            if price_lower_low and rsi_higher_low:
                price_diff = recent_close[last_price_low_idx] - recent_close[second_last_price_low_idx]
                rsi_diff = recent_rsi[last_rsi_low_idx] - recent_rsi[second_last_rsi_low_idx]
                reason = f"Bullish RSI divergence: Price lower by {abs(price_diff):.2f}, RSI higher by {rsi_diff:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Check current RSI level
            current_rsi = rsi[-1]
            
            # Additional buy conditions
            if current_rsi < 35:  # Oversold region
                reason = f"RSI near oversold: {current_rsi:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Check for RSI momentum
            rsi_momentum = rsi[-1] - rsi[-5] if len(rsi) >= 5 else 0
            if current_rsi < 50 and rsi_momentum > 0:
                reason = f"RSI improving: {current_rsi:.2f}, momentum {rsi_momentum:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Default to sell/no signal
            reason = f"No bullish divergence: RSI {current_rsi:.2f}"
            self.log_signal(-1, reason, data)
            return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in RSI divergence calculation: {str(e)}", data)
            return -1
