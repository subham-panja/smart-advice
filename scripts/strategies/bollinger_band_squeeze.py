"""
Bollinger Band Squeeze Strategy
File: scripts/strategies/bollinger_band_squeeze.py

This strategy identifies Bollinger Band squeezes and subsequent breakouts.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Bollinger_Band_Squeeze(BaseStrategy):
    """
    Bollinger Band Squeeze Strategy.
    
    Buy Signal: Bollinger Bands are squeezing (low volatility) with upward breakout
    Sell Signal: High volatility or downward breakout
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.bb_period = self.get_parameter('bb_period', 20)
        self.bb_std = self.get_parameter('bb_std', 2.0)
        self.kc_period = self.get_parameter('kc_period', 20)
        self.atr_multiplier = self.get_parameter('atr_multiplier', 1.5)
        self.squeeze_threshold = self.get_parameter('squeeze_threshold', 0.95)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core Bollinger Band Squeeze strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=max(self.bb_period, self.kc_period) + 1):
            return -1
            
        try:
            # Calculate Bollinger Bands
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values
            
            bb_upper, bb_middle, bb_lower = ta.BBANDS(
                close_prices, 
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0
            )
            
            # Calculate Keltner Channels for squeeze detection
            ema = ta.EMA(close_prices, timeperiod=self.kc_period)
            atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=self.kc_period)
            kc_upper = ema + (self.atr_multiplier * atr)
            kc_lower = ema - (self.atr_multiplier * atr)
            
            # Check if we have valid values
            if (pd.isna(bb_upper[-1]) or pd.isna(bb_lower[-1]) or 
                pd.isna(kc_upper[-1]) or pd.isna(kc_lower[-1])):
                self.log_signal(-1, "Insufficient data for Bollinger Band Squeeze calculation", data)
                return -1
            
            # Calculate squeeze condition
            # Squeeze occurs when Bollinger Bands are inside Keltner Channels
            squeeze_ratio = (bb_upper[-1] - bb_lower[-1]) / (kc_upper[-1] - kc_lower[-1])
            is_squeeze = squeeze_ratio < self.squeeze_threshold
            
            # Check previous squeeze condition to detect breakouts
            prev_squeeze_ratio = (bb_upper[-2] - bb_lower[-2]) / (kc_upper[-2] - kc_lower[-2]) if len(bb_upper) > 1 else squeeze_ratio
            was_squeeze = prev_squeeze_ratio < self.squeeze_threshold
            
            current_price = close_prices[-1]
            previous_price = close_prices[-2] if len(close_prices) > 1 else current_price
            
            # Buy signal: Breakout from squeeze to the upside
            if was_squeeze and not is_squeeze and current_price > bb_middle[-1]:
                reason = f"Bullish breakout from squeeze: Price {current_price:.2f} > BB middle {bb_middle[-1]:.2f}, squeeze ratio {squeeze_ratio:.3f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Buy signal: Currently in squeeze with upward momentum
            elif is_squeeze and current_price > previous_price and current_price > bb_middle[-1]:
                reason = f"Squeeze with upward momentum: Price {current_price:.2f}, squeeze ratio {squeeze_ratio:.3f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: Breakout from squeeze to the downside
            elif was_squeeze and not is_squeeze and current_price < bb_middle[-1]:
                reason = f"Bearish breakout from squeeze: Price {current_price:.2f} < BB middle {bb_middle[-1]:.2f}, squeeze ratio {squeeze_ratio:.3f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Sell signal: High volatility (wide bands)
            elif squeeze_ratio > 1.2:  # Bands are 20% wider than Keltner Channels
                reason = f"High volatility: Squeeze ratio {squeeze_ratio:.3f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check position relative to bands when not in squeeze
            elif not is_squeeze:
                if current_price > bb_upper[-1]:
                    reason = f"Above upper Bollinger Band: {current_price:.2f} > {bb_upper[-1]:.2f}"
                    self.log_signal(-1, reason, data)
                    return -1
                elif current_price < bb_lower[-1]:
                    reason = f"Below lower Bollinger Band: {current_price:.2f} < {bb_lower[-1]:.2f}"
                    self.log_signal(1, reason, data)
                    return 1
                elif current_price > bb_middle[-1]:
                    reason = f"Above BB middle: {current_price:.2f} > {bb_middle[-1]:.2f}"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Below BB middle: {current_price:.2f} < {bb_middle[-1]:.2f}"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Default case: in squeeze, wait for breakout
            else:
                reason = f"In squeeze, waiting for breakout: squeeze ratio {squeeze_ratio:.3f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Bollinger Band Squeeze calculation: {str(e)}", data)
            return -1
