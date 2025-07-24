"""
Hammer Candlestick Pattern Strategy
File: scripts/strategies/candlestick_hammer.py

This strategy identifies hammer candlestick patterns, which are bullish reversal patterns.
A hammer has a small body near the high of the day with a long lower shadow.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class Candlestick_Hammer(BaseStrategy):
    """
    Strategy that identifies hammer candlestick patterns.
    
    Hammer criteria:
    1. Small body (open and close are close together)
    2. Long lower shadow (at least 2x the body size)
    3. Little to no upper shadow (body near high of the day)
    4. Occurs after a downtrend for reversal signal
    """
    
    def __init__(self, params=None):
        """
        Initialize the Hammer candlestick strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - body_threshold: Maximum body size ratio (default: 0.1)
                   - shadow_ratio: Minimum lower shadow to body ratio (default: 2.0)
                   - upper_shadow_threshold: Maximum upper shadow ratio (default: 0.1)
                   - trend_periods: Periods to check for downtrend (default: 10)
        """
        super().__init__(params)
        self.body_threshold = self.get_parameter('body_threshold', 0.1)
        self.shadow_ratio = self.get_parameter('shadow_ratio', 2.0)
        self.upper_shadow_threshold = self.get_parameter('upper_shadow_threshold', 0.1)
        self.trend_periods = self.get_parameter('trend_periods', 10)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Hammer candlestick pattern strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=self.trend_periods + 2):
            self.log_signal(-1, "Insufficient data for Hammer analysis", data)
            return -1
        
        try:
            # Get the latest candlestick
            latest = data.iloc[-1]
            open_price = latest['Open']
            high_price = latest['High']
            low_price = latest['Low']
            close_price = latest['Close']
            
            # Calculate candlestick components
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            
            # Avoid division by zero
            if total_range == 0:
                self.log_signal(-1, "No price movement in current candle", data)
                return -1
            
            # Check hammer criteria
            body_ratio = body_size / total_range
            
            # 1. Small body
            if body_ratio > self.body_threshold:
                self.log_signal(-1, f"Body too large: {body_ratio:.3f} > {self.body_threshold}", data)
                return -1
            
            # 2. Long lower shadow
            if body_size > 0:  # Avoid division by zero
                lower_shadow_ratio = lower_shadow / body_size
                if lower_shadow_ratio < self.shadow_ratio:
                    self.log_signal(-1, f"Lower shadow too short: {lower_shadow_ratio:.2f} < {self.shadow_ratio}", data)
                    return -1
            else:
                # For doji-like candles, use total range
                lower_shadow_ratio = lower_shadow / total_range
                if lower_shadow_ratio < 0.6:  # At least 60% should be lower shadow
                    self.log_signal(-1, f"Lower shadow insufficient for doji-like hammer: {lower_shadow_ratio:.3f}", data)
                    return -1
            
            # 3. Little to no upper shadow
            upper_shadow_ratio = upper_shadow / total_range
            if upper_shadow_ratio > self.upper_shadow_threshold:
                self.log_signal(-1, f"Upper shadow too long: {upper_shadow_ratio:.3f} > {self.upper_shadow_threshold}", data)
                return -1
            
            # 4. Check for prior downtrend
            if len(data) >= self.trend_periods + 1:
                recent_closes = data['Close'].iloc[-self.trend_periods-1:-1]  # Exclude current candle
                if len(recent_closes) >= 2:
                    # Simple trend check - more closes should be declining
                    declining_count = 0
                    for i in range(1, len(recent_closes)):
                        if recent_closes.iloc[i] < recent_closes.iloc[i-1]:
                            declining_count += 1
                    
                    trend_ratio = declining_count / (len(recent_closes) - 1)
                    if trend_ratio < 0.5:  # At least 50% should be declining
                        self.log_signal(-1, f"No clear downtrend: {trend_ratio:.2f} declining ratio", data)
                        return -1
            
            # All criteria met - Hammer pattern detected
            self.log_signal(1, f"Hammer pattern: body={body_ratio:.3f}, lower_shadow={lower_shadow_ratio:.2f}x body, upper_shadow={upper_shadow_ratio:.3f}", data)
            return 1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Hammer analysis: {str(e)}", data)
            return -1
