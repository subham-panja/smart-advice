"""
Doji Candlestick Pattern Strategy
File: scripts/strategies/candlestick_doji.py

This strategy identifies doji candlestick patterns, which indicate market indecision
and potential reversal points. A doji has nearly equal open and close prices.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class Candlestick_Doji(BaseStrategy):
    """
    Strategy that identifies doji candlestick patterns.
    
    Doji criteria:
    1. Open and close prices are nearly equal (small body)
    2. Has upper and/or lower shadows
    3. Occurs after a significant trend for reversal signal
    4. Volume and context determine the signal strength
    
    Types of doji patterns:
    - Standard Doji: Small body with shadows on both sides
    - Dragonfly Doji: Small body at high with long lower shadow
    - Gravestone Doji: Small body at low with long upper shadow
    - Four Price Doji: Open = High = Low = Close (rare)
    """
    
    def __init__(self, params=None):
        """
        Initialize the Doji candlestick strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - body_threshold: Maximum body size ratio for doji (default: 0.05)
                   - min_shadow_ratio: Minimum shadow to range ratio (default: 0.3)
                   - trend_periods: Periods to check for trend (default: 10)
                   - trend_strength: Minimum trend strength for reversal signal (default: 0.02)
        """
        super().__init__(params)
        self.body_threshold = self.get_parameter('body_threshold', 0.05)
        self.min_shadow_ratio = self.get_parameter('min_shadow_ratio', 0.3)
        self.trend_periods = self.get_parameter('trend_periods', 10)
        self.trend_strength = self.get_parameter('trend_strength', 0.02)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Doji candlestick pattern strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal (bullish reversal), -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=self.trend_periods + 2):
            self.log_signal(-1, "Insufficient data for Doji analysis", data)
            return -1
        
        try:
            # Get the latest candlestick
            latest = data.iloc[-1]
            open_price = latest['Open']
            high_price = latest['High']
            low_price = latest['Low']
            close_price = latest['Close']
            volume = latest['Volume']
            
            # Calculate candlestick components
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            # Check for price movement
            if total_range == 0:
                # Four Price Doji - very rare, treat as neutral
                self.log_signal(-1, "Four Price Doji - no price movement", data)
                return -1
            
            # 1. Check if body is small enough to be considered a doji
            body_ratio = body_size / total_range
            if body_ratio > self.body_threshold:
                self.log_signal(-1, f"Body too large for doji: {body_ratio:.3f} > {self.body_threshold}", data)
                return -1
            
            # 2. Check for meaningful shadows
            shadow_ratio = (upper_shadow + lower_shadow) / total_range
            if shadow_ratio < self.min_shadow_ratio:
                self.log_signal(-1, f"Insufficient shadows: {shadow_ratio:.3f} < {self.min_shadow_ratio}", data)
                return -1
            
            # 3. Determine doji type
            upper_shadow_ratio = upper_shadow / total_range
            lower_shadow_ratio = lower_shadow / total_range
            
            doji_type = "Standard"
            if lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1:
                doji_type = "Dragonfly"  # Bullish reversal pattern
            elif upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1:
                doji_type = "Gravestone"  # Bearish reversal pattern
            
            # 4. Check for significant prior trend
            if len(data) >= self.trend_periods + 1:
                # Calculate trend over recent periods
                recent_closes = data['Close'].iloc[-self.trend_periods-1:]
                first_close = recent_closes.iloc[0]
                last_close = recent_closes.iloc[-2]  # Exclude current doji candle
                
                trend_change = (last_close - first_close) / first_close
                trend_direction = "up" if trend_change > self.trend_strength else "down" if trend_change < -self.trend_strength else "sideways"
                
                # Determine signal based on trend and doji type
                if trend_direction == "down":
                    # After downtrend, doji suggests potential bullish reversal
                    if doji_type == "Dragonfly":
                        # Strong bullish signal
                        self.log_signal(1, f"Dragonfly Doji after {abs(trend_change)*100:.1f}% downtrend - strong bullish reversal", data)
                        return 1
                    elif doji_type == "Standard":
                        # Moderate bullish signal
                        self.log_signal(1, f"Standard Doji after {abs(trend_change)*100:.1f}% downtrend - bullish reversal", data)
                        return 1
                    else:
                        # Gravestone after downtrend - less reliable
                        self.log_signal(-1, f"Gravestone Doji after downtrend - conflicting signals", data)
                        return -1
                
                elif trend_direction == "up":
                    # After uptrend, doji suggests potential bearish reversal
                    # For a buy-focused system, this is not favorable
                    self.log_signal(-1, f"{doji_type} Doji after {trend_change*100:.1f}% uptrend - potential bearish reversal", data)
                    return -1
                
                else:
                    # Sideways trend - doji less significant
                    self.log_signal(-1, f"{doji_type} Doji in sideways market - low significance", data)
                    return -1
            
            else:
                # Insufficient trend data - treat cautiously
                if doji_type == "Dragonfly":
                    self.log_signal(1, f"Dragonfly Doji - potential bullish signal (limited trend data)", data)
                    return 1
                else:
                    self.log_signal(-1, f"{doji_type} Doji - insufficient trend context", data)
                    return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Doji analysis: {str(e)}", data)
            return -1
