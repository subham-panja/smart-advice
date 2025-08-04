"""
Channel Trading Strategy
File: scripts/strategies/channel_trading.py

This strategy identifies price channels and trades breakouts or bounces.
Focuses on channel breakouts for trend continuation signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Channel_Trading(BaseStrategy):
    """
    Channel Trading Strategy.
    
    Buy Signal: Breakout above channel resistance or bounce from channel support
    Uses linear regression channels and traditional support/resistance levels
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.channel_period = self.get_parameter('channel_period', 20)  # Period for channel calculation
        self.breakout_threshold = self.get_parameter('breakout_threshold', 1.0)  # % above resistance for breakout
        self.support_bounce_threshold = self.get_parameter('support_bounce_threshold', 2.0)  # % above support for bounce
        self.volume_confirmation = self.get_parameter('volume_confirmation', 1.2)  # Volume multiplier for confirmation
        
    def calculate_linear_regression_channel(self, data: pd.Series, period: int):
        """
        Calculate linear regression channel with upper and lower bounds.
        """
        try:
            # Get the data for regression
            y = data.tail(period).values
            x = np.arange(len(y))
            
            # Calculate linear regression
            coeffs = np.polyfit(x, y, 1)
            regression_line = np.polyval(coeffs, x)
            
            # Calculate standard deviation of residuals
            residuals = y - regression_line
            std_dev = np.std(residuals)
            
            # Calculate channel bounds (2 standard deviations)
            upper_channel = regression_line + (2 * std_dev)
            lower_channel = regression_line - (2 * std_dev)
            
            return {
                'regression': regression_line[-1],
                'upper': upper_channel[-1],
                'lower': lower_channel[-1],
                'slope': coeffs[0],  # Trend direction
                'std_dev': std_dev
            }
            
        except Exception as e:
            return None
    
    def find_support_resistance_levels(self, data: pd.DataFrame, period: int):
        """
        Find support and resistance levels using pivot points.
        """
        try:
            high_prices = data['High'].tail(period)
            low_prices = data['Low'].tail(period)
            
            # Find recent highs and lows
            resistance_levels = []
            support_levels = []
            
            # Look for local maxima (resistance) and minima (support)
            for i in range(2, len(high_prices) - 2):
                # Resistance: local maximum
                if (high_prices.iloc[i] > high_prices.iloc[i-1] and 
                    high_prices.iloc[i] > high_prices.iloc[i+1] and
                    high_prices.iloc[i] > high_prices.iloc[i-2] and 
                    high_prices.iloc[i] > high_prices.iloc[i+2]):
                    resistance_levels.append(high_prices.iloc[i])
                
                # Support: local minimum
                if (low_prices.iloc[i] < low_prices.iloc[i-1] and 
                    low_prices.iloc[i] < low_prices.iloc[i+1] and
                    low_prices.iloc[i] < low_prices.iloc[i-2] and 
                    low_prices.iloc[i] < low_prices.iloc[i+2]):
                    support_levels.append(low_prices.iloc[i])
            
            # Get the most relevant levels (closest to current price)
            current_price = data['Close'].iloc[-1]
            
            # Find nearest resistance above current price
            resistance_above = [r for r in resistance_levels if r > current_price]
            nearest_resistance = min(resistance_above) if resistance_above else None
            
            # Find nearest support below current price
            support_below = [s for s in support_levels if s < current_price]
            nearest_support = max(support_below) if support_below else None
            
            return {
                'resistance': nearest_resistance,
                'support': nearest_support,
                'all_resistance': resistance_levels,
                'all_support': support_levels
            }
            
        except Exception as e:
            return {'resistance': None, 'support': None, 'all_resistance': [], 'all_support': []}
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the channel trading strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        min_periods = self.channel_period + 5
        if not self.validate_data(data, min_periods=min_periods):
            return -1
            
        try:
            current_price = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].tail(20).mean()
            volume_ratio = current_volume / avg_volume
            
            # Calculate linear regression channel
            lr_channel = self.calculate_linear_regression_channel(data['Close'], self.channel_period)
            
            # Find support and resistance levels
            sr_levels = self.find_support_resistance_levels(data, self.channel_period * 2)
            
            if lr_channel is None:
                self.log_signal(-1, "Unable to calculate regression channel", data)
                return -1
            
            # Channel breakout signals
            upper_channel = lr_channel['upper']
            lower_channel = lr_channel['lower']
            slope = lr_channel['slope']
            
            # Breakout above upper channel (bullish)
            breakout_level = upper_channel * (1 + self.breakout_threshold / 100)
            if current_price > breakout_level:
                # Volume confirmation
                if volume_ratio >= self.volume_confirmation:
                    reason = f"Channel breakout: Price {current_price:.2f} breaks above {breakout_level:.2f} with {volume_ratio:.1f}x volume"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Weak breakout: Above channel but low volume ({volume_ratio:.1f}x)"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Bounce from lower channel (bullish reversal)
            bounce_level = lower_channel * (1 + self.support_bounce_threshold / 100)
            if current_price > lower_channel and current_price <= bounce_level:
                # Additional confirmation: upward slope suggests uptrend
                if slope > 0 and volume_ratio >= self.volume_confirmation:
                    reason = f"Channel support bounce: Price {current_price:.2f} bouncing from {lower_channel:.2f} in uptrend"
                    self.log_signal(1, reason, data)
                    return 1
            
            # Traditional resistance breakout
            if sr_levels['resistance'] is not None:
                resistance_breakout = sr_levels['resistance'] * (1 + self.breakout_threshold / 100)
                if current_price > resistance_breakout and volume_ratio >= self.volume_confirmation:
                    reason = f"Resistance breakout: Price {current_price:.2f} breaks {sr_levels['resistance']:.2f} with volume"
                    self.log_signal(1, reason, data)
                    return 1
            
            # Support bounce with traditional levels
            if sr_levels['support'] is not None:
                support_bounce = sr_levels['support'] * (1 + self.support_bounce_threshold / 100)
                if (current_price > sr_levels['support'] and current_price <= support_bounce and 
                    volume_ratio >= self.volume_confirmation):
                    reason = f"Support bounce: Price {current_price:.2f} bouncing from {sr_levels['support']:.2f}"
                    self.log_signal(1, reason, data)
                    return 1
            
            # Price in middle of channel - check trend direction
            if lower_channel < current_price < upper_channel:
                if slope > 0.1:  # Positive slope indicates uptrend
                    reason = f"Channel uptrend: Price {current_price:.2f} in upward channel (slope: {slope:.4f})"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Channel neutral: Price in channel but no clear trend (slope: {slope:.4f})"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Price below lower channel (bearish)
            if current_price < lower_channel:
                reason = f"Below channel: Price {current_price:.2f} below support {lower_channel:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Default case
            reason = f"No clear channel signal: Price {current_price:.2f} in range [{lower_channel:.2f}, {upper_channel:.2f}]"
            self.log_signal(-1, reason, data)
            return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in channel trading calculation: {str(e)}", data)
            return -1
