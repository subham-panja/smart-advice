"""
Linear Regression Channel Strategy
File: scripts/strategies/linear_regression_channel.py

This strategy uses linear regression channels to identify trend direction
and potential reversal points when price touches channel boundaries.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class Linear_Regression_Channel(BaseStrategy):
    """
    Strategy based on Linear Regression Channels.
    
    The strategy creates a linear regression line through recent price data
    and builds upper/lower channels based on standard deviation.
    
    Signals:
    - Price bouncing off lower channel: Buy signal
    - Price bouncing off upper channel: Sell signal
    - Channel breakouts: Strong trend signals
    """
    
    def __init__(self, params=None):
        """
        Initialize the Linear Regression Channel strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - period: Period for regression calculation (default: 20)
                   - std_dev_multiplier: Standard deviation multiplier for channels (default: 2.0)
                   - min_touches: Minimum touches for reliable channel (default: 2)
                   - breakout_threshold: Threshold for breakout confirmation (default: 0.5%)
        """
        super().__init__(params)
        self.period = self.get_parameter('period', 20)
        self.std_dev_multiplier = self.get_parameter('std_dev_multiplier', 2.0)
        self.min_touches = self.get_parameter('min_touches', 2)
        self.breakout_threshold = self.get_parameter('breakout_threshold', 0.005)  # 0.5%
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Linear Regression Channel strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=self.period + 5):
            self.log_signal(-1, "Insufficient data for Linear Regression analysis", data)
            return -1
        
        try:
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            
            # Use recent data for regression
            recent_close = close[-self.period:]
            recent_high = high[-self.period:]
            recent_low = low[-self.period:]
            
            if len(recent_close) < self.period:
                self.log_signal(-1, "Insufficient recent data", data)
                return -1
            
            # Calculate linear regression
            x_values = np.arange(len(recent_close)).reshape(-1, 1)
            y_values = recent_close
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(x_values, y_values)
            
            # Get regression line values
            regression_line = model.predict(x_values)
            
            # Calculate residuals (distance from regression line)
            residuals = y_values - regression_line
            std_dev = np.std(residuals)
            
            # Create channels
            upper_channel = regression_line + (std_dev * self.std_dev_multiplier)
            lower_channel = regression_line - (std_dev * self.std_dev_multiplier)
            
            # Current values
            current_close = close[-1]
            current_regression = regression_line[-1]
            current_upper = upper_channel[-1]
            current_lower = lower_channel[-1]
            
            # Calculate trend slope (coefficient from regression)
            trend_slope = model.coef_[0]
            trend_slope_normalized = trend_slope / current_close * 100  # As percentage per period
            
            # Determine position within channel
            channel_width = current_upper - current_lower
            if channel_width == 0:
                self.log_signal(-1, "Zero channel width", data)
                return -1
            
            channel_position = (current_close - current_lower) / channel_width
            distance_from_regression = abs(current_close - current_regression) / current_close
            
            # Count touches of upper and lower channels
            upper_touches = 0
            lower_touches = 0
            touch_threshold = std_dev * 0.3  # 30% of std dev for touch detection
            
            for i in range(len(recent_high)):
                if abs(recent_high[i] - upper_channel[i]) < touch_threshold:
                    upper_touches += 1
                if abs(recent_low[i] - lower_channel[i]) < touch_threshold:
                    lower_touches += 1
            
            total_touches = upper_touches + lower_touches
            
            # Check for trend strength
            trend_strength = abs(trend_slope_normalized)
            is_strong_trend = trend_strength > 0.5  # More than 0.5% per period
            is_uptrend = trend_slope_normalized > 0.1
            is_downtrend = trend_slope_normalized < -0.1
            
            # Generate signals
            
            # Check for breakouts first (strongest signals)
            prev_close = close[-2] if len(close) > 1 else current_close
            prev_upper = upper_channel[-2] if len(upper_channel) > 1 else current_upper
            prev_lower = lower_channel[-2] if len(lower_channel) > 1 else current_lower
            
            # Bullish breakout above upper channel
            if prev_close <= prev_upper and current_close > current_upper * (1 + self.breakout_threshold):
                if is_uptrend or not is_downtrend:  # Confirm with trend
                    self.log_signal(1, f"Bullish breakout: Price({current_close:.2f}) > Upper({current_upper:.2f}), trend {trend_slope_normalized:.2f}%", data)
                    return 1
                else:
                    self.log_signal(-1, f"False breakout: Price above channel but downtrend {trend_slope_normalized:.2f}%", data)
                    return -1
            
            # Bearish breakdown below lower channel
            elif prev_close >= prev_lower and current_close < current_lower * (1 - self.breakout_threshold):
                self.log_signal(-1, f"Bearish breakdown: Price({current_close:.2f}) < Lower({current_lower:.2f}), trend {trend_slope_normalized:.2f}%", data)
                return -1
            
            # Channel bounce signals
            elif channel_position < 0.2 and total_touches >= self.min_touches:
                # Near lower channel - potential bounce
                if is_uptrend or (not is_downtrend and lower_touches >= self.min_touches):
                    self.log_signal(1, f"Lower channel bounce: position {channel_position*100:.1f}%, {lower_touches} touches, trend {trend_slope_normalized:.2f}%", data)
                    return 1
                else:
                    self.log_signal(-1, f"Weak lower channel: downtrend {trend_slope_normalized:.2f}%, insufficient support", data)
                    return -1
            
            elif channel_position > 0.8 and total_touches >= self.min_touches:
                # Near upper channel - potential resistance
                if is_downtrend or upper_touches >= self.min_touches:
                    self.log_signal(-1, f"Upper channel resistance: position {channel_position*100:.1f}%, {upper_touches} touches", data)
                    return -1
                else:
                    # Strong uptrend might break through
                    if is_strong_trend and is_uptrend:
                        self.log_signal(1, f"Strong uptrend near upper channel: trend {trend_slope_normalized:.2f}%", data)
                        return 1
                    else:
                        self.log_signal(-1, f"Near upper channel without strong trend", data)
                        return -1
            
            # Trend following within channel
            elif 0.3 <= channel_position <= 0.7:
                # Middle of channel - follow trend
                if is_uptrend and is_strong_trend:
                    self.log_signal(1, f"Uptrend within channel: trend {trend_slope_normalized:.2f}%, position {channel_position*100:.1f}%", data)
                    return 1
                elif is_downtrend and is_strong_trend:
                    self.log_signal(-1, f"Downtrend within channel: trend {trend_slope_normalized:.2f}%", data)
                    return -1
                else:
                    # Weak trend or sideways
                    if current_close > current_regression:
                        self.log_signal(1, f"Above regression line: weak trend {trend_slope_normalized:.2f}%", data)
                        return 1
                    else:
                        self.log_signal(-1, f"Below regression line: weak trend {trend_slope_normalized:.2f}%", data)
                        return -1
            
            else:
                # Other positions - apply conservative approach
                if is_uptrend and channel_position < 0.5:
                    self.log_signal(1, f"Conservative uptrend: position {channel_position*100:.1f}%, trend {trend_slope_normalized:.2f}%", data)
                    return 1
                else:
                    self.log_signal(-1, f"Conservative: position {channel_position*100:.1f}%, trend {trend_slope_normalized:.2f}%", data)
                    return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Linear Regression analysis: {str(e)}", data)
            return -1
