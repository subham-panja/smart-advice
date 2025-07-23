"""
Keltner Channel Squeeze Strategy
File: scripts/strategies/keltner_channel_squeeze.py

This strategy identifies squeeze conditions in Keltner Channels and generates
signals when the squeeze releases, indicating potential breakout moves.
"""

import pandas as pd
import numpy as np
import talib as ta
from scripts.strategies.base_strategy import BaseStrategy


class Keltner_Channel_Squeeze(BaseStrategy):
    """
    Strategy based on Keltner Channel squeeze conditions.
    
    A squeeze occurs when volatility is low and the channels are narrow.
    The squeeze release often leads to strong breakout moves.
    
    Signals:
    - Squeeze release with upward momentum: Buy signal
    - Squeeze release with downward momentum: Sell signal
    - Squeeze condition: Hold/wait signal
    """
    
    def __init__(self, params=None):
        """
        Initialize the Keltner Channel Squeeze strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - ema_period: EMA period for middle line (default: 20)
                   - atr_period: ATR period for channel width (default: 10)
                   - multiplier: ATR multiplier for channels (default: 2.0)
                   - squeeze_threshold: Threshold ratio for squeeze detection (default: 0.015)
                   - momentum_period: Period for momentum calculation (default: 12)
        """
        super().__init__(params)
        self.ema_period = self.get_parameter('ema_period', 20)
        self.atr_period = self.get_parameter('atr_period', 10)
        self.multiplier = self.get_parameter('multiplier', 2.0)
        self.squeeze_threshold = self.get_parameter('squeeze_threshold', 0.015)  # 1.5%
        self.momentum_period = self.get_parameter('momentum_period', 12)
    
    def _execute_strategy_logic(self, data):
        """
        Execute the Keltner Channel Squeeze strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        min_periods = max(self.ema_period, self.atr_period, self.momentum_period) + 10
        if not self.validate_data(data, min_periods=min_periods):
            self.log_signal(-1, "Insufficient data for Keltner Channel analysis", data)
            return -1
        
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            # Calculate Keltner Channels
            ema = ta.EMA(close, timeperiod=self.ema_period)
            atr = ta.ATR(high, low, close, timeperiod=self.atr_period)
            
            if len(ema) < 5 or len(atr) < 5 or np.isnan(ema[-1]) or np.isnan(atr[-1]):
                self.log_signal(-1, "Insufficient EMA/ATR data", data)
                return -1
            
            # Calculate channel boundaries
            upper_channel = ema + (atr * self.multiplier)
            lower_channel = ema - (atr * self.multiplier)
            
            current_close = close[-1]
            current_ema = ema[-1]
            current_upper = upper_channel[-1]
            current_lower = lower_channel[-1]
            
            # Calculate channel width as percentage of price
            channel_width = (current_upper - current_lower) / current_close
            
            # Calculate momentum (linear regression slope of closes)
            momentum_values = []
            if len(close) >= self.momentum_period:
                recent_closes = close[-self.momentum_period:]
                x_values = np.arange(len(recent_closes))
                
                # Simple momentum calculation (slope of linear regression)
                if len(recent_closes) > 1:
                    momentum = np.polyfit(x_values, recent_closes, 1)[0]
                    momentum_normalized = momentum / current_close * 100  # As percentage
                else:
                    momentum_normalized = 0
            else:
                momentum_normalized = 0
            
            # Check for squeeze condition
            is_squeezed = channel_width < self.squeeze_threshold
            
            # Get historical squeeze data to detect releases
            historical_widths = []
            for i in range(max(5, len(ema) - 10), len(ema)):
                if i >= 0 and not np.isnan(upper_channel[i]) and not np.isnan(lower_channel[i]):
                    width = (upper_channel[i] - lower_channel[i]) / close[i]
                    historical_widths.append(width)
            
            if len(historical_widths) < 3:
                self.log_signal(-1, "Insufficient historical width data", data)
                return -1
            
            # Check if we're coming out of a squeeze (expanding after contraction)
            was_squeezed = np.mean(historical_widths[-3:-1]) < self.squeeze_threshold
            is_expanding = historical_widths[-1] > np.mean(historical_widths[-3:-1])
            
            # Determine position relative to channel
            channel_position = (current_close - current_lower) / (current_upper - current_lower)
            
            # Generate signals
            if was_squeezed and is_expanding:
                # Squeeze release detected
                if momentum_normalized > 0.1 and channel_position > 0.5:
                    # Bullish squeeze release
                    self.log_signal(1, f"Bullish squeeze release: momentum {momentum_normalized:.2f}%, expanding from {np.mean(historical_widths[-3:-1])*100:.2f}% to {channel_width*100:.2f}%", data)
                    return 1
                elif momentum_normalized < -0.1 and channel_position < 0.5:
                    # Bearish squeeze release
                    self.log_signal(-1, f"Bearish squeeze release: momentum {momentum_normalized:.2f}%, position {channel_position*100:.1f}%", data)
                    return -1
                else:
                    # Unclear direction
                    self.log_signal(-1, f"Squeeze release with unclear direction: momentum {momentum_normalized:.2f}%", data)
                    return -1
            
            elif not is_squeezed:
                # Not in squeeze - check for normal channel signals
                if current_close > current_ema and momentum_normalized > 0.2:
                    # Above middle line with positive momentum
                    if channel_position > 0.7:
                        # Near upper channel - potential breakout
                        self.log_signal(1, f"Near upper channel with momentum: position {channel_position*100:.1f}%, momentum {momentum_normalized:.2f}%", data)
                        return 1
                    else:
                        # Moderate bullish position
                        self.log_signal(1, f"Above EMA with momentum: momentum {momentum_normalized:.2f}%", data)
                        return 1
                elif current_close < current_ema and momentum_normalized < -0.2:
                    # Below middle line with negative momentum
                    self.log_signal(-1, f"Below EMA with negative momentum: {momentum_normalized:.2f}%", data)
                    return -1
                else:
                    # Neutral condition
                    self.log_signal(-1, f"Neutral: position {channel_position*100:.1f}%, momentum {momentum_normalized:.2f}%", data)
                    return -1
            
            else:
                # Currently in squeeze - wait for release
                squeeze_duration = sum(1 for w in historical_widths if w < self.squeeze_threshold)
                self.log_signal(-1, f"In squeeze: width {channel_width*100:.2f}% < {self.squeeze_threshold*100:.1f}%, duration {squeeze_duration} periods", data)
                return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Keltner Channel analysis: {str(e)}", data)
            return -1
