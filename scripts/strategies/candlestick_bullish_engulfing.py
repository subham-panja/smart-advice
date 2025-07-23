"""
Bullish Engulfing Candlestick Pattern Strategy
File: scripts/strategies/candlestick_bullish_engulfing.py

This strategy identifies bullish engulfing candlestick patterns.
A bullish engulfing pattern consists of two candles where the second (bullish) candle 
completely engulfs the body of the first (bearish) candle.
"""

import pandas as pd
import numpy as np
from scripts.strategies.base_strategy import BaseStrategy


class Candlestick_Bullish_Engulfing(BaseStrategy):
    """
    Strategy that identifies bullish engulfing candlestick patterns.
    
    Bullish engulfing criteria:
    1. First candle is bearish (red/black)
    2. Second candle is bullish (green/white)
    3. Second candle's body completely engulfs the first candle's body
    4. Occurs after a downtrend for reversal signal
    5. Higher volume on the engulfing candle is preferred
    """
    
    def __init__(self, params=None):
        """
        Initialize the Bullish Engulfing candlestick strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - min_body_ratio: Minimum body size ratio for significance (default: 0.02)
                   - volume_multiplier: Preferred volume increase (default: 1.2)
                   - trend_periods: Periods to check for downtrend (default: 10)
        """
        super().__init__(params)
        self.min_body_ratio = self.get_parameter('min_body_ratio', 0.02)
        self.volume_multiplier = self.get_parameter('volume_multiplier', 1.2)
        self.trend_periods = self.get_parameter('trend_periods', 10)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Bullish Engulfing candlestick pattern strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=self.trend_periods + 3):
            self.log_signal(-1, "Insufficient data for Bullish Engulfing analysis", data)
            return -1
        
        try:
            # Get the last two candlesticks
            current = data.iloc[-1]  # Second candle (should be bullish)
            previous = data.iloc[-2]  # First candle (should be bearish)
            
            # Current candle components
            curr_open = current['Open']
            curr_high = current['High']
            curr_low = current['Low']
            curr_close = current['Close']
            curr_volume = current['Volume']
            
            # Previous candle components
            prev_open = previous['Open']
            prev_high = previous['High']
            prev_low = previous['Low']
            prev_close = previous['Close']
            prev_volume = previous['Volume']
            
            # Calculate body sizes
            curr_body = abs(curr_close - curr_open)
            prev_body = abs(prev_close - prev_open)
            curr_range = curr_high - curr_low
            prev_range = prev_high - prev_low
            
            # Check for minimum body significance
            if curr_range == 0 or prev_range == 0:
                self.log_signal(-1, "No price movement in candles", data)
                return -1
                
            curr_body_ratio = curr_body / curr_range
            prev_body_ratio = prev_body / prev_range
            
            if curr_body_ratio < self.min_body_ratio or prev_body_ratio < self.min_body_ratio:
                self.log_signal(-1, f"Insignificant bodies: curr={curr_body_ratio:.3f}, prev={prev_body_ratio:.3f}", data)
                return -1
            
            # 1. First candle must be bearish
            if prev_close >= prev_open:
                self.log_signal(-1, f"First candle not bearish: close={prev_close:.2f} >= open={prev_open:.2f}", data)
                return -1
            
            # 2. Second candle must be bullish
            if curr_close <= curr_open:
                self.log_signal(-1, f"Second candle not bullish: close={curr_close:.2f} <= open={curr_open:.2f}", data)
                return -1
            
            # 3. Second candle's body must engulf first candle's body
            # Current open must be below previous close AND
            # Current close must be above previous open
            if not (curr_open < prev_close and curr_close > prev_open):
                self.log_signal(-1, f"No engulfing: curr_open={curr_open:.2f}, prev_close={prev_close:.2f}, curr_close={curr_close:.2f}, prev_open={prev_open:.2f}", data)
                return -1
            
            # 4. Check for prior downtrend
            if len(data) >= self.trend_periods + 2:
                # Look at closes before the pattern (exclude the two pattern candles)
                trend_data = data['Close'].iloc[-self.trend_periods-2:-2]
                if len(trend_data) >= 2:
                    # Check if trend is generally declining
                    declining_count = 0
                    for i in range(1, len(trend_data)):
                        if trend_data.iloc[i] < trend_data.iloc[i-1]:
                            declining_count += 1
                    
                    trend_ratio = declining_count / (len(trend_data) - 1)
                    if trend_ratio < 0.4:  # At least 40% should be declining
                        self.log_signal(-1, f"No clear downtrend: {trend_ratio:.2f} declining ratio", data)
                        return -1
            
            # 5. Check volume confirmation (preferred but not mandatory)
            volume_increase = 1.0
            if prev_volume > 0:
                volume_increase = curr_volume / prev_volume
            
            volume_confirmed = volume_increase >= self.volume_multiplier
            
            # Calculate engulfing strength
            engulfing_strength = (curr_close - curr_open) / (prev_open - prev_close)
            
            # All criteria met - Bullish Engulfing pattern detected
            volume_note = "with volume confirmation" if volume_confirmed else f"volume increase: {volume_increase:.2f}x"
            self.log_signal(1, f"Bullish Engulfing pattern: strength={engulfing_strength:.2f}x, {volume_note}", data)
            return 1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Bullish Engulfing analysis: {str(e)}", data)
            return -1
