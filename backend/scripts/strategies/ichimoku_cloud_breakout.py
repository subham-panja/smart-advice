"""
Ichimoku Cloud Breakout Strategy
File: scripts/strategies/ichimoku_cloud_breakout.py

This strategy uses Ichimoku Cloud breakouts to identify strong trend changes
and generate buy/sell signals based on price breaking above/below the cloud.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class Ichimoku_Cloud_Breakout(BaseStrategy):
    """
    Strategy based on Ichimoku Cloud breakouts.
    
    Ichimoku Components:
    - Tenkan Sen (Conversion Line): (H9 + L9) / 2
    - Kijun Sen (Base Line): (H26 + L26) / 2
    - Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted +26
    - Senkou Span B (Leading Span B): (H52 + L52) / 2, shifted +26
    - Chikou Span (Lagging Span): Close, shifted -26
    
    Cloud (Kumo) = Area between Senkou Span A and B
    """
    
    def __init__(self, params=None):
        """
        Initialize the Ichimoku Cloud Breakout strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - tenkan_period: Tenkan Sen period (default: 9)
                   - kijun_period: Kijun Sen period (default: 26)
                   - senkou_b_period: Senkou Span B period (default: 52)
                   - displacement: Cloud displacement (default: 26)
                   - min_cloud_thickness: Minimum cloud thickness for valid signal (default: 0.5%)
        """
        super().__init__(params)
        self.tenkan_period = self.get_parameter('tenkan_period', 9)
        self.kijun_period = self.get_parameter('kijun_period', 26)
        self.senkou_b_period = self.get_parameter('senkou_b_period', 52)
        self.displacement = self.get_parameter('displacement', 26)
        self.min_cloud_thickness = self.get_parameter('min_cloud_thickness', 0.005)  # 0.5%
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Ichimoku Cloud Breakout strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        min_periods = max(self.senkou_b_period, self.displacement) + 10
        if not self.validate_data(data, min_periods=min_periods):
            self.log_signal(-1, "Insufficient data for Ichimoku analysis", data)
            return -1
        
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            # Calculate Ichimoku components
            tenkan_sen = self._calculate_line(high, low, self.tenkan_period)
            kijun_sen = self._calculate_line(high, low, self.kijun_period)
            
            # Senkou Span A (displaced forward)
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            
            # Senkou Span B (displaced forward)
            senkou_span_b = self._calculate_line(high, low, self.senkou_b_period)
            
            # For current analysis, we look at the cloud at current time
            # (which was calculated 26 periods ago)
            if len(senkou_span_a) < self.displacement or len(senkou_span_b) < self.displacement:
                self.log_signal(-1, "Insufficient data for cloud calculation", data)
                return -1
            
            current_close = close[-1]
            prev_close = close[-2] if len(close) > 1 else current_close
            
            # Current cloud values (these represent the cloud "now")
            current_span_a = senkou_span_a[-self.displacement] if len(senkou_span_a) >= self.displacement else senkou_span_a[-1]
            current_span_b = senkou_span_b[-self.displacement] if len(senkou_span_b) >= self.displacement else senkou_span_b[-1]
            
            # Previous cloud values
            prev_span_a = senkou_span_a[-self.displacement-1] if len(senkou_span_a) >= self.displacement+1 else current_span_a
            prev_span_b = senkou_span_b[-self.displacement-1] if len(senkou_span_b) >= self.displacement+1 else current_span_b
            
            # Determine cloud boundaries
            current_cloud_top = max(current_span_a, current_span_b)
            current_cloud_bottom = min(current_span_a, current_span_b)
            
            prev_cloud_top = max(prev_span_a, prev_span_b)
            prev_cloud_bottom = min(prev_span_a, prev_span_b)
            
            # Check cloud thickness (avoid thin/weak clouds)
            cloud_thickness = abs(current_span_a - current_span_b) / current_close
            if cloud_thickness < self.min_cloud_thickness:
                self.log_signal(-1, f"Cloud too thin: {cloud_thickness*100:.2f}% < {self.min_cloud_thickness*100:.1f}%", data)
                return -1
            
            # Determine cloud color/trend
            cloud_bullish = current_span_a > current_span_b  # Green/bullish cloud
            cloud_bearish = current_span_a < current_span_b  # Red/bearish cloud
            
            # Check for breakout signals
            
            # Bullish breakout: Price breaks above cloud
            if (prev_close <= prev_cloud_top and current_close > current_cloud_top):
                if cloud_bullish:
                    self.log_signal(1, f"Bullish cloud breakout: Price({current_close:.2f}) > Cloud({current_cloud_top:.2f}), Green cloud", data)
                    return 1
                else:
                    # Breaking above bearish cloud - less strong but still bullish
                    self.log_signal(1, f"Bullish breakout above red cloud: Price({current_close:.2f}) > Cloud({current_cloud_top:.2f})", data)
                    return 1
            
            # Bearish breakdown: Price breaks below cloud
            elif (prev_close >= prev_cloud_bottom and current_close < current_cloud_bottom):
                self.log_signal(-1, f"Bearish cloud breakdown: Price({current_close:.2f}) < Cloud({current_cloud_bottom:.2f})", data)
                return -1
            
            # Check for position relative to cloud
            if current_close > current_cloud_top:
                # Above cloud
                if cloud_bullish:
                    # Above bullish cloud - strong uptrend
                    distance_above = (current_close - current_cloud_top) / current_close
                    if distance_above > 0.02:  # More than 2% above cloud
                        self.log_signal(1, f"Strong position above green cloud: {distance_above*100:.1f}% above", data)
                        return 1
                    else:
                        self.log_signal(1, f"Above green cloud: {distance_above*100:.1f}% above", data)
                        return 1
                else:
                    # Above bearish cloud - potential reversal but cautious
                    self.log_signal(-1, f"Above red cloud - mixed signals", data)
                    return -1
                    
            elif current_close < current_cloud_bottom:
                # Below cloud - bearish
                self.log_signal(-1, f"Below cloud: Price({current_close:.2f}) < Cloud({current_cloud_bottom:.2f})", data)
                return -1
                
            else:
                # Inside cloud - indecision/consolidation
                cloud_position = (current_close - current_cloud_bottom) / (current_cloud_top - current_cloud_bottom)
                self.log_signal(-1, f"Inside cloud: {cloud_position*100:.1f}% through cloud (indecision)", data)
                return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Ichimoku analysis: {str(e)}", data)
            return -1
    
    def _calculate_line(self, high: np.ndarray, low: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate Ichimoku line (highest high + lowest low) / 2 for given period.
        
        Args:
            high: High prices array
            low: Low prices array  
            period: Calculation period
            
        Returns:
            Calculated line values
        """
        result = np.full(len(high), np.nan)
        
        for i in range(period - 1, len(high)):
            highest_high = np.max(high[i - period + 1:i + 1])
            lowest_low = np.min(low[i - period + 1:i + 1])
            result[i] = (highest_high + lowest_low) / 2
            
        return result
