"""
Ichimoku Kijun-Tenkan Crossover Strategy
File: scripts/strategies/ichimoku_kijun_tenkan_crossover.py

This strategy uses crossovers between Tenkan Sen and Kijun Sen lines
to generate buy/sell signals. This is one of the key Ichimoku signals.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class Ichimoku_Kijun_Tenkan_Crossover(BaseStrategy):
    """
    Strategy based on Ichimoku Tenkan-Kijun crossovers.
    
    Signals:
    - Tenkan Sen crossing above Kijun Sen: Bullish signal (Golden Cross)
    - Tenkan Sen crossing below Kijun Sen: Bearish signal (Dead Cross)
    - Additional filters: price position relative to cloud, cloud color
    """
    
    def __init__(self, params=None):
        """
        Initialize the Ichimoku Kijun-Tenkan Crossover strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - tenkan_period: Tenkan Sen period (default: 9)
                   - kijun_period: Kijun Sen period (default: 26)
                   - senkou_b_period: Senkou Span B period (default: 52)
                   - displacement: Cloud displacement (default: 26)
                   - use_cloud_filter: Whether to filter signals with cloud position (default: True)
        """
        super().__init__(params)
        self.tenkan_period = self.get_parameter('tenkan_period', 9)
        self.kijun_period = self.get_parameter('kijun_period', 26)
        self.senkou_b_period = self.get_parameter('senkou_b_period', 52)
        self.displacement = self.get_parameter('displacement', 26)
        self.use_cloud_filter = self.get_parameter('use_cloud_filter', True)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Ichimoku Kijun-Tenkan Crossover strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        min_periods = max(self.senkou_b_period, self.displacement) + 5 if self.use_cloud_filter else self.kijun_period + 5
        if not self.validate_data(data, min_periods=min_periods):
            self.log_signal(-1, "Insufficient data for Ichimoku crossover analysis", data)
            return -1
        
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            # Calculate Ichimoku lines
            tenkan_sen = self._calculate_line(high, low, self.tenkan_period)
            kijun_sen = self._calculate_line(high, low, self.kijun_period)
            
            if np.isnan(tenkan_sen[-1]) or np.isnan(kijun_sen[-1]):
                self.log_signal(-1, "Insufficient data for Tenkan/Kijun calculation", data)
                return -1
            
            current_tenkan = tenkan_sen[-1]
            current_kijun = kijun_sen[-1]
            prev_tenkan = tenkan_sen[-2] if len(tenkan_sen) > 1 else current_tenkan
            prev_kijun = kijun_sen[-2] if len(kijun_sen) > 1 else current_kijun
            
            current_close = close[-1]
            
            # Calculate cloud if using cloud filter
            cloud_bullish = None
            above_cloud = None
            below_cloud = None
            in_cloud = None
            
            if self.use_cloud_filter and len(data) >= self.displacement + self.senkou_b_period:
                # Calculate cloud components
                senkou_span_a = (tenkan_sen + kijun_sen) / 2
                senkou_span_b = self._calculate_line(high, low, self.senkou_b_period)
                
                # Get current cloud values (displaced)
                if len(senkou_span_a) >= self.displacement and len(senkou_span_b) >= self.displacement:
                    current_span_a = senkou_span_a[-self.displacement]
                    current_span_b = senkou_span_b[-self.displacement]
                    
                    cloud_top = max(current_span_a, current_span_b)
                    cloud_bottom = min(current_span_a, current_span_b)
                    
                    cloud_bullish = current_span_a > current_span_b
                    above_cloud = current_close > cloud_top
                    below_cloud = current_close < cloud_bottom
                    in_cloud = not above_cloud and not below_cloud
                else:
                    self.use_cloud_filter = False  # Fallback if insufficient cloud data
            
            # Check for crossover signals
            
            # Bullish crossover: Tenkan crosses above Kijun
            if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
                # Additional confirmation: ensure meaningful separation
                separation = abs(current_tenkan - current_kijun) / current_close
                if separation < 0.001:  # Less than 0.1% separation
                    self.log_signal(-1, f"Insignificant crossover: separation {separation*100:.3f}%", data)
                    return -1
                
                # Apply cloud filter if enabled
                if self.use_cloud_filter:
                    if above_cloud and cloud_bullish:
                        self.log_signal(1, f"Strong bullish crossover: Above green cloud, Tenkan({current_tenkan:.2f}) > Kijun({current_kijun:.2f})", data)
                        return 1
                    elif above_cloud:
                        self.log_signal(1, f"Bullish crossover above red cloud: Tenkan({current_tenkan:.2f}) > Kijun({current_kijun:.2f})", data)
                        return 1
                    elif in_cloud and cloud_bullish:
                        self.log_signal(1, f"Moderate bullish crossover in green cloud: Tenkan({current_tenkan:.2f}) > Kijun({current_kijun:.2f})", data)
                        return 1
                    elif below_cloud:
                        self.log_signal(-1, f"Weak bullish crossover below cloud: may be false signal", data)
                        return -1
                    else:
                        self.log_signal(-1, f"Bullish crossover in bearish cloud: conflicting signals", data)
                        return -1
                else:
                    # No cloud filter - simple crossover
                    self.log_signal(1, f"Bullish crossover: Tenkan({current_tenkan:.2f}) > Kijun({current_kijun:.2f})", data)
                    return 1
            
            # Bearish crossover: Tenkan crosses below Kijun
            elif prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
                separation = abs(current_tenkan - current_kijun) / current_close
                if separation < 0.001:
                    self.log_signal(-1, f"Insignificant bearish crossover: separation {separation*100:.3f}%", data)
                    return -1
                
                self.log_signal(-1, f"Bearish crossover: Tenkan({current_tenkan:.2f}) < Kijun({current_kijun:.2f})", data)
                return -1
            
            # Check current trend direction (no crossover)
            if current_tenkan > current_kijun:
                # Bullish alignment
                spread = (current_tenkan - current_kijun) / current_close
                
                if self.use_cloud_filter:
                    if above_cloud and cloud_bullish:
                        if spread > 0.01:  # Strong separation
                            self.log_signal(1, f"Strong bullish trend: Above green cloud, spread {spread*100:.2f}%", data)
                            return 1
                        else:
                            self.log_signal(1, f"Moderate bullish trend: Above green cloud", data)
                            return 1
                    elif above_cloud:
                        self.log_signal(1, f"Bullish trend above red cloud: spread {spread*100:.2f}%", data)
                        return 1
                    elif in_cloud and cloud_bullish:
                        self.log_signal(-1, f"Weak bullish trend in green cloud", data)
                        return -1
                    else:
                        self.log_signal(-1, f"Bullish alignment but poor cloud context", data)
                        return -1
                else:
                    if spread > 0.015:  # 1.5% spread
                        self.log_signal(1, f"Strong bullish alignment: spread {spread*100:.2f}%", data)
                        return 1
                    else:
                        self.log_signal(-1, f"Weak bullish alignment: spread {spread*100:.2f}%", data)
                        return -1
            else:
                # Bearish alignment
                spread = (current_kijun - current_tenkan) / current_close
                self.log_signal(-1, f"Bearish alignment: Kijun > Tenkan, spread {spread*100:.2f}%", data)
                return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Ichimoku crossover analysis: {str(e)}", data)
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
