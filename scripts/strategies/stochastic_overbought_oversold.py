"""
Stochastic Oscillator Strategy
File: scripts/strategies/stochastic_overbought_oversold.py

This strategy uses the Stochastic Oscillator to identify overbought and oversold conditions.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Stochastic_Overbought_Oversold(BaseStrategy):
    """
    Stochastic Oscillator Strategy for overbought/oversold conditions.
    
    Buy Signal: Stochastic %K crosses above %D from oversold region
    Sell Signal: Stochastic %K crosses below %D from overbought region
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.k_period = self.get_parameter('k_period', 14)
        self.d_period = self.get_parameter('d_period', 3)
        self.overbought_level = self.get_parameter('overbought_level', 80)
        self.oversold_level = self.get_parameter('oversold_level', 20)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Stochastic Oscillator strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.k_period):
            return -1
            
        try:
            # Calculate Stochastic using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            # Calculate Stochastic %K and %D
            slowk, slowd = ta.STOCH(high_prices, low_prices, close_prices, 
                                   fastk_period=self.k_period, 
                                   slowk_period=self.d_period, 
                                   slowd_period=self.d_period)
            
            # Check if we have valid values for the latest periods
            if (pd.isna(slowk[-1]) or pd.isna(slowd[-1]) or 
                pd.isna(slowk[-2]) or pd.isna(slowd[-2])):
                self.log_signal(-1, "Insufficient data for Stochastic calculation", data)
                return -1
            
            current_k = slowk[-1]
            current_d = slowd[-1]
            prev_k = slowk[-2]
            prev_d = slowd[-2]
            
            # Check for bullish signal - %K crosses above %D from oversold
            if (prev_k <= prev_d and current_k > current_d and 
                current_k < self.oversold_level + 10):  # Within oversold recovery zone
                volume_result = self.apply_volume_filtering(
                    1, data, signal_type='bullish', 
                    min_volume_factor=1.0  # Standard threshold for Stochastic signals
                )
                
                if not volume_result['volume_filtered']:
                    reason = f"Stochastic bullish crossover: %K ({current_k:.2f}) crosses above %D ({current_d:.2f}) from oversold - {volume_result['reason']}"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Stochastic signal filtered: {volume_result['reason']}"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Check for bearish signal - %K crosses below %D from overbought
            elif (prev_k >= prev_d and current_k < current_d and 
                  current_k > self.overbought_level - 10):  # Within overbought zone
                reason = f"Stochastic bearish crossover: %K ({current_k:.2f}) crosses below %D ({current_d:.2f}) from overbought"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check for oversold condition (potential buy)
            elif current_k < self.oversold_level and current_d < self.oversold_level:
                volume_result = self.apply_volume_filtering(
                    1, data, signal_type='bullish', 
                    min_volume_factor=0.8  # Lower threshold for oversold conditions
                )
                
                if not volume_result['volume_filtered']:
                    reason = f"Stochastic oversold: %K ({current_k:.2f}) and %D ({current_d:.2f}) both below {self.oversold_level} - {volume_result['reason']}"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Stochastic oversold but weak volume: {volume_result['reason']}"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Check for overbought condition (potential sell)
            elif current_k > self.overbought_level and current_d > self.overbought_level:
                reason = f"Stochastic overbought: %K ({current_k:.2f}) and %D ({current_d:.2f}) both above {self.overbought_level}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Neutral zone
            else:
                reason = f"Stochastic neutral: %K ({current_k:.2f}), %D ({current_d:.2f})"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Stochastic calculation: {str(e)}", data)
            return -1
