"""
Stochastic K-D Crossover Strategy
File: scripts/strategies/stochastic_k_d_crossover.py

This strategy uses Stochastic %K and %D crossovers to identify signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Stochastic_K_D_Crossover(BaseStrategy):
    """
    Stochastic %K-%D Crossover Strategy.
    
    Buy Signal: %K crosses above %D
    Sell Signal: %K crosses below %D
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.k_period = self.get_parameter('k_period', 14)
        self.d_period = self.get_parameter('d_period', 3)
        self.oversold_level = self.get_parameter('oversold_level', 20)
        self.overbought_level = self.get_parameter('overbought_level', 80)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Stochastic K-D crossover strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.k_period + self.d_period):
            return -1
            
        try:
            # Calculate Stochastic using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            slowk, slowd = ta.STOCH(
                high_prices, low_prices, close_prices,
                fastk_period=self.k_period,
                slowk_period=self.d_period,
                slowk_matype=0,
                slowd_period=self.d_period,
                slowd_matype=0
            )
            
            # Check if we have valid values
            if pd.isna(slowk[-1]) or pd.isna(slowd[-1]) or pd.isna(slowk[-2]) or pd.isna(slowd[-2]):
                self.log_signal(-1, "Insufficient data for Stochastic calculation", data)
                return -1
            
            current_k = slowk[-1]
            current_d = slowd[-1]
            previous_k = slowk[-2]
            previous_d = slowd[-2]
            
            # Buy signal: %K crosses above %D in oversold region
            if (previous_k <= previous_d and current_k > current_d and 
                current_k < self.oversold_level + 10):  # Within 10 points of oversold
                reason = f"Bullish Stoch crossover in oversold: %K {current_k:.2f} crosses above %D {current_d:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: Both in oversold and %K crosses above %D
            elif (previous_k <= previous_d and current_k > current_d and 
                  current_k < self.oversold_level):
                reason = f"Strong bullish crossover: %K {current_k:.2f} > %D {current_d:.2f} in oversold region"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: %K crosses below %D in overbought region
            elif (previous_k >= previous_d and current_k < current_d and 
                  current_k > self.overbought_level - 10):  # Within 10 points of overbought
                reason = f"Bearish Stoch crossover in overbought: %K {current_k:.2f} crosses below %D {current_d:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong sell signal: Both in overbought and %K crosses below %D
            elif (previous_k >= previous_d and current_k < current_d and 
                  current_k > self.overbought_level):
                reason = f"Strong bearish crossover: %K {current_k:.2f} < %D {current_d:.2f} in overbought region"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current position and trend
            elif current_k > current_d and current_k < self.overbought_level:
                reason = f"Bullish Stoch trend: %K {current_k:.2f} > %D {current_d:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            elif current_k < current_d and current_k > self.oversold_level:
                reason = f"Bearish Stoch trend: %K {current_k:.2f} < %D {current_d:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # In oversold region
            elif current_k < self.oversold_level:
                reason = f"Stochastic oversold: %K {current_k:.2f}, %D {current_d:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # In overbought region
            elif current_k > self.overbought_level:
                reason = f"Stochastic overbought: %K {current_k:.2f}, %D {current_d:.2f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Neutral zone
            else:
                reason = f"Stochastic neutral: %K {current_k:.2f}, %D {current_d:.2f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Stochastic K-D crossover calculation: {str(e)}", data)
            return -1
