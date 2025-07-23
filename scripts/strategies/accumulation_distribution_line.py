"""
Accumulation Distribution Line Strategy
File: scripts/strategies/accumulation_distribution_line.py

This strategy uses the Accumulation/Distribution Line to identify buying and selling pressure.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Accumulation_Distribution_Line(BaseStrategy):
    """
    Accumulation Distribution Line Strategy.
    
    Buy Signal: A/D Line is rising (accumulation)
    Sell Signal: A/D Line is falling (distribution)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback_period = self.get_parameter('lookback_period', 5)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Accumulation Distribution Line strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.lookback_period + 1):
            return -1
            
        try:
            # Calculate Accumulation Distribution Line using TA-Lib
            high_prices = data['High'].values.astype(float)
            low_prices = data['Low'].values.astype(float)
            close_prices = data['Close'].values.astype(float)
            volume = data['Volume'].values.astype(float)
            
            ad_line = ta.AD(high_prices, low_prices, close_prices, volume)
            
            # Check if we have valid values
            if pd.isna(ad_line[-1]) or len(ad_line) < self.lookback_period + 1:
                self.log_signal(-1, "Insufficient data for A/D Line calculation", data)
                return -1
            
            current_ad = ad_line[-1]
            previous_ad = ad_line[-(self.lookback_period + 1)]
            short_term_ad = ad_line[-2]
            
            # Calculate A/D Line trend over lookback period
            ad_trend = current_ad - previous_ad
            short_term_trend = current_ad - short_term_ad
            
            # Calculate price trend over the same period
            current_price = close_prices[-1]
            previous_price = close_prices[-(self.lookback_period + 1)]
            price_trend = current_price - previous_price
            
            # Buy signal: A/D Line rising strongly (accumulation)
            if ad_trend > 0 and short_term_trend > 0:
                reason = f"Strong accumulation: A/D trend {ad_trend:.0f}, recent {short_term_trend:.0f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong buy signal: A/D Line rising while price flat/down (stealth accumulation)
            elif ad_trend > 0 and price_trend <= 0:
                reason = f"Stealth accumulation: A/D rising {ad_trend:.0f} while price flat/down {price_trend:.2f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: A/D Line falling (distribution)
            elif ad_trend < 0 and short_term_trend < 0:
                reason = f"Distribution: A/D trend {ad_trend:.0f}, recent {short_term_trend:.0f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Divergence signal: Price rising but A/D falling (bearish divergence)
            elif price_trend > 0 and ad_trend < 0:
                reason = f"Bearish divergence: Price up {price_trend:.2f} but A/D down {ad_trend:.0f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check shorter-term trend when longer trend is unclear
            elif short_term_trend > 0:
                reason = f"Recent accumulation: A/D short-term trend {short_term_trend:.0f}"
                self.log_signal(1, reason, data)
                return 1
            
            elif short_term_trend < 0:
                reason = f"Recent distribution: A/D short-term trend {short_term_trend:.0f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Neutral case
            else:
                reason = f"Neutral A/D Line: trend {ad_trend:.0f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in A/D Line calculation: {str(e)}", data)
            return -1
