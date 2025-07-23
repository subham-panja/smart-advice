"""
ATR Volatility Strategy
File: scripts/strategies/atr_volatility.py

This strategy uses Average True Range to identify volatility-based signals.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class ATR_Volatility(BaseStrategy):
    """
    ATR Volatility Strategy.
    
    Buy Signal: Low volatility (ATR) suggesting potential breakout
    Sell Signal: High volatility suggesting potential reversal
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 14)
        self.lookback_period = self.get_parameter('lookback_period', 20)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the ATR Volatility strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=max(self.period, self.lookback_period) + 1):
            return -1
            
        try:
            # Calculate ATR using TA-Lib
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            atr = ta.ATR(high_prices, low_prices, close_prices, timeperiod=self.period)
            
            # Check if we have valid ATR values
            if pd.isna(atr[-1]) or len(atr) < self.lookback_period:
                self.log_signal(-1, "Insufficient data for ATR calculation", data)
                return -1
            
            current_atr = atr[-1]
            current_price = close_prices[-1]
            
            # Calculate ATR as percentage of price
            atr_percent = (current_atr / current_price) * 100
            
            # Calculate average ATR over lookback period
            recent_atr = atr[-self.lookback_period:]
            recent_prices = close_prices[-self.lookback_period:]
            avg_atr_percent = np.mean([(atr_val / price) * 100 for atr_val, price in zip(recent_atr, recent_prices)])
            
            # Buy signal: ATR is below average (low volatility, potential breakout setup)
            if atr_percent < avg_atr_percent * 0.8:  # 20% below average
                reason = f"Low volatility setup: ATR {atr_percent:.2f}% vs avg {avg_atr_percent:.2f}%"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: ATR is significantly above average (high volatility, potential reversal)
            elif atr_percent > avg_atr_percent * 1.5:  # 50% above average
                reason = f"High volatility warning: ATR {atr_percent:.2f}% vs avg {avg_atr_percent:.2f}%"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check ATR trend
            atr_trend = atr[-1] - atr[-5] if len(atr) >= 5 else 0
            
            if atr_trend < 0 and atr_percent < avg_atr_percent:
                reason = f"Decreasing volatility: ATR trend {atr_trend:.4f}, current {atr_percent:.2f}%"
                self.log_signal(1, reason, data)
                return 1
            else:
                reason = f"Neutral/increasing volatility: ATR {atr_percent:.2f}%"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in ATR calculation: {str(e)}", data)
            return -1
