"""
Directional Indicator Crossover Strategy
File: scripts/strategies/di_crossover.py

This strategy uses the Directional Indicator (+DI and -DI) crossovers to identify
trend changes and generate buy/sell signals. Part of the ADX indicator system.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy


class DI_Crossover(BaseStrategy):
    """
    Strategy based on Directional Indicator (+DI and -DI) crossovers.
    
    DI Crossover signals:
    - +DI crossing above -DI: Bullish signal (buy)
    - -DI crossing above +DI: Bearish signal (sell)
    - ADX can be used to filter signals (stronger trends)
    - Multiple confirmations improve signal reliability
    """
    
    def __init__(self, params=None):
        """
        Initialize the DI Crossover strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - period: DI calculation period (default: 14)
                   - min_adx: Minimum ADX for signal validation (default: 20)
                   - di_separation: Minimum separation between DIs (default: 2)
                   - confirmation_periods: Periods to confirm crossover (default: 2)
        """
        super().__init__(params)
        self.period = self.get_parameter('period', 14)
        self.min_adx = self.get_parameter('min_adx', 20)
        self.di_separation = self.get_parameter('di_separation', 2)
        self.confirmation_periods = self.get_parameter('confirmation_periods', 2)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the DI Crossover strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=self.period + self.confirmation_periods + 5):
            self.log_signal(-1, "Insufficient data for DI analysis", data)
            return -1
        
        try:
            # Calculate Directional Indicators using TA-Lib
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            # Calculate +DI, -DI, and ADX
            plus_di = ta.PLUS_DI(high, low, close, timeperiod=self.period)
            minus_di = ta.MINUS_DI(high, low, close, timeperiod=self.period)
            adx = ta.ADX(high, low, close, timeperiod=self.period)
            
            # Check for sufficient data
            if (len(plus_di) < self.confirmation_periods + 1 or 
                np.isnan(plus_di[-1]) or np.isnan(minus_di[-1]) or np.isnan(adx[-1])):
                self.log_signal(-1, "Insufficient DI data", data)
                return -1
            
            # Get recent values
            current_plus_di = plus_di[-1]
            current_minus_di = minus_di[-1]
            current_adx = adx[-1]
            
            prev_plus_di = plus_di[-2]
            prev_minus_di = minus_di[-2]
            
            # Check ADX strength filter
            if current_adx < self.min_adx:
                self.log_signal(-1, f"Weak trend strength: ADX {current_adx:.2f} < {self.min_adx}", data)
                return -1
            
            # Check for bullish crossover: +DI crosses above -DI
            if prev_plus_di <= prev_minus_di and current_plus_di > current_minus_di:
                # Confirm the separation is meaningful
                di_difference = current_plus_di - current_minus_di
                if di_difference >= self.di_separation:
                    # Additional confirmation: check if trend is sustained
                    confirmation_count = 0
                    for i in range(1, min(self.confirmation_periods + 1, len(plus_di))):
                        if plus_di[-i] > minus_di[-i]:
                            confirmation_count += 1
                    
                    if confirmation_count >= self.confirmation_periods - 1:
                        self.log_signal(1, f"Bullish DI crossover: +DI({current_plus_di:.2f}) > -DI({current_minus_di:.2f}), ADX:{current_adx:.2f}", data)
                        return 1
                    else:
                        self.log_signal(-1, f"DI crossover lacks confirmation: {confirmation_count}/{self.confirmation_periods-1}", data)
                        return -1
                else:
                    self.log_signal(-1, f"Insufficient DI separation: {di_difference:.2f} < {self.di_separation}", data)
                    return -1
            
            # Check for bearish crossover: -DI crosses above +DI
            elif prev_minus_di <= prev_plus_di and current_minus_di > current_plus_di:
                di_difference = current_minus_di - current_plus_di
                if di_difference >= self.di_separation:
                    self.log_signal(-1, f"Bearish DI crossover: -DI({current_minus_di:.2f}) > +DI({current_plus_di:.2f}), ADX:{current_adx:.2f}", data)
                    return -1
            
            # Check current trend direction
            if current_plus_di > current_minus_di:
                # Bullish trend continuation
                di_spread = current_plus_di - current_minus_di
                if di_spread >= self.di_separation * 2:  # Strong bullish trend
                    self.log_signal(1, f"Strong bullish trend: +DI({current_plus_di:.2f}) >> -DI({current_minus_di:.2f}), spread:{di_spread:.2f}", data)
                    return 1
                elif di_spread >= self.di_separation:  # Moderate bullish trend
                    # Check if ADX is rising (strengthening trend)
                    if len(adx) >= 3 and adx[-1] > adx[-2]:
                        self.log_signal(1, f"Strengthening bullish trend: +DI lead, rising ADX({current_adx:.2f})", data)
                        return 1
                    else:
                        self.log_signal(-1, f"Weak bullish trend: +DI({current_plus_di:.2f}) > -DI({current_minus_di:.2f}) but weakening", data)
                        return -1
                else:
                    self.log_signal(-1, f"Marginal +DI lead: spread {di_spread:.2f} too small", data)
                    return -1
            else:
                # Bearish trend
                di_spread = current_minus_di - current_plus_di
                self.log_signal(-1, f"Bearish trend: -DI({current_minus_di:.2f}) > +DI({current_plus_di:.2f}), spread:{di_spread:.2f}", data)
                return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in DI analysis: {str(e)}", data)
            return -1
