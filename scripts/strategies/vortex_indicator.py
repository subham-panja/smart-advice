"""
Vortex Indicator Strategy
File: scripts/strategies/vortex_indicator.py

This strategy uses the Vortex Indicator to identify trend reversals and momentum.
"""

import pandas as pd
import numpy as np
import talib as ta
from .base_strategy import BaseStrategy

class Vortex_Indicator(BaseStrategy):
    """
    Vortex Indicator Strategy.
    
    Buy Signal: VI+ crosses above VI- (positive vortex momentum)
    Sell Signal: VI- crosses above VI+ (negative vortex momentum)
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.period = self.get_parameter('period', 14)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core Vortex Indicator strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.period + 1):
            return -1
            
        try:
            # Calculate Vortex Indicator manually (TA-Lib doesn't have VI)
            high_prices = data['High'].values
            low_prices = data['Low'].values
            close_prices = data['Close'].values
            
            # Calculate True Range
            tr = ta.TRANGE(high_prices, low_prices, close_prices)
            
            # Check if we have valid TR values
            if pd.isna(tr[-1]) or len(tr) < self.period + 1:
                self.log_signal(-1, "Insufficient data for Vortex calculation", data)
                return -1
            
            # Calculate Vortex Movement
            vm_plus = np.abs(high_prices[1:] - low_prices[:-1])
            vm_minus = np.abs(low_prices[1:] - high_prices[:-1])
            
            # Pad with NaN to match original length
            vm_plus = np.concatenate([[np.nan], vm_plus])
            vm_minus = np.concatenate([[np.nan], vm_minus])
            
            # Calculate VI+ and VI-
            vi_plus = []
            vi_minus = []
            
            for i in range(self.period - 1, len(tr)):
                sum_vm_plus = np.sum(vm_plus[i - self.period + 1:i + 1])
                sum_vm_minus = np.sum(vm_minus[i - self.period + 1:i + 1])
                sum_tr = np.sum(tr[i - self.period + 1:i + 1])
                
                if sum_tr != 0:
                    vi_plus.append(sum_vm_plus / sum_tr)
                    vi_minus.append(sum_vm_minus / sum_tr)
                else:
                    vi_plus.append(1.0)
                    vi_minus.append(1.0)
            
            # Convert to numpy arrays
            vi_plus = np.array(vi_plus)
            vi_minus = np.array(vi_minus)
            
            # Check if we have enough data
            if len(vi_plus) < 2 or len(vi_minus) < 2:
                self.log_signal(-1, "Insufficient data for VI calculation", data)
                return -1
            
            current_vi_plus = vi_plus[-1]
            current_vi_minus = vi_minus[-1]
            previous_vi_plus = vi_plus[-2]
            previous_vi_minus = vi_minus[-2]
            
            # Buy signal: VI+ crosses above VI-
            if previous_vi_plus <= previous_vi_minus and current_vi_plus > current_vi_minus:
                reason = f"Bullish VI crossover: VI+ {current_vi_plus:.3f} crosses above VI- {current_vi_minus:.3f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Sell signal: VI- crosses above VI+
            elif previous_vi_minus <= previous_vi_plus and current_vi_minus > current_vi_plus:
                reason = f"Bearish VI crossover: VI- {current_vi_minus:.3f} crosses above VI+ {current_vi_plus:.3f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Strong buy signal: VI+ significantly higher than VI-
            elif current_vi_plus > current_vi_minus * 1.1:  # 10% higher
                reason = f"Strong positive vortex: VI+ {current_vi_plus:.3f} >> VI- {current_vi_minus:.3f}"
                self.log_signal(1, reason, data)
                return 1
            
            # Strong sell signal: VI- significantly higher than VI+
            elif current_vi_minus > current_vi_plus * 1.1:  # 10% higher
                reason = f"Strong negative vortex: VI- {current_vi_minus:.3f} >> VI+ {current_vi_plus:.3f}"
                self.log_signal(-1, reason, data)
                return -1
            
            # Check current trend
            elif current_vi_plus > current_vi_minus:
                reason = f"Positive vortex trend: VI+ {current_vi_plus:.3f} > VI- {current_vi_minus:.3f}"
                self.log_signal(1, reason, data)
                return 1
            
            else:
                reason = f"Negative vortex trend: VI- {current_vi_minus:.3f} > VI+ {current_vi_plus:.3f}"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in Vortex Indicator calculation: {str(e)}", data)
            return -1
