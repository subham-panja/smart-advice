"""
Gap Trading Strategy
File: scripts/strategies/gap_trading.py

This strategy identifies and trades based on price gaps between sessions.
Focuses on gap-ups that indicate strong bullish momentum.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy

class Gap_Trading(BaseStrategy):
    """
    Gap Trading Strategy.
    
    Buy Signal: Bullish gap-up with volume confirmation
    Focus on gaps that are likely to continue rather than fill
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.min_gap_percent = self.get_parameter('min_gap_percent', 2.0)  # Minimum gap percentage
        self.max_gap_percent = self.get_parameter('max_gap_percent', 10.0)  # Maximum gap percentage (avoid news-driven spikes)
        self.volume_multiplier = self.get_parameter('volume_multiplier', 1.5)  # Volume should be 1.5x average
        self.volume_lookback = self.get_parameter('volume_lookback', 20)  # Days to calculate average volume
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the gap trading strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        min_periods = self.volume_lookback + 2
        if not self.validate_data(data, min_periods=min_periods):
            return -1
            
        try:
            # Get current and previous day data
            current_open = data['Open'].iloc[-1]
            previous_close = data['Close'].iloc[-2]
            current_high = data['High'].iloc[-1]
            current_low = data['Low'].iloc[-1]
            current_close = data['Close'].iloc[-1]
            current_volume = data['Volume'].iloc[-1]
            
            # Calculate average volume
            avg_volume = data['Volume'].tail(self.volume_lookback).mean()
            
            # Calculate gap percentage
            gap_percent = ((current_open - previous_close) / previous_close) * 100
            
            # Check for bullish gap
            if gap_percent >= self.min_gap_percent and gap_percent <= self.max_gap_percent:
                
                # Enhanced volume confirmation using new system
                volume_result = self.apply_volume_filtering(
                    1, data, signal_type='bullish', 
                    min_volume_factor=self.volume_multiplier  # Use configured multiplier
                )
                
                if not volume_result['volume_filtered']:
                    # Additional checks for gap continuation vs. fill
                    
                    # Gap holding strength - price should stay above gap level
                    gap_hold_strength = (current_low - previous_close) / previous_close
                    
                    # Price action within the day
                    intraday_strength = (current_close - current_open) / current_open * 100
                    
                    # Strong gap: holds above previous close and shows positive intraday action
                    if gap_hold_strength > 0 and intraday_strength >= -1.0:  # Allow small intraday pullback
                        reason = f"Strong gap: {gap_percent:.2f}% gap-up, gap holding - {volume_result['reason']}"
                        self.log_signal(1, reason, data)
                        return 1
                    
                    # Moderate gap: some weakness but still above previous close
                    elif gap_hold_strength > -0.5 and current_close > previous_close:
                        reason = f"Moderate gap: {gap_percent:.2f}% gap-up, some filling but close above previous - {volume_result['reason']}"
                        self.log_signal(1, reason, data)
                        return 1
                    
                    # Weak gap: significant gap filling
                    else:
                        reason = f"Gap filling: {gap_percent:.2f}% gap but filling significantly, gap_hold: {gap_hold_strength:.2f}%"
                        self.log_signal(-1, reason, data)
                        return -1
                
                # Gap without volume confirmation
                else:
                    reason = f"Gap without volume confirmation: {gap_percent:.2f}% gap - {volume_result['reason']}"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Small positive gap (less than minimum threshold)
            elif gap_percent > 0.5 and gap_percent < self.min_gap_percent:
                # Check if it's part of a strong uptrend
                recent_performance = (current_close - data['Close'].iloc[-5]) / data['Close'].iloc[-5] * 100
                
                if recent_performance > 5.0 and current_volume > avg_volume:
                    reason = f"Small gap in uptrend: {gap_percent:.2f}% gap, {recent_performance:.1f}% 5-day performance"
                    self.log_signal(1, reason, data)
                    return 1
                else:
                    reason = f"Insignificant gap: {gap_percent:.2f}% gap, not enough momentum"
                    self.log_signal(-1, reason, data)
                    return -1
            
            # Gap down or no gap
            elif gap_percent < -1.0:
                reason = f"Gap down: {gap_percent:.2f}% negative gap"
                self.log_signal(-1, reason, data)
                return -1
            
            # No significant gap
            else:
                reason = f"No significant gap: {gap_percent:.2f}% gap"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in gap trading calculation: {str(e)}", data)
            return -1
