"""
Pivot Points Bounce Strategy
File: scripts/strategies/pivot_points_bounce.py

This strategy uses pivot points (support and resistance levels) to identify
potential bounce opportunities when price approaches these key levels.
"""

import pandas as pd
import numpy as np
from scripts.strategies.base_strategy import BaseStrategy


class Pivot_Points_Bounce(BaseStrategy):
    """
    Strategy based on pivot point bounces.
    
    Pivot Points calculation:
    - Pivot Point (PP) = (High + Low + Close) / 3
    - Resistance 1 (R1) = (2 * PP) - Low
    - Support 1 (S1) = (2 * PP) - High
    - Resistance 2 (R2) = PP + (High - Low)
    - Support 2 (S2) = PP - (High - Low)
    
    Signals:
    - Price bouncing off support levels: Buy signal
    - Price bouncing off resistance levels: Sell signal
    - Price breaking through levels: Continuation signal
    """
    
    def __init__(self, params=None):
        """
        Initialize the Pivot Points Bounce strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - period: Period for pivot calculation (default: 1 for daily pivots)
                   - bounce_threshold: Distance threshold for bounce detection (default: 0.5%)
                   - break_threshold: Distance threshold for breakout confirmation (default: 0.3%)
                   - min_approach_distance: Minimum approach distance to pivot (default: 1%)
        """
        super().__init__(params)
        self.period = self.get_parameter('period', 1)
        self.bounce_threshold = self.get_parameter('bounce_threshold', 0.005)  # 0.5%
        self.break_threshold = self.get_parameter('break_threshold', 0.003)    # 0.3%
        self.min_approach_distance = self.get_parameter('min_approach_distance', 0.01)  # 1%
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Pivot Points Bounce strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=5):
            self.log_signal(-1, "Insufficient data for Pivot Points analysis", data)
            return -1
        
        try:
            # Calculate pivot points based on previous period
            if len(data) < 2:
                self.log_signal(-1, "Need at least 2 periods for pivot calculation", data)
                return -1
            
            # Use previous day's high, low, close for pivot calculation
            prev_high = data['High'].iloc[-2]
            prev_low = data['Low'].iloc[-2]
            prev_close = data['Close'].iloc[-2]
            
            # Calculate pivot levels
            pivot_point = (prev_high + prev_low + prev_close) / 3
            
            # Support and Resistance levels
            r1 = (2 * pivot_point) - prev_low
            s1 = (2 * pivot_point) - prev_high
            r2 = pivot_point + (prev_high - prev_low)
            s2 = pivot_point - (prev_high - prev_low)
            
            # Additional levels (mid-points)
            mid_r1 = (pivot_point + r1) / 2
            mid_s1 = (pivot_point + s1) / 2
            
            # Current price data
            current_close = data['Close'].iloc[-1]
            current_high = data['High'].iloc[-1]
            current_low = data['Low'].iloc[-1]
            
            # Previous price for bounce detection
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_close
            
            # Organize pivot levels
            pivot_levels = {
                'S2': s2,
                'S1': s1,
                'Mid_S1': mid_s1,
                'PP': pivot_point,
                'Mid_R1': mid_r1,
                'R1': r1,
                'R2': r2
            }
            
            # Find the closest support and resistance levels
            supports = {k: v for k, v in pivot_levels.items() if v < current_close}
            resistances = {k: v for k, v in pivot_levels.items() if v > current_close}
            
            closest_support = max(supports.values()) if supports else None
            closest_resistance = min(resistances.values()) if resistances else None
            
            closest_support_name = None
            closest_resistance_name = None
            
            if closest_support:
                closest_support_name = [k for k, v in supports.items() if v == closest_support][0]
            if closest_resistance:
                closest_resistance_name = [k for k, v in resistances.items() if v == closest_resistance][0]
            
            # Check for bounce patterns
            
            # Support bounce (bullish signal)
            if closest_support:
                support_distance = abs(current_close - closest_support) / current_close
                
                # Check if price approached support and bounced
                if support_distance <= self.bounce_threshold:
                    # Confirm bounce: current price above support, previous price was closer to support
                    if (current_close > closest_support and 
                        current_low <= closest_support * (1 + self.bounce_threshold)):
                        
                        # Additional confirmation: price moving away from support
                        if current_close > prev_price:
                            bounce_strength = (current_close - current_low) / current_close
                            self.log_signal(1, f"Support bounce at {closest_support_name}({closest_support:.2f}): bounce strength {bounce_strength*100:.2f}%", data)
                            return 1
                        else:
                            self.log_signal(-1, f"Weak support bounce: price declining despite support", data)
                            return -1
                
                # Check if approaching support for potential bounce
                elif support_distance <= self.min_approach_distance:
                    # Price approaching support - potential bounce setup
                    approach_momentum = (current_close - prev_price) / prev_price
                    if approach_momentum > -0.01:  # Not falling too fast
                        self.log_signal(1, f"Approaching {closest_support_name} support({closest_support:.2f}): distance {support_distance*100:.2f}%", data)
                        return 1
            
            # Resistance bounce (bearish signal)
            if closest_resistance:
                resistance_distance = abs(current_close - closest_resistance) / current_close
                
                # Check if price approached resistance and bounced down
                if resistance_distance <= self.bounce_threshold:
                    if (current_close < closest_resistance and 
                        current_high >= closest_resistance * (1 - self.bounce_threshold)):
                        
                        # Confirm bearish bounce
                        if current_close < prev_price:
                            bounce_weakness = (current_high - current_close) / current_close
                            self.log_signal(-1, f"Resistance rejection at {closest_resistance_name}({closest_resistance:.2f}): weakness {bounce_weakness*100:.2f}%", data)
                            return -1
            
            # Check for breakouts
            
            # Bullish breakout above resistance
            if closest_resistance:
                if current_close > closest_resistance * (1 + self.break_threshold):
                    # Confirmed breakout above resistance
                    breakout_strength = (current_close - closest_resistance) / closest_resistance
                    
                    # Additional confirmation: volume or momentum
                    if current_close > prev_price:  # Price momentum confirmation
                        self.log_signal(1, f"Bullish breakout above {closest_resistance_name}({closest_resistance:.2f}): strength {breakout_strength*100:.2f}%", data)
                        return 1
                    else:
                        self.log_signal(-1, f"False breakout above {closest_resistance_name}: price declining", data)
                        return -1
            
            # Bearish breakdown below support
            if closest_support:
                if current_close < closest_support * (1 - self.break_threshold):
                    breakdown_severity = (closest_support - current_close) / closest_support
                    self.log_signal(-1, f"Bearish breakdown below {closest_support_name}({closest_support:.2f}): severity {breakdown_severity*100:.2f}%", data)
                    return -1
            
            # Price between pivot levels - neutral zone analysis
            if closest_support and closest_resistance:
                level_range = closest_resistance - closest_support
                position_in_range = (current_close - closest_support) / level_range
                
                if position_in_range > 0.7:
                    # Near resistance - cautious bullish
                    self.log_signal(-1, f"Near resistance {closest_resistance_name}: position {position_in_range*100:.1f}% in range", data)
                    return -1
                elif position_in_range < 0.3:
                    # Near support - cautious bullish
                    self.log_signal(1, f"Near support {closest_support_name}: position {position_in_range*100:.1f}% in range", data)
                    return 1
                else:
                    # Middle zone - follow pivot point
                    if current_close > pivot_point:
                        self.log_signal(1, f"Above pivot point({pivot_point:.2f}): bullish bias", data)
                        return 1
                    else:
                        self.log_signal(-1, f"Below pivot point({pivot_point:.2f}): bearish bias", data)
                        return -1
            
            # Default case - compare with pivot point
            if current_close > pivot_point:
                pp_distance = (current_close - pivot_point) / pivot_point
                if pp_distance > 0.01:  # More than 1% above pivot
                    self.log_signal(1, f"Well above pivot point: {pp_distance*100:.2f}%", data)
                    return 1
                else:
                    self.log_signal(-1, f"Just above pivot point: {pp_distance*100:.2f}%", data)
                    return -1
            else:
                self.log_signal(-1, f"Below pivot point({pivot_point:.2f})", data)
                return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in Pivot Points analysis: {str(e)}", data)
            return -1
