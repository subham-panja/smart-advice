"""
Advanced Chart Pattern Recognition Strategy
File: scripts/strategies/chart_patterns.py

This strategy identifies and analyzes advanced chart patterns crucial for swing trading:
- Inside Bars (consolidation patterns)
- NR7 (Narrow Range 7) patterns
- Advanced Doji variations (Dragonfly, Gravestone)
- Multi-candlestick patterns (Harami, Morning/Evening Star)
- Supply and Demand zones
"""

import pandas as pd
import numpy as np
import talib as ta
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy

class ChartPatterns(BaseStrategy):
    """
    Advanced Chart Pattern Recognition for Swing Trading.
    
    This strategy identifies multiple chart patterns and provides confluence scoring
    based on the strength and combination of detected patterns.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback_period = self.get_parameter('lookback_period', 20)
        self.nr7_lookback = self.get_parameter('nr7_lookback', 7)
        self.min_pattern_strength = self.get_parameter('min_pattern_strength', 0.6)
        self.volume_confirmation = self.get_parameter('volume_confirmation', True)
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the chart pattern recognition strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for strong bullish patterns, -1 for bearish/no patterns
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.lookback_period):
            return -1
            
        try:
            patterns_detected = []
            pattern_strength = 0
            
            # 1. Check for Inside Bar patterns
            inside_bar_signal = self._detect_inside_bars(data)
            if inside_bar_signal:
                patterns_detected.append(inside_bar_signal)
                pattern_strength += inside_bar_signal['strength']
            
            # 2. Check for NR7 (Narrow Range 7) patterns
            nr7_signal = self._detect_nr7_pattern(data)
            if nr7_signal:
                patterns_detected.append(nr7_signal)
                pattern_strength += nr7_signal['strength']
            
            # 3. Check for advanced Doji patterns
            doji_signal = self._detect_advanced_doji(data)
            if doji_signal:
                patterns_detected.append(doji_signal)
                pattern_strength += doji_signal['strength']
            
            # 4. Check for Harami patterns
            harami_signal = self._detect_harami_pattern(data)
            if harami_signal:
                patterns_detected.append(harami_signal)
                pattern_strength += harami_signal['strength']
            
            # 5. Check for Morning/Evening Star patterns
            star_signal = self._detect_star_patterns(data)
            if star_signal:
                patterns_detected.append(star_signal)
                pattern_strength += star_signal['strength']
            
            # 6. Check for Supply/Demand zones
            supply_demand_signal = self._detect_supply_demand_zones(data)
            if supply_demand_signal:
                patterns_detected.append(supply_demand_signal)
                pattern_strength += supply_demand_signal['strength']
            
            # Enhanced volume confirmation using new system
            if self.volume_confirmation and patterns_detected:
                volume_factor = self._get_volume_confirmation(data)
                pattern_strength *= volume_factor
            
            # Generate signal based on pattern strength
            if pattern_strength >= self.min_pattern_strength:
                # Apply enhanced volume filtering
                initial_signal = 1
                volume_result = self.apply_volume_filtering(
                    initial_signal, data, signal_type='bullish', 
                    min_volume_factor=0.9  # Slightly lower threshold for patterns
                )
                
                if volume_result['volume_filtered']:
                    self.log_signal(-1, volume_result['reason'], data)
                    return -1
                else:
                    pattern_names = [p['name'] for p in patterns_detected]
                    reason = f"Strong chart patterns: {', '.join(pattern_names)} (Strength: {pattern_strength:.2f}) - {volume_result['reason']}"
                    self.log_signal(1, reason, data)
                    return 1
            else:
                if patterns_detected:
                    pattern_names = [p['name'] for p in patterns_detected]
                    reason = f"Weak patterns detected: {', '.join(pattern_names)} (Strength: {pattern_strength:.2f})"
                else:
                    reason = "No significant chart patterns detected"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in chart pattern analysis: {str(e)}", data)
            return -1
    
    def _detect_inside_bars(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Inside Bar patterns - bars with high/low contained within previous bar.
        
        Inside bars indicate consolidation and often precede breakouts.
        """
        try:
            if len(data) < 2:
                return None
            
            # Get last two bars
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Check if current bar is inside previous bar
            if (current['High'] <= previous['High'] and 
                current['Low'] >= previous['Low']):
                
                # Calculate pattern strength based on range compression
                current_range = current['High'] - current['Low']
                previous_range = previous['High'] - previous['Low']
                compression_ratio = current_range / previous_range if previous_range > 0 else 0
                
                # Stronger signal with more compression
                strength = max(0, 1 - compression_ratio) * 0.7  # Max 0.7 strength for inside bars
                
                return {
                    'name': 'Inside Bar',
                    'type': 'consolidation',
                    'strength': strength,
                    'compression_ratio': compression_ratio
                }
                
            return None
            
        except Exception as e:
            return None
    
    def _detect_nr7_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect NR7 (Narrow Range 7) patterns.
        
        NR7 occurs when the current bar has the narrowest range of the last 7 bars.
        """
        try:
            if len(data) < self.nr7_lookback:
                return None
            
            # Calculate ranges for last 7 bars
            recent_data = data.tail(self.nr7_lookback)
            ranges = recent_data['High'] - recent_data['Low']
            
            # Check if current bar has the narrowest range
            if ranges.iloc[-1] == ranges.min():
                # Calculate strength based on how much narrower it is
                avg_range = ranges.mean()
                current_range = ranges.iloc[-1]
                narrowness_ratio = current_range / avg_range if avg_range > 0 else 0
                
                # Stronger signal with more compression
                strength = max(0, 1 - narrowness_ratio) * 0.8  # Max 0.8 strength for NR7
                
                return {
                    'name': 'NR7',
                    'type': 'consolidation',
                    'strength': strength,
                    'narrowness_ratio': narrowness_ratio
                }
                
            return None
            
        except Exception as e:
            return None
    
    def _detect_advanced_doji(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect advanced Doji variations: Dragonfly and Gravestone.
        """
        try:
            if len(data) < 1:
                return None
            
            current = data.iloc[-1]
            open_price = current['Open']
            close_price = current['Close']
            high_price = current['High']
            low_price = current['Low']
            
            # Calculate body and wick sizes
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            if total_range == 0:
                return None
            
            # Doji threshold - body should be small relative to total range
            doji_threshold = 0.1  # Body < 10% of total range
            body_ratio = body_size / total_range
            
            if body_ratio <= doji_threshold:
                upper_wick_ratio = upper_wick / total_range
                lower_wick_ratio = lower_wick / total_range
                
                # Dragonfly Doji: Long lower wick, minimal upper wick
                if lower_wick_ratio > 0.6 and upper_wick_ratio < 0.2:
                    strength = lower_wick_ratio * 0.9  # Strong bullish signal
                    return {
                        'name': 'Dragonfly Doji',
                        'type': 'reversal_bullish',
                        'strength': strength,
                        'lower_wick_ratio': lower_wick_ratio
                    }
                
                # Gravestone Doji: Long upper wick, minimal lower wick
                elif upper_wick_ratio > 0.6 and lower_wick_ratio < 0.2:
                    # This is bearish, so we give it negative strength for our bullish strategy
                    return None  # Skip bearish patterns
                
                # Regular Doji: Balanced wicks
                elif abs(upper_wick_ratio - lower_wick_ratio) < 0.3:
                    strength = 0.4  # Moderate indecision signal
                    return {
                        'name': 'Regular Doji',
                        'type': 'indecision',
                        'strength': strength,
                        'body_ratio': body_ratio
                    }
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_harami_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Bullish Harami pattern - small body contained within previous large body.
        """
        try:
            if len(data) < 2:
                return None
            
            current = data.iloc[-1]
            previous = data.iloc[-2]
            
            # Calculate bodies
            current_body = abs(current['Close'] - current['Open'])
            previous_body = abs(previous['Close'] - previous['Open'])
            
            # Previous should be bearish (red) and current should be bullish (green)
            prev_bearish = previous['Close'] < previous['Open']
            curr_bullish = current['Close'] > current['Open']
            
            if not (prev_bearish and curr_bullish):
                return None
            
            # Current body should be contained within previous body
            if (current['Open'] > min(previous['Open'], previous['Close']) and
                current['Close'] < max(previous['Open'], previous['Close']) and
                current_body < previous_body * 0.7):  # Current body < 70% of previous
                
                # Calculate strength based on size ratio
                size_ratio = current_body / previous_body if previous_body > 0 else 0
                strength = (1 - size_ratio) * 0.8  # Smaller current body = stronger signal
                
                return {
                    'name': 'Bullish Harami',
                    'type': 'reversal_bullish',
                    'strength': strength,
                    'size_ratio': size_ratio
                }
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_star_patterns(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Morning Star patterns (3-candle bullish reversal).
        """
        try:
            if len(data) < 3:
                return None
            
            # Get last three bars
            first = data.iloc[-3]   # Should be bearish
            second = data.iloc[-2]  # Should be small (star)
            third = data.iloc[-1]   # Should be bullish
            
            # Check Morning Star pattern
            first_bearish = first['Close'] < first['Open']
            third_bullish = third['Close'] > third['Open']
            
            if not (first_bearish and third_bullish):
                return None
            
            # Second candle should be small and gap down
            second_body = abs(second['Close'] - second['Open'])
            first_body = abs(first['Close'] - first['Open'])
            third_body = abs(third['Close'] - third['Open'])
            
            # Star should be smaller than both other candles
            if (second_body < first_body * 0.5 and second_body < third_body * 0.5):
                # Check for gaps
                gap_down = second['High'] < first['Close']
                gap_up = third['Open'] > second['High']
                
                base_strength = 0.6
                if gap_down and gap_up:
                    base_strength = 0.9  # Perfect Morning Star with gaps
                elif gap_down or gap_up:
                    base_strength = 0.7  # Partial gaps
                
                return {
                    'name': 'Morning Star',
                    'type': 'reversal_bullish',
                    'strength': base_strength,
                    'has_gaps': gap_down and gap_up
                }
            
            return None
            
        except Exception as e:
            return None
    
    def _detect_supply_demand_zones(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Supply and Demand zones based on significant price levels with volume.
        """
        try:
            if len(data) < 20:
                return None
            
            # Use last 20 bars for analysis
            recent_data = data.tail(20)
            
            # Find significant highs and lows
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            volumes = recent_data['Volume'].values
            
            # Find peaks and troughs
            high_peaks, _ = find_peaks(highs, prominence=np.std(highs) * 0.5)
            low_troughs, _ = find_peaks(-lows, prominence=np.std(lows) * 0.5)
            
            current_price = recent_data['Close'].iloc[-1]
            
            # Check if current price is near a demand zone (previous low with high volume)
            for trough_idx in low_troughs:
                if trough_idx < len(recent_data) - 2:  # Not the last bar
                    zone_price = lows[trough_idx]
                    zone_volume = volumes[trough_idx]
                    avg_volume = np.mean(volumes)
                    
                    # Price within 2% of demand zone and volume was above average
                    if (abs(current_price - zone_price) / zone_price < 0.02 and
                        zone_volume > avg_volume * 1.2):
                        
                        volume_strength = min(2.0, zone_volume / avg_volume) / 2.0  # Normalize
                        proximity_strength = 1 - (abs(current_price - zone_price) / zone_price) / 0.02
                        
                        strength = (volume_strength + proximity_strength) / 2 * 0.7
                        
                        return {
                            'name': 'Demand Zone',
                            'type': 'support_bullish',
                            'strength': strength,
                            'zone_price': zone_price,
                            'volume_ratio': zone_volume / avg_volume
                        }
            
            return None
            
        except Exception as e:
            return None
    
    def _get_volume_confirmation(self, data: pd.DataFrame) -> float:
        """
        Get volume confirmation factor for pattern strength.
        
        Returns multiplier between 0.5 and 1.5 based on current volume vs average.
        """
        try:
            if len(data) < 10:
                return 1.0
            
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].tail(10).mean()
            
            if avg_volume == 0:
                return 1.0
            
            volume_ratio = current_volume / avg_volume
            
            # Higher volume strengthens the signal, lower volume weakens it
            if volume_ratio >= 1.5:
                return 1.3  # Strong volume confirmation
            elif volume_ratio >= 1.2:
                return 1.1  # Moderate volume confirmation
            elif volume_ratio >= 0.8:
                return 1.0  # Normal volume
            elif volume_ratio >= 0.5:
                return 0.8  # Low volume warning
            else:
                return 0.6  # Very low volume - pattern less reliable
                
        except Exception as e:
            return 1.0
