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
            
            # 7. Check for Bull Flag patterns
            flag_signal = self._detect_bull_flag_pattern(data)
            if flag_signal:
                patterns_detected.append(flag_signal)
                pattern_strength += flag_signal['strength']
            
            # 8. Check for Triangle patterns
            triangle_signal = self._detect_triangle_patterns(data)
            if triangle_signal:
                patterns_detected.append(triangle_signal)
                pattern_strength += triangle_signal['strength']
            
            # 9. Check for Head and Shoulders patterns
            hs_signal = self._detect_head_shoulders_patterns(data)
            if hs_signal:
                patterns_detected.append(hs_signal)
                pattern_strength += hs_signal['strength']
            
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
    
    def _detect_bull_flag_pattern(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Bull Flag patterns - strong uptrend followed by consolidation.
        
        A bull flag consists of:
        1. Strong uptrend (flagpole)
        2. Brief consolidation with declining volume (flag)
        3. Breakout above consolidation with increased volume
        """
        try:
            if len(data) < 15:  # Need at least 15 bars for pattern analysis
                return None
            
            # Get recent data for analysis
            recent_data = data.tail(15)
            closes = recent_data['Close'].values
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            volumes = recent_data['Volume'].values
            
            # 1. Check for strong uptrend (flagpole) in first part
            flagpole_start = 0
            flagpole_end = 7  # First 8 bars for flagpole
            
            flagpole_gain = (closes[flagpole_end] - closes[flagpole_start]) / closes[flagpole_start]
            
            # Require at least 5% gain for flagpole
            if flagpole_gain < 0.05:
                return None
            
            # 2. Check for consolidation (flag) in recent bars
            flag_start = flagpole_end + 1
            flag_data = recent_data.iloc[flag_start:]
            
            if len(flag_data) < 5:  # Need at least 5 bars for flag
                return None
            
            flag_highs = flag_data['High'].values
            flag_lows = flag_data['Low'].values
            flag_volumes = flag_data['Volume'].values
            
            # Calculate consolidation range
            flag_high = np.max(flag_highs)
            flag_low = np.min(flag_lows)
            flag_range = (flag_high - flag_low) / flag_low
            
            # Flag should be a tight consolidation (< 5% range)
            if flag_range > 0.05:
                return None
            
            # 3. Check volume pattern - should decline during consolidation
            avg_flagpole_volume = np.mean(volumes[flagpole_start:flagpole_end+1])
            avg_flag_volume = np.mean(flag_volumes[:-1])  # Exclude current bar
            current_volume = volumes[-1]
            
            # Volume should decline during flag formation
            volume_decline = avg_flag_volume < avg_flagpole_volume * 0.8
            
            # 4. Check for potential breakout
            current_price = closes[-1]
            breakout_level = flag_high
            
            # Check if price is near or above breakout level
            near_breakout = current_price >= breakout_level * 0.98
            
            if not (volume_decline and near_breakout):
                return None
            
            # Calculate pattern strength
            flagpole_strength = min(flagpole_gain * 10, 1.0)  # Scale gain to 0-1
            consolidation_strength = max(0, 1 - flag_range * 20)  # Tighter = stronger
            volume_strength = min(avg_flagpole_volume / avg_flag_volume, 2.0) / 2.0
            
            # Check for volume confirmation on current bar
            volume_breakout = current_volume > avg_flag_volume * 1.2
            volume_multiplier = 1.2 if volume_breakout else 1.0
            
            strength = (flagpole_strength + consolidation_strength + volume_strength) / 3 * 0.85 * volume_multiplier
            
            return {
                'name': 'Bull Flag',
                'type': 'continuation_bullish',
                'strength': min(strength, 1.0),
                'flagpole_gain': flagpole_gain,
                'flag_range': flag_range,
                'volume_decline': volume_decline,
                'breakout_level': breakout_level
            }
            
        except Exception as e:
            return None
    
    def _detect_triangle_patterns(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Triangle patterns - converging trendlines indicating consolidation.
        
        Types: Ascending, Descending, Symmetrical triangles
        """
        try:
            if len(data) < 20:  # Need sufficient data for triangle analysis
                return None
            
            # Get recent data for analysis
            recent_data = data.tail(20)
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            closes = recent_data['Close'].values
            volumes = recent_data['Volume'].values
            
            # Find significant peaks and troughs
            high_peaks, _ = find_peaks(highs, prominence=np.std(highs) * 0.3, distance=3)
            low_troughs, _ = find_peaks(-lows, prominence=np.std(lows) * 0.3, distance=3)
            
            # Need at least 2 peaks and 2 troughs
            if len(high_peaks) < 2 or len(low_troughs) < 2:
                return None
            
            # Get the most recent peaks and troughs
            recent_peaks = high_peaks[-2:] if len(high_peaks) >= 2 else high_peaks
            recent_troughs = low_troughs[-2:] if len(low_troughs) >= 2 else low_troughs
            
            # Calculate trendline slopes
            if len(recent_peaks) >= 2:
                peak_slope = (highs[recent_peaks[-1]] - highs[recent_peaks[-2]]) / (recent_peaks[-1] - recent_peaks[-2])
            else:
                peak_slope = 0
            
            if len(recent_troughs) >= 2:
                trough_slope = (lows[recent_troughs[-1]] - lows[recent_troughs[-2]]) / (recent_troughs[-1] - recent_troughs[-2])
            else:
                trough_slope = 0
            
            # Determine triangle type
            triangle_type = None
            strength_base = 0.6
            
            # Ascending Triangle: Horizontal resistance, rising support
            if abs(peak_slope) < 0.001 and trough_slope > 0.001:  # Flat top, rising bottom
                triangle_type = 'Ascending Triangle'
                strength_base = 0.8  # Bullish pattern
                
            # Descending Triangle: Falling resistance, horizontal support
            elif peak_slope < -0.001 and abs(trough_slope) < 0.001:  # Falling top, flat bottom
                triangle_type = 'Descending Triangle'
                strength_base = 0.3  # Bearish pattern - lower strength for our bullish strategy
                
            # Symmetrical Triangle: Converging trendlines
            elif peak_slope < -0.001 and trough_slope > 0.001:  # Falling top, rising bottom
                triangle_type = 'Symmetrical Triangle'
                strength_base = 0.6  # Neutral pattern
            
            if not triangle_type:
                return None
            
            # Check for breakout potential
            current_price = closes[-1]
            resistance_level = highs[recent_peaks[-1]] if len(recent_peaks) > 0 else np.max(highs[-5:])
            support_level = lows[recent_troughs[-1]] if len(recent_troughs) > 0 else np.min(lows[-5:])
            
            triangle_range = (resistance_level - support_level) / support_level
            
            # Triangle should show convergence (narrowing range)
            if triangle_range < 0.02 or triangle_range > 0.08:  # Too narrow or too wide
                return None
            
            # Check volume pattern - should decline during formation
            early_volume = np.mean(volumes[:10])
            recent_volume = np.mean(volumes[-5:])
            volume_decline = recent_volume < early_volume * 0.8
            
            # Check proximity to breakout
            distance_to_resistance = (resistance_level - current_price) / current_price
            distance_to_support = (current_price - support_level) / current_price
            
            # Bullish patterns get higher strength when near resistance
            if triangle_type in ['Ascending Triangle', 'Symmetrical Triangle']:
                if distance_to_resistance < 0.02:  # Near resistance breakout
                    proximity_bonus = 0.2
                else:
                    proximity_bonus = 0
            else:
                proximity_bonus = 0
            
            # Calculate final strength
            convergence_strength = max(0, 1 - triangle_range * 10)  # Tighter = stronger
            volume_strength = 0.1 if volume_decline else 0
            
            final_strength = min(strength_base + convergence_strength * 0.2 + volume_strength + proximity_bonus, 1.0)
            
            # Only return bullish or neutral patterns
            if triangle_type == 'Descending Triangle':
                return None  # Skip bearish patterns
            
            return {
                'name': triangle_type,
                'type': 'consolidation',
                'strength': final_strength,
                'resistance_level': resistance_level,
                'support_level': support_level,
                'triangle_range': triangle_range,
                'volume_decline': volume_decline
            }
            
        except Exception as e:
            return None
    
    def _detect_head_shoulders_patterns(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Inverse Head and Shoulders patterns - a bullish reversal pattern.
        
        The pattern consists of:
        1. Left shoulder (a trough)
        2. Head (a lower trough)
        3. Right shoulder (a trough higher than the head)
        4. Neckline (resistance connecting the peaks between the troughs)
        """
        try:
            if len(data) < 25:  # Need enough data for H&S analysis
                return None
            
            # Get recent data for analysis
            recent_data = data.tail(25)
            highs = recent_data['High'].values
            lows = recent_data['Low'].values
            closes = recent_data['Close'].values
            
            # Find significant peaks and troughs
            high_peaks, _ = find_peaks(highs, prominence=np.std(highs) * 0.4, distance=4)
            low_troughs, _ = find_peaks(-lows, prominence=np.std(lows) * 0.4, distance=4)
            
            # Need at least 3 troughs and 2 peaks for an inverse H&S
            if len(low_troughs) < 3 or len(high_peaks) < 2:
                return None
            
            # Identify potential shoulders and head
            left_shoulder_idx = low_troughs[-3]
            head_idx = low_troughs[-2]
            right_shoulder_idx = low_troughs[-1]
            
            left_shoulder = lows[left_shoulder_idx]
            head = lows[head_idx]
            right_shoulder = lows[right_shoulder_idx]
            
            # Basic H&S structure checks
            if not (head < left_shoulder and head < right_shoulder):
                return None
            
            # Shoulders should be roughly at the same level
            if abs(left_shoulder - right_shoulder) / right_shoulder > 0.05:  # Less than 5% difference
                return None
            
            # Identify peaks for the neckline
            peak1_idx = high_peaks[np.where(high_peaks > left_shoulder_idx)[0][0]]
            peak2_idx = high_peaks[np.where(high_peaks > head_idx)[0][0]]
            
            if peak1_idx >= peak2_idx:
                return None
            
            # Calculate neckline
            neckline_p1 = (peak1_idx, highs[peak1_idx])
            neckline_p2 = (peak2_idx, highs[peak2_idx])
            neckline_slope = (neckline_p2[1] - neckline_p1[1]) / (neckline_p2[0] - neckline_p1[0])
            neckline_intercept = neckline_p1[1] - neckline_slope * neckline_p1[0]
            
            # Check if current price is breaking the neckline
            current_price = closes[-1]
            current_neckline_level = neckline_slope * (len(recent_data) - 1) + neckline_intercept
            
            if current_price < current_neckline_level * 0.98:  # Price must be close to or above neckline
                return None
            
            # Volume confirmation: should increase on neckline breakout
            avg_volume_shoulders = np.mean(data['Volume'].iloc[left_shoulder_idx:right_shoulder_idx])
            breakout_volume = data['Volume'].iloc[-1]
            volume_confirmation = breakout_volume > avg_volume_shoulders * 1.3
            
            # Calculate pattern strength
            depth_strength = (left_shoulder - head) / head * 10  # Deeper head is stronger
            symmetry_strength = 1 - abs(left_shoulder - right_shoulder) / right_shoulder * 20
            volume_strength = 0.2 if volume_confirmation else 0
            
            final_strength = min(0.7 * (depth_strength + symmetry_strength) / 2 + volume_strength, 1.0)
            
            if final_strength < 0.6:  # Minimum strength threshold
                return None
                
            return {
                'name': 'Inverse Head & Shoulders',
                'type': 'reversal_bullish',
                'strength': final_strength,
                'neckline_level': current_neckline_level,
                'head_price': head,
                'volume_confirmed': volume_confirmation
            }

        except Exception as e:
            return None
