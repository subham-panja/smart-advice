"""
Volume Profile Analysis Strategy
File: scripts/strategies/volume_profile.py

This strategy analyzes volume profile to identify key support/resistance levels:
- Volume at Price (VPVR) analysis
- Point of Control (POC) identification
- Value Area (VA) calculations
- High/Low Volume Nodes (HVN/LVN)
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Tuple, Optional
from .base_strategy import BaseStrategy

class VolumeProfile(BaseStrategy):
    """
    Volume Profile Analysis for identifying key price levels based on trading activity.
    
    This strategy identifies significant support and resistance levels using volume distribution
    at different price levels over a specified period.
    """
    
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback_period = self.get_parameter('lookback_period', 50)
        self.price_bins = self.get_parameter('price_bins', 50)  # Number of price levels to analyze
        self.value_area_percentage = self.get_parameter('value_area_percentage', 0.68)  # 68% of volume
        self.min_volume_threshold = self.get_parameter('min_volume_threshold', 0.1)
        self.proximity_threshold = self.get_parameter('proximity_threshold', 0.01)  # 1% price proximity
        
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the volume profile analysis strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for bullish volume profile signal, -1 for bearish/no signal
        """
        # Validate data
        if not self.validate_data(data, min_periods=self.lookback_period):
            return -1
            
        try:
            # Analyze recent data for volume profile
            recent_data = data.tail(self.lookback_period)
            current_price = recent_data['Close'].iloc[-1]
            
            # Calculate volume profile
            volume_profile = self._calculate_volume_profile(recent_data)
            if not volume_profile:
                self.log_signal(-1, "Unable to calculate volume profile", data)
                return -1
            
            # Identify key levels
            poc_price = volume_profile['poc_price']
            value_area_high = volume_profile['value_area_high']
            value_area_low = volume_profile['value_area_low']
            hvn_levels = volume_profile['hvn_levels']  # High Volume Nodes
            lvn_levels = volume_profile['lvn_levels']  # Low Volume Nodes
            
            signal_strength = 0
            signal_reasons = []
            
            # 1. Check proximity to Point of Control (POC)
            poc_signal = self._analyze_poc_proximity(current_price, poc_price)
            if poc_signal:
                signal_strength += poc_signal['strength']
                signal_reasons.append(poc_signal['reason'])
            
            # 2. Check Value Area analysis
            va_signal = self._analyze_value_area(current_price, value_area_high, value_area_low)
            if va_signal:
                signal_strength += va_signal['strength']
                signal_reasons.append(va_signal['reason'])
            
            # 3. Check High Volume Node support
            hvn_signal = self._analyze_hvn_support(current_price, hvn_levels)
            if hvn_signal:
                signal_strength += hvn_signal['strength']
                signal_reasons.append(hvn_signal['reason'])
            
            # 4. Check Low Volume Node resistance/breakout
            lvn_signal = self._analyze_lvn_breakout(current_price, lvn_levels, recent_data)
            if lvn_signal:
                signal_strength += lvn_signal['strength']
                signal_reasons.append(lvn_signal['reason'])
            
            # 5. Volume trend analysis
            volume_trend_signal = self._analyze_volume_trend(recent_data, volume_profile)
            if volume_trend_signal:
                signal_strength += volume_trend_signal['strength']
                signal_reasons.append(volume_trend_signal['reason'])
            
            # Generate final signal
            if signal_strength >= 0.6:  # Strong bullish volume profile
                reason = f"Strong volume profile signals: {'; '.join(signal_reasons)} (Strength: {signal_strength:.2f})"
                self.log_signal(1, reason, data)
                return 1
            elif signal_strength >= 0.3:  # Moderate signal
                reason = f"Moderate volume profile signals: {'; '.join(signal_reasons)} (Strength: {signal_strength:.2f})"
                self.log_signal(1, reason, data)
                return 1
            else:
                if signal_reasons:
                    reason = f"Weak volume profile signals: {'; '.join(signal_reasons)} (Strength: {signal_strength:.2f})"
                else:
                    reason = "No significant volume profile signals detected"
                self.log_signal(-1, reason, data)
                return -1
                
        except Exception as e:
            self.log_signal(-1, f"Error in volume profile analysis: {str(e)}", data)
            return -1
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Calculate volume profile for the given data period.
        
        Returns dictionary with POC, Value Area, and volume nodes.
        """
        try:
            if len(data) < 10:
                return None
            
            # Calculate price range
            price_min = data['Low'].min()
            price_max = data['High'].max()
            price_range = price_max - price_min
            
            if price_range == 0:
                return None
            
            # Create price bins
            price_step = price_range / self.price_bins
            price_levels = np.arange(price_min, price_max + price_step, price_step)
            
            # Initialize volume at each price level
            volume_at_price = np.zeros(len(price_levels) - 1)
            
            # Distribute volume across price levels for each bar
            for idx, row in data.iterrows():
                bar_low = row['Low']
                bar_high = row['High']
                bar_volume = row['Volume']
                
                if bar_volume == 0 or bar_high == bar_low:
                    continue
                
                # Find which price bins this bar covers
                low_bin = max(0, int((bar_low - price_min) / price_step))
                high_bin = min(len(volume_at_price) - 1, int((bar_high - price_min) / price_step))
                
                # Distribute volume proportionally across the price range of the bar
                bins_covered = max(1, high_bin - low_bin + 1)
                volume_per_bin = bar_volume / bins_covered
                
                for bin_idx in range(low_bin, high_bin + 1):
                    if bin_idx < len(volume_at_price):
                        volume_at_price[bin_idx] += volume_per_bin
            
            # Find Point of Control (highest volume)
            poc_idx = np.argmax(volume_at_price)
            poc_price = price_min + (poc_idx + 0.5) * price_step
            
            # Calculate Value Area (68% of total volume)
            total_volume = np.sum(volume_at_price)
            if total_volume == 0:
                return None
            
            value_area_volume = total_volume * self.value_area_percentage
            
            # Find Value Area by expanding from POC
            va_volume = volume_at_price[poc_idx]
            va_low_idx = poc_idx
            va_high_idx = poc_idx
            
            while va_volume < value_area_volume and (va_low_idx > 0 or va_high_idx < len(volume_at_price) - 1):
                # Decide whether to expand up or down
                volume_below = volume_at_price[va_low_idx - 1] if va_low_idx > 0 else 0
                volume_above = volume_at_price[va_high_idx + 1] if va_high_idx < len(volume_at_price) - 1 else 0
                
                if volume_below > volume_above and va_low_idx > 0:
                    va_low_idx -= 1
                    va_volume += volume_at_price[va_low_idx]
                elif va_high_idx < len(volume_at_price) - 1:
                    va_high_idx += 1
                    va_volume += volume_at_price[va_high_idx]
                else:
                    break
            
            value_area_low = price_min + va_low_idx * price_step
            value_area_high = price_min + (va_high_idx + 1) * price_step
            
            # Find High Volume Nodes (HVN) - peaks in volume
            hvn_levels = self._find_volume_nodes(volume_at_price, price_levels, 'high')
            
            # Find Low Volume Nodes (LVN) - valleys in volume
            lvn_levels = self._find_volume_nodes(volume_at_price, price_levels, 'low')
            
            return {
                'poc_price': poc_price,
                'poc_volume': volume_at_price[poc_idx],
                'value_area_high': value_area_high,
                'value_area_low': value_area_low,
                'hvn_levels': hvn_levels,
                'lvn_levels': lvn_levels,
                'total_volume': total_volume,
                'volume_at_price': volume_at_price,
                'price_levels': price_levels
            }
            
        except Exception as e:
            return None
    
    def _find_volume_nodes(self, volume_at_price: np.ndarray, price_levels: np.ndarray, node_type: str) -> List[float]:
        """
        Find High Volume Nodes (peaks) or Low Volume Nodes (valleys) in the volume profile.
        """
        try:
            from scipy.signal import find_peaks
            
            if node_type == 'high':
                # Find peaks (HVN)
                peaks, _ = find_peaks(volume_at_price, prominence=np.std(volume_at_price) * 0.3)
                # Convert indices to prices
                hvn_prices = []
                for peak_idx in peaks:
                    if peak_idx < len(price_levels) - 1:
                        price = price_levels[peak_idx] + (price_levels[1] - price_levels[0]) * 0.5
                        volume = volume_at_price[peak_idx]
                        # Only include significant HVNs
                        if volume > np.mean(volume_at_price) * 1.2:
                            hvn_prices.append(price)
                return hvn_prices
            
            else:  # node_type == 'low'
                # Find valleys (LVN) by inverting the data
                inverted_volume = -volume_at_price
                valleys, _ = find_peaks(inverted_volume, prominence=np.std(inverted_volume) * 0.3)
                # Convert indices to prices
                lvn_prices = []
                for valley_idx in valleys:
                    if valley_idx < len(price_levels) - 1:
                        price = price_levels[valley_idx] + (price_levels[1] - price_levels[0]) * 0.5
                        volume = volume_at_price[valley_idx]
                        # Only include significant LVNs (low volume areas)
                        if volume < np.mean(volume_at_price) * 0.5:
                            lvn_prices.append(price)
                return lvn_prices
                
        except Exception:
            return []
    
    def _analyze_poc_proximity(self, current_price: float, poc_price: float) -> Optional[Dict]:
        """
        Analyze proximity to Point of Control for potential support/resistance.
        """
        try:
            distance_ratio = abs(current_price - poc_price) / current_price
            
            if distance_ratio <= self.proximity_threshold:
                # Very close to POC - strong support/resistance
                strength = 0.8 * (1 - distance_ratio / self.proximity_threshold)
                
                if current_price >= poc_price:
                    reason = f"Price near POC support at {poc_price:.2f} (current: {current_price:.2f})"
                else:
                    reason = f"Price testing POC resistance at {poc_price:.2f} (current: {current_price:.2f})"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            elif distance_ratio <= self.proximity_threshold * 2:
                # Moderately close to POC
                strength = 0.4 * (1 - distance_ratio / (self.proximity_threshold * 2))
                reason = f"Price approaching POC level at {poc_price:.2f} (current: {current_price:.2f})"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            return None
            
        except Exception:
            return None
    
    def _analyze_value_area(self, current_price: float, va_high: float, va_low: float) -> Optional[Dict]:
        """
        Analyze current price position relative to Value Area.
        """
        try:
            if va_low <= current_price <= va_high:
                # Price within Value Area - neutral to slightly bullish
                va_range = va_high - va_low
                position_ratio = (current_price - va_low) / va_range if va_range > 0 else 0.5
                
                if position_ratio > 0.6:
                    strength = 0.3
                    reason = f"Price in upper Value Area ({va_low:.2f} - {va_high:.2f})"
                else:
                    strength = 0.2
                    reason = f"Price in Value Area ({va_low:.2f} - {va_high:.2f})"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            elif current_price < va_low:
                # Price below Value Area - potential oversold
                distance_ratio = abs(current_price - va_low) / current_price
                
                if distance_ratio <= self.proximity_threshold:
                    strength = 0.6  # Strong support at VA low
                    reason = f"Price near Value Area low support at {va_low:.2f}"
                else:
                    strength = 0.4  # Oversold condition
                    reason = f"Price below Value Area ({va_low:.2f}), potentially oversold"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            else:  # current_price > va_high
                # Price above Value Area - check for breakout
                distance_ratio = abs(current_price - va_high) / current_price
                
                if distance_ratio <= self.proximity_threshold:
                    strength = 0.3  # Testing resistance
                    reason = f"Price testing Value Area high resistance at {va_high:.2f}"
                else:
                    strength = 0.5  # Potential breakout
                    reason = f"Price above Value Area ({va_high:.2f}), potential strength"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
        except Exception:
            return None
    
    def _analyze_hvn_support(self, current_price: float, hvn_levels: List[float]) -> Optional[Dict]:
        """
        Analyze proximity to High Volume Nodes for support levels.
        """
        try:
            if not hvn_levels:
                return None
            
            # Find closest HVN below current price (potential support)
            support_hvns = [level for level in hvn_levels if level <= current_price]
            
            if not support_hvns:
                return None
            
            closest_support = max(support_hvns)  # Closest support level
            distance_ratio = abs(current_price - closest_support) / current_price
            
            if distance_ratio <= self.proximity_threshold:
                # Very close to HVN support
                strength = 0.7 * (1 - distance_ratio / self.proximity_threshold)
                reason = f"Price near HVN support at {closest_support:.2f}"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            elif distance_ratio <= self.proximity_threshold * 3:
                # Moderately close to HVN support
                strength = 0.3 * (1 - distance_ratio / (self.proximity_threshold * 3))
                reason = f"Price above HVN support at {closest_support:.2f}"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            return None
            
        except Exception:
            return None
    
    def _analyze_lvn_breakout(self, current_price: float, lvn_levels: List[float], data: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze potential breakouts through Low Volume Nodes (areas of low resistance).
        """
        try:
            if not lvn_levels or len(data) < 5:
                return None
            
            # Find LVNs close to current price
            nearby_lvns = [level for level in lvn_levels 
                          if abs(level - current_price) / current_price <= self.proximity_threshold * 2]
            
            if not nearby_lvns:
                return None
            
            # Check if price is breaking through or has recently broken through an LVN
            recent_prices = data['Close'].tail(5).values
            
            for lvn_price in nearby_lvns:
                # Check if price has crossed the LVN recently
                below_count = sum(1 for p in recent_prices if p < lvn_price)
                above_count = sum(1 for p in recent_prices if p > lvn_price)
                
                if above_count >= 3 and below_count <= 2:  # Recent breakout above LVN
                    distance_ratio = abs(current_price - lvn_price) / current_price
                    strength = 0.5 * (1 - distance_ratio / (self.proximity_threshold * 2))
                    reason = f"Breakout above LVN resistance at {lvn_price:.2f}"
                    
                    return {
                        'strength': strength,
                        'reason': reason
                    }
            
            return None
            
        except Exception:
            return None
    
    def _analyze_volume_trend(self, data: pd.DataFrame, volume_profile: Dict) -> Optional[Dict]:
        """
        Analyze volume trend and its relationship with price movement.
        """
        try:
            if len(data) < 10:
                return None
            
            recent_volume = data['Volume'].tail(5).mean()
            historical_volume = data['Volume'].tail(20).mean()
            
            if historical_volume == 0:
                return None
            
            volume_ratio = recent_volume / historical_volume
            
            # Check price trend
            recent_close = data['Close'].iloc[-1]
            prev_close = data['Close'].iloc[-5]
            price_change = (recent_close - prev_close) / prev_close
            
            # Volume confirmation analysis
            if volume_ratio >= 1.3 and price_change > 0.02:  # High volume + price up
                strength = min(0.6, volume_ratio * 0.3)
                reason = f"Strong volume confirmation (ratio: {volume_ratio:.1f}x) with price rise"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            elif volume_ratio >= 1.1 and price_change > 0.01:  # Moderate volume + modest price up
                strength = min(0.4, volume_ratio * 0.2)
                reason = f"Moderate volume support (ratio: {volume_ratio:.1f}x) with price rise"
                
                return {
                    'strength': strength,
                    'reason': reason
                }
            
            return None
            
        except Exception:
            return None
