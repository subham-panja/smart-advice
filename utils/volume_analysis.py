"""
Enhanced Volume Analysis Utility
File: utils/volume_analysis.py

This module provides sophisticated volume analysis capabilities for improved
signal confirmation across all trading strategies. It implements multiple
volume confirmation techniques including:
- Volume breakout detection
- Volume divergence analysis
- Volume trend analysis
- Volume-weighted price levels
- Accumulation/Distribution patterns
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, List, Tuple, Optional
from utils.logger import setup_logging

logger = setup_logging()


class VolumeAnalyzer:
    """
    Enhanced Volume Analysis for trading signal confirmation.
    
    This class provides comprehensive volume analysis capabilities to improve
    the quality and reliability of trading signals across all strategies.
    """
    
    def __init__(self, params: Dict = None):
        """Initialize the Volume Analyzer with configurable parameters."""
        self.params = params or {}
        
        # Volume confirmation thresholds
        self.volume_breakout_multiplier = self.params.get('volume_breakout_multiplier', 1.5)
        self.volume_strong_multiplier = self.params.get('volume_strong_multiplier', 2.0)
        self.volume_weak_threshold = self.params.get('volume_weak_threshold', 0.7)
        self.volume_lookback = self.params.get('volume_lookback', 20)
        
        # Volume trend analysis
        self.trend_lookback = self.params.get('trend_lookback', 10)
        self.divergence_lookback = self.params.get('divergence_lookback', 15)
        
    def get_volume_confirmation_factor(self, data: pd.DataFrame, signal_type: str = 'bullish') -> Dict:
        """
        Calculate comprehensive volume confirmation factor.
        
        Args:
            data: DataFrame with OHLCV data
            signal_type: 'bullish' or 'bearish' signal type
            
        Returns:
            Dictionary with volume confirmation details
        """
        try:
            if len(data) < self.volume_lookback:
                return {'factor': 1.0, 'strength': 'insufficient_data', 'details': []}
            
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].tail(self.volume_lookback).mean()
            
            if avg_volume == 0:
                return {'factor': 1.0, 'strength': 'no_volume_data', 'details': []}
            
            volume_ratio = current_volume / avg_volume
            details = []
            base_factor = 1.0
            
            # 1. Basic volume confirmation
            if volume_ratio >= self.volume_strong_multiplier:
                base_factor = 1.4
                details.append(f"Strong volume: {volume_ratio:.1f}x average")
            elif volume_ratio >= self.volume_breakout_multiplier:
                base_factor = 1.2
                details.append(f"Good volume: {volume_ratio:.1f}x average")
            elif volume_ratio >= 1.0:
                base_factor = 1.0
                details.append(f"Normal volume: {volume_ratio:.1f}x average")
            elif volume_ratio >= self.volume_weak_threshold:
                base_factor = 0.9
                details.append(f"Weak volume: {volume_ratio:.1f}x average")
            else:
                base_factor = 0.7
                details.append(f"Very weak volume: {volume_ratio:.1f}x average")
            
            # 2. Volume trend analysis
            volume_trend_factor = self._analyze_volume_trend(data)
            if volume_trend_factor['factor'] != 1.0:
                base_factor *= volume_trend_factor['factor']
                details.extend(volume_trend_factor['details'])
            
            # 3. Volume-price divergence
            divergence_factor = self._analyze_volume_price_divergence(data, signal_type)
            if divergence_factor['factor'] != 1.0:
                base_factor *= divergence_factor['factor']
                details.extend(divergence_factor['details'])
            
            # 4. Volume accumulation pattern
            accumulation_factor = self._analyze_volume_accumulation(data)
            if accumulation_factor['factor'] != 1.0:
                base_factor *= accumulation_factor['factor']
                details.extend(accumulation_factor['details'])
            
            # Determine overall strength
            if base_factor >= 1.3:
                strength = 'very_strong'
            elif base_factor >= 1.1:
                strength = 'strong'
            elif base_factor >= 0.95:
                strength = 'normal'
            elif base_factor >= 0.8:
                strength = 'weak'
            else:
                strength = 'very_weak'
            
            return {
                'factor': round(base_factor, 2),
                'strength': strength,
                'volume_ratio': round(volume_ratio, 2),
                'details': details
            }
            
        except Exception as e:
            logger.error(f"Error in volume confirmation analysis: {e}")
            return {'factor': 1.0, 'strength': 'error', 'details': [str(e)]}
    
    def _analyze_volume_trend(self, data: pd.DataFrame) -> Dict:
        """Analyze volume trend over recent periods."""
        try:
            if len(data) < self.trend_lookback * 2:
                return {'factor': 1.0, 'details': []}
            
            # Compare recent volume trend to historical
            recent_volume = data['Volume'].tail(self.trend_lookback).mean()
            historical_volume = data['Volume'].tail(self.trend_lookback * 2).head(self.trend_lookback).mean()
            
            if historical_volume == 0:
                return {'factor': 1.0, 'details': []}
            
            trend_ratio = recent_volume / historical_volume
            
            if trend_ratio >= 1.2:
                return {
                    'factor': 1.1,
                    'details': [f"Increasing volume trend: {trend_ratio:.1f}x recent vs historical"]
                }
            elif trend_ratio <= 0.8:
                return {
                    'factor': 0.9,
                    'details': [f"Declining volume trend: {trend_ratio:.1f}x recent vs historical"]
                }
            
            return {'factor': 1.0, 'details': []}
            
        except Exception:
            return {'factor': 1.0, 'details': []}
    
    def _analyze_volume_price_divergence(self, data: pd.DataFrame, signal_type: str) -> Dict:
        """Analyze volume-price divergence patterns."""
        try:
            if len(data) < self.divergence_lookback:
                return {'factor': 1.0, 'details': []}
            
            recent_data = data.tail(self.divergence_lookback)
            
            # Calculate price and volume trends
            price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
            volume_change = (recent_data['Volume'].iloc[-1] - recent_data['Volume'].iloc[0]) / recent_data['Volume'].iloc[0]
            
            # Look for bullish divergence (price declining, volume increasing)
            if signal_type == 'bullish':
                if price_change < -0.02 and volume_change > 0.1:  # Price down 2%+, volume up 10%+
                    return {
                        'factor': 1.2,
                        'details': [f"Bullish volume divergence: price {price_change:.1%}, volume {volume_change:.1%}"]
                    }
                elif price_change > 0.02 and volume_change < -0.1:  # Price up but volume declining
                    return {
                        'factor': 0.9,
                        'details': [f"Weak volume confirmation: price {price_change:.1%}, volume {volume_change:.1%}"]
                    }
            
            # Look for bearish divergence (price rising, volume declining)
            elif signal_type == 'bearish':
                if price_change > 0.02 and volume_change < -0.1:  # Price up 2%+, volume down 10%+
                    return {
                        'factor': 1.2,
                        'details': [f"Bearish volume divergence: price {price_change:.1%}, volume {volume_change:.1%}"]
                    }
            
            return {'factor': 1.0, 'details': []}
            
        except Exception:
            return {'factor': 1.0, 'details': []}
    
    def _analyze_volume_accumulation(self, data: pd.DataFrame) -> Dict:
        """Analyze volume accumulation patterns."""
        try:
            if len(data) < 10:
                return {'factor': 1.0, 'details': []}
            
            # Calculate volume accumulation using price-volume relationship
            recent_data = data.tail(5)
            
            # Up volume vs Down volume analysis
            up_volume = 0
            down_volume = 0
            
            for i in range(1, len(recent_data)):
                current = recent_data.iloc[i]
                previous = recent_data.iloc[i-1]
                
                if current['Close'] > previous['Close']:
                    up_volume += current['Volume']
                elif current['Close'] < previous['Close']:
                    down_volume += current['Volume']
            
            total_directional_volume = up_volume + down_volume
            
            if total_directional_volume > 0:
                up_volume_ratio = up_volume / total_directional_volume
                
                if up_volume_ratio >= 0.7:
                    return {
                        'factor': 1.15,
                        'details': [f"Strong buying pressure: {up_volume_ratio:.1%} up-volume"]
                    }
                elif up_volume_ratio <= 0.3:
                    return {
                        'factor': 0.85,
                        'details': [f"Selling pressure: {up_volume_ratio:.1%} up-volume"]
                    }
            
            return {'factor': 1.0, 'details': []}
            
        except Exception:
            return {'factor': 1.0, 'details': []}
    
    def detect_volume_breakout(self, data: pd.DataFrame, price_breakout: bool = False) -> Dict:
        """
        Detect volume breakouts that confirm price movements.
        
        Args:
            data: DataFrame with OHLCV data
            price_breakout: Whether there's an accompanying price breakout
            
        Returns:
            Dictionary with volume breakout analysis
        """
        try:
            if len(data) < self.volume_lookback:
                return {'detected': False, 'strength': 0.0, 'details': []}
            
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].tail(self.volume_lookback).mean()
            volume_std = data['Volume'].tail(self.volume_lookback).std()
            
            if avg_volume == 0 or volume_std == 0:
                return {'detected': False, 'strength': 0.0, 'details': []}
            
            # Calculate volume z-score
            volume_z_score = (current_volume - avg_volume) / volume_std
            volume_ratio = current_volume / avg_volume
            
            details = []
            
            # Determine breakout strength
            if volume_ratio >= self.volume_strong_multiplier and volume_z_score >= 2.0:
                strength = 1.0
                details.append(f"Very strong volume breakout: {volume_ratio:.1f}x avg, z-score: {volume_z_score:.1f}")
            elif volume_ratio >= self.volume_breakout_multiplier and volume_z_score >= 1.5:
                strength = 0.8
                details.append(f"Strong volume breakout: {volume_ratio:.1f}x avg, z-score: {volume_z_score:.1f}")
            elif volume_ratio >= 1.2 and volume_z_score >= 1.0:
                strength = 0.6
                details.append(f"Moderate volume breakout: {volume_ratio:.1f}x avg, z-score: {volume_z_score:.1f}")
            else:
                return {'detected': False, 'strength': 0.0, 'details': ['No significant volume breakout']}
            
            # Enhanced strength if accompanied by price breakout
            if price_breakout:
                strength *= 1.2
                details.append("Volume breakout confirms price breakout")
            
            return {
                'detected': True,
                'strength': min(1.0, strength),
                'volume_ratio': volume_ratio,
                'z_score': volume_z_score,
                'details': details
            }
            
        except Exception as e:
            return {'detected': False, 'strength': 0.0, 'details': [f"Error: {e}"]}
    
    def analyze_volume_at_support_resistance(self, data: pd.DataFrame, level: float, tolerance: float = 0.02) -> Dict:
        """
        Analyze volume behavior at key support/resistance levels.
        
        Args:
            data: DataFrame with OHLCV data
            level: Support/resistance price level
            tolerance: Price tolerance for level detection (percentage)
            
        Returns:
            Dictionary with volume analysis at the level
        """
        try:
            if len(data) < 10:
                return {'volume_confirmation': False, 'strength': 0.0, 'details': []}
            
            # Find instances where price tested the level
            level_tests = []
            avg_volume = data['Volume'].mean()
            
            for i in range(1, len(data)):
                low = data['Low'].iloc[i]
                high = data['High'].iloc[i]
                volume = data['Volume'].iloc[i]
                
                # Check if this bar tested the level
                if (low <= level * (1 + tolerance) and high >= level * (1 - tolerance)):
                    level_tests.append({
                        'index': i,
                        'volume': volume,
                        'volume_ratio': volume / avg_volume if avg_volume > 0 else 0,
                        'price_action': 'bounce' if data['Close'].iloc[i] > level else 'break'
                    })
            
            if not level_tests:
                return {'volume_confirmation': False, 'strength': 0.0, 'details': ['Level not tested recently']}
            
            # Analyze volume at recent tests
            recent_tests = level_tests[-3:] if len(level_tests) >= 3 else level_tests
            avg_test_volume_ratio = np.mean([test['volume_ratio'] for test in recent_tests])
            
            details = []
            
            # High volume at level indicates strong support/resistance
            if avg_test_volume_ratio >= 1.5:
                strength = 0.9
                details.append(f"Strong volume at level: {avg_test_volume_ratio:.1f}x average")
            elif avg_test_volume_ratio >= 1.2:
                strength = 0.7
                details.append(f"Good volume at level: {avg_test_volume_ratio:.1f}x average")
            elif avg_test_volume_ratio >= 0.8:
                strength = 0.5
                details.append(f"Normal volume at level: {avg_test_volume_ratio:.1f}x average")
            else:
                strength = 0.3
                details.append(f"Low volume at level: {avg_test_volume_ratio:.1f}x average")
            
            # Check for volume expansion on recent test
            latest_test = recent_tests[-1]
            if latest_test['volume_ratio'] >= 1.3:
                strength *= 1.1
                details.append("Volume expansion on latest test")
            
            return {
                'volume_confirmation': strength >= 0.5,
                'strength': strength,
                'avg_volume_ratio': avg_test_volume_ratio,
                'test_count': len(level_tests),
                'latest_test_volume': latest_test['volume_ratio'],
                'details': details
            }
            
        except Exception as e:
            return {'volume_confirmation': False, 'strength': 0.0, 'details': [f"Error: {e}"]}
    
    def get_volume_weighted_price(self, data: pd.DataFrame, periods: int = 20) -> Dict:
        """
        Calculate Volume Weighted Average Price (VWAP) and related metrics.
        
        Args:
            data: DataFrame with OHLCV data
            periods: Number of periods for calculation
            
        Returns:
            Dictionary with VWAP analysis
        """
        try:
            if len(data) < periods:
                return {'vwap': None, 'analysis': 'insufficient_data'}
            
            recent_data = data.tail(periods)
            
            # Calculate VWAP
            typical_price = (recent_data['High'] + recent_data['Low'] + recent_data['Close']) / 3
            volume_price = typical_price * recent_data['Volume']
            total_volume = recent_data['Volume'].sum()
            
            if total_volume == 0:
                return {'vwap': None, 'analysis': 'no_volume'}
            
            vwap = volume_price.sum() / total_volume
            current_price = data['Close'].iloc[-1]
            
            # Calculate price deviation from VWAP
            price_deviation = (current_price - vwap) / vwap
            
            # Analyze position relative to VWAP
            if price_deviation > 0.02:
                analysis = 'above_vwap_bullish'
                details = f"Price {price_deviation:.1%} above VWAP"
            elif price_deviation > 0.005:
                analysis = 'slightly_above_vwap'
                details = f"Price {price_deviation:.1%} above VWAP"
            elif price_deviation < -0.02:
                analysis = 'below_vwap_bearish'
                details = f"Price {price_deviation:.1%} below VWAP"
            elif price_deviation < -0.005:
                analysis = 'slightly_below_vwap'
                details = f"Price {price_deviation:.1%} below VWAP"
            else:
                analysis = 'near_vwap'
                details = f"Price near VWAP ({price_deviation:.1%} deviation)"
            
            return {
                'vwap': round(vwap, 2),
                'current_price': round(current_price, 2),
                'deviation': round(price_deviation, 4),
                'analysis': analysis,
                'details': details
            }
            
        except Exception as e:
            return {'vwap': None, 'analysis': 'error', 'details': str(e)}


def get_enhanced_volume_confirmation(data: pd.DataFrame, signal_type: str = 'bullish', 
                                   breakout: bool = False, level: float = None) -> Dict:
    """
    Convenience function to get enhanced volume confirmation for any strategy.
    
    Args:
        data: DataFrame with OHLCV data
        signal_type: 'bullish' or 'bearish'
        breakout: Whether this is a breakout signal
        level: Support/resistance level if applicable
        
    Returns:
        Dictionary with comprehensive volume analysis
    """
    analyzer = VolumeAnalyzer()
    
    # Get base volume confirmation
    confirmation = analyzer.get_volume_confirmation_factor(data, signal_type)
    
    # Add breakout analysis if applicable
    if breakout:
        breakout_analysis = analyzer.detect_volume_breakout(data, price_breakout=True)
        if breakout_analysis['detected']:
            confirmation['factor'] *= (1 + breakout_analysis['strength'] * 0.2)
            confirmation['details'].extend(breakout_analysis['details'])
    
    # Add support/resistance analysis if level provided
    if level is not None:
        level_analysis = analyzer.analyze_volume_at_support_resistance(data, level)
        if level_analysis['volume_confirmation']:
            confirmation['factor'] *= (1 + level_analysis['strength'] * 0.1)
            confirmation['details'].extend(level_analysis['details'])
    
    # Add VWAP context
    vwap_analysis = analyzer.get_volume_weighted_price(data)
    if vwap_analysis['vwap'] is not None:
        confirmation['vwap_context'] = vwap_analysis['details']
    
    return confirmation
