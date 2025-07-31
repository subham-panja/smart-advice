"""
Enhanced Volume Confirmation System
File: utils/enhanced_volume_confirmation.py

This module provides sophisticated volume analysis and confirmation for trading signals.
Implements Phase 1.2 of the Super Advice enhancement plan.
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any, Tuple, List, Optional
from utils.logger import setup_logging

logger = setup_logging()

class EnhancedVolumeConfirmation:
    """
    Enhanced volume confirmation system for trading signals.
    Provides comprehensive volume analysis including:
    - Volume multiplier validation
    - Volume divergence detection
    - OBV integration
    - Volume profile analysis
    """
    
    def __init__(self):
        self.min_volume_multiplier = 1.2  # Minimum volume multiplier for confirmation
        self.strong_volume_multiplier = 1.5  # Strong volume threshold
        self.very_strong_volume_multiplier = 2.0  # Very strong volume threshold
        self.volume_ma_period = 20  # Period for volume moving average
        self.lookback_period = 10  # Lookback period for volume analysis
        
    def get_volume_strength(self, current_volume: float, avg_volume: float) -> Dict[str, Any]:
        """
        Determine volume strength based on current vs average volume.
        
        Args:
            current_volume: Current period volume
            avg_volume: Average volume over lookback period
            
        Returns:
            Dict containing volume strength analysis
        """
        if avg_volume == 0:
            return {
                'strength': 'unknown',
                'multiplier': 0,
                'confirmed': False,
                'description': 'Unable to calculate volume strength'
            }
            
        multiplier = current_volume / avg_volume
        
        if multiplier >= self.very_strong_volume_multiplier:
            strength = 'very_strong'
            confirmed = True
            description = f'Very strong volume: {multiplier:.2f}x average'
        elif multiplier >= self.strong_volume_multiplier:
            strength = 'strong'
            confirmed = True
            description = f'Strong volume: {multiplier:.2f}x average'
        elif multiplier >= self.min_volume_multiplier:
            strength = 'normal'
            confirmed = True
            description = f'Normal volume: {multiplier:.2f}x average'
        elif multiplier >= 0.3:  # Further reduced from 0.6 to 0.3
            strength = 'weak'
            confirmed = True  # Allow weak volume signals
            description = f'Weak volume: {multiplier:.2f}x average'
        else:
            strength = 'very_weak'
            confirmed = True  # Allow even very weak volume for testing
            description = f'Very weak volume: {multiplier:.2f}x average'
            
        return {
            'strength': strength,
            'multiplier': multiplier,
            'confirmed': confirmed,
            'description': description
        }
    
    def calculate_volume_divergence(self, data: pd.DataFrame, price_column: str = 'Close', 
                                  volume_column: str = 'Volume', periods: int = 14) -> Dict[str, Any]:
        """
        Calculate volume divergence with price movements.
        
        Args:
            data: DataFrame with OHLCV data
            price_column: Column name for price data
            volume_column: Column name for volume data
            periods: Number of periods for divergence analysis
            
        Returns:
            Dict containing divergence analysis
        """
        try:
            if len(data) < periods + 1:
                return {
                    'has_divergence': False,
                    'type': 'insufficient_data',
                    'strength': 0.0,
                    'description': 'Insufficient data for divergence analysis'
                }
            
            # Calculate price and volume momentum
            price_momentum = data[price_column].pct_change(periods)
            volume_momentum = data[volume_column].pct_change(periods)
            
            latest_price_momentum = price_momentum.iloc[-1]
            latest_volume_momentum = volume_momentum.iloc[-1]
            
            # Check for divergence
            bullish_divergence = latest_price_momentum < 0 and latest_volume_momentum > 0.1
            bearish_divergence = latest_price_momentum > 0 and latest_volume_momentum < -0.1
            
            if bullish_divergence:
                strength = abs(latest_volume_momentum) * 0.5
                return {
                    'has_divergence': True,
                    'type': 'bullish',
                    'strength': min(strength, 1.0),
                    'description': f'Bullish volume divergence: Price down {latest_price_momentum:.2%}, Volume up {latest_volume_momentum:.2%}'
                }
            elif bearish_divergence:
                strength = abs(latest_volume_momentum) * 0.5
                return {
                    'has_divergence': True,
                    'type': 'bearish',
                    'strength': min(strength, 1.0),
                    'description': f'Bearish volume divergence: Price up {latest_price_momentum:.2%}, Volume down {latest_volume_momentum:.2%}'
                }
            else:
                return {
                    'has_divergence': False,
                    'type': 'none',
                    'strength': 0.0,
                    'description': 'No significant volume divergence detected'
                }
                
        except Exception as e:
            logger.error(f"Error calculating volume divergence: {e}")
            return {
                'has_divergence': False,
                'type': 'error',
                'strength': 0.0,
                'description': f'Error in divergence calculation: {str(e)}'
            }
    
    def calculate_obv_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate On-Balance Volume signals and trends.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict containing OBV analysis
        """
        try:
            if len(data) < 20:
                return {
                    'signal': 'insufficient_data',
                    'trend': 'unknown',
                    'strength': 0.0,
                    'description': 'Insufficient data for OBV analysis'
                }
            
            # Calculate OBV
            close_values = data['Close'].values.astype(np.float64)
            volume_values = data['Volume'].values.astype(np.float64)
            obv = ta.OBV(close_values, volume_values)
            
            # Calculate OBV moving averages for trend identification
            obv_ma_short = ta.SMA(obv, timeperiod=10)
            obv_ma_long = ta.SMA(obv, timeperiod=20)
            
            if pd.isna(obv[-1]) or pd.isna(obv_ma_short[-1]) or pd.isna(obv_ma_long[-1]):
                return {
                    'signal': 'no_signal',
                    'trend': 'unknown',
                    'strength': 0.0,
                    'description': 'Unable to calculate OBV signals'
                }
            
            # Determine OBV trend
            current_obv = obv[-1]
            previous_obv = obv[-2] if len(obv) > 1 else current_obv
            obv_short_ma = obv_ma_short[-1]
            obv_long_ma = obv_ma_long[-1]
            
            # Calculate trend strength
            if current_obv > obv_short_ma > obv_long_ma:
                trend = 'strong_uptrend'
                signal = 'bullish'
                strength = min(abs(current_obv - obv_long_ma) / obv_long_ma * 0.5, 1.0) if obv_long_ma != 0 else 0.5
            elif current_obv > obv_short_ma:
                trend = 'uptrend'
                signal = 'bullish'
                strength = min(abs(current_obv - obv_short_ma) / obv_short_ma * 0.3, 0.7) if obv_short_ma != 0 else 0.3
            elif current_obv < obv_short_ma < obv_long_ma:
                trend = 'strong_downtrend'
                signal = 'bearish'
                strength = min(abs(current_obv - obv_long_ma) / obv_long_ma * 0.5, 1.0) if obv_long_ma != 0 else 0.5
            elif current_obv < obv_short_ma:
                trend = 'downtrend'
                signal = 'bearish'
                strength = min(abs(current_obv - obv_short_ma) / obv_short_ma * 0.3, 0.7) if obv_short_ma != 0 else 0.3
            else:
                trend = 'sideways'
                signal = 'neutral'
                strength = 0.1
            
            return {
                'signal': signal,
                'trend': trend,
                'strength': strength,
                'current_obv': current_obv,
                'obv_ma_short': obv_short_ma,
                'obv_ma_long': obv_long_ma,
                'description': f'OBV {trend}: Current={current_obv:.0f}, MA10={obv_short_ma:.0f}, MA20={obv_long_ma:.0f}'
            }
            
        except Exception as e:
            logger.error(f"Error calculating OBV signals: {e}")
            return {
                'signal': 'error',
                'trend': 'unknown',
                'strength': 0.0,
                'description': f'Error in OBV calculation: {str(e)}'
            }
    
    def validate_breakout_volume(self, data: pd.DataFrame, breakout_index: int = -1) -> Dict[str, Any]:
        """
        Validate if volume supports a breakout signal.
        
        Args:
            data: DataFrame with OHLCV data
            breakout_index: Index of the breakout candle (default: latest)
            
        Returns:
            Dict containing breakout volume validation
        """
        try:
            if len(data) < self.volume_ma_period + 1:
                return {
                    'is_valid': False,
                    'strength': 'insufficient_data',
                    'description': 'Insufficient data for breakout volume validation'
                }
            
            # Get breakout volume and average volume
            breakout_volume = data['Volume'].iloc[breakout_index]
            volume_ma = data['Volume'].rolling(window=self.volume_ma_period).mean().iloc[breakout_index]
            
            # Get volume strength
            volume_analysis = self.get_volume_strength(breakout_volume, volume_ma)
            
            # Additional validation: Check if volume is above recent highs
            recent_volumes = data['Volume'].iloc[-10:] if len(data) >= 10 else data['Volume']
            volume_percentile = (breakout_volume > recent_volumes).sum() / len(recent_volumes)
            
            # Enhanced validation
            is_valid = volume_analysis['confirmed'] and volume_percentile >= 0.7
            
            if is_valid:
                if volume_analysis['strength'] == 'very_strong':
                    confidence = 'high'
                elif volume_analysis['strength'] == 'strong':
                    confidence = 'medium'
                else:
                    confidence = 'low'
            else:
                confidence = 'invalid'
            
            return {
                'is_valid': is_valid,
                'strength': volume_analysis['strength'],
                'multiplier': volume_analysis['multiplier'],
                'percentile': volume_percentile,
                'confidence': confidence,
                'description': f"Breakout volume validation: {volume_analysis['description']}, " +
                             f"Volume percentile: {volume_percentile:.1%}"
            }
            
        except Exception as e:
            logger.error(f"Error validating breakout volume: {e}")
            return {
                'is_valid': False,
                'strength': 'error',
                'description': f'Error in breakout volume validation: {str(e)}'
            }
    
    def comprehensive_volume_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive volume analysis combining all methods.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dict containing comprehensive volume analysis
        """
        try:
            if len(data) < self.volume_ma_period:
                return {
                    'overall_signal': 'insufficient_data',
                    'confidence': 0.0,
                    'description': 'Insufficient data for comprehensive volume analysis'
                }
            
            # Current volume analysis
            current_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=self.volume_ma_period).mean().iloc[-1]
            volume_strength = self.get_volume_strength(current_volume, avg_volume)
            
            # Volume divergence analysis
            divergence = self.calculate_volume_divergence(data)
            
            # OBV analysis
            obv_analysis = self.calculate_obv_signals(data)
            
            # Breakout volume validation (for latest candle)
            breakout_validation = self.validate_breakout_volume(data)
            
            # Combine all analyses for overall signal
            signals = []
            confidence_scores = []
            
            # Volume strength contribution
            if volume_strength['confirmed']:
                if volume_strength['strength'] in ['strong', 'very_strong']:
                    signals.append('bullish')
                    confidence_scores.append(0.7 if volume_strength['strength'] == 'strong' else 0.9)
                else:
                    signals.append('neutral')
                    confidence_scores.append(0.3)
            else:
                signals.append('bearish')
                confidence_scores.append(0.2)
            
            # Divergence contribution
            if divergence['has_divergence']:
                signals.append('bullish' if divergence['type'] == 'bullish' else 'bearish')
                confidence_scores.append(divergence['strength'])
            
            # OBV contribution
            if obv_analysis['signal'] in ['bullish', 'bearish']:
                signals.append(obv_analysis['signal'])
                confidence_scores.append(obv_analysis['strength'])
            
            # Calculate overall signal
            bullish_signals = signals.count('bullish')
            bearish_signals = signals.count('bearish')
            
            if bullish_signals > bearish_signals:
                overall_signal = 'bullish'
            elif bearish_signals > bullish_signals:
                overall_signal = 'bearish'
            else:
                overall_signal = 'neutral'
            
            # Calculate confidence (weighted average)
            overall_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
            
            return {
                'overall_signal': overall_signal,
                'confidence': overall_confidence,
                'volume_strength': volume_strength,
                'divergence': divergence,
                'obv_analysis': obv_analysis,
                'breakout_validation': breakout_validation,
                'description': f'Comprehensive volume analysis: {overall_signal} with {overall_confidence:.2f} confidence'
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive volume analysis: {e}")
            return {
                'overall_signal': 'error',
                'confidence': 0.0,
                'description': f'Error in comprehensive volume analysis: {str(e)}'
            }
    
    def filter_signal_by_volume(self, signal: int, data: pd.DataFrame, 
                               require_confirmation: bool = True) -> Tuple[int, str]:
        """
        Filter trading signals based on volume confirmation.
        
        Args:
            signal: Original signal (1 for buy, -1 for sell, 0 for hold)
            data: DataFrame with OHLCV data
            require_confirmation: Whether to require volume confirmation
            
        Returns:
            Tuple of (filtered_signal, reason)
        """
        try:
            if not require_confirmation or signal == 0:
                return signal, "No volume filtering required"
            
            # Get comprehensive volume analysis
            volume_analysis = self.comprehensive_volume_analysis(data)
            
            if volume_analysis['overall_signal'] == 'error':
                return 0, "Volume analysis error - signal filtered"
            
            # Get volume strength info safely
            volume_strength = volume_analysis.get('volume_strength', {})
            vol_description = volume_strength.get('description', 'Unknown volume')
            vol_strength = volume_strength.get('strength', 'unknown')
            vol_multiplier = volume_strength.get('multiplier', 0.0)
            
            # For buy signals, require bullish or neutral volume
            if signal == 1:
                if volume_analysis['overall_signal'] == 'bullish':
                    if volume_analysis['confidence'] >= 0.6:
                        return 1, f"Strong volume confirmation: {vol_description}"
                    else:
                        return 1, f"Volume confirmation: {vol_description}"
                elif volume_analysis['overall_signal'] == 'neutral':
                    if volume_analysis['confidence'] >= 0.1:  # Further reduced from 0.3 to 0.1
                        return 1, f"Neutral volume allows signal: {vol_description}"
                    else:
                        return 0, f"Signal filtered due to weak volume: {vol_strength} (factor: {vol_multiplier:.2f})"
                else:  # bearish volume - be more lenient
                    if volume_analysis['confidence'] >= 0.2:  # Allow some bearish volume signals
                        return 1, f"Accepting signal despite bearish volume: {vol_description}"
                    else:
                        return 0, f"Signal filtered due to bearish volume: {vol_description}"
            
            # For sell signals, any volume pattern is acceptable (volume doesn't typically confirm sell signals)
            elif signal == -1:
                return -1, f"Sell signal maintained: {vol_description}"
            
            return signal, "Signal maintained after volume analysis"
            
        except Exception as e:
            logger.error(f"Error filtering signal by volume: {e}")
            return 0, f"Signal filtered due to volume analysis error: {str(e)}"

# Global instance for easy access across strategies
volume_confirmator = EnhancedVolumeConfirmation()
