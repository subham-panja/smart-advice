"""
Support/Resistance Breakout Strategy
File: scripts/strategies/support_resistance_breakout.py

This strategy identifies significant support and resistance levels and trades breakouts
from these levels. It uses multiple timeframe analysis and pivot point detection
to identify high-probability breakout opportunities.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from utils.logger import setup_logging
from scipy.signal import argrelextrema

logger = setup_logging()


class SupportResistanceBreakoutStrategy(BaseStrategy):
    """
    Support/Resistance Breakout Strategy for swing trading.
    
    Logic:
    1. Identify significant support and resistance levels using pivot points
    2. Look for price consolidation near these levels
    3. Trade breakouts with volume confirmation
    4. Use multiple touches to validate level significance
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Support_Resistance_Breakout"
        self.description = "Breakout from significant support/resistance levels"
        
    def find_support_resistance_levels(self, data: pd.DataFrame, window: int = 10, min_touches: int = 2) -> dict:
        """
        Find significant support and resistance levels using pivot points.
        
        Args:
            data: DataFrame with OHLCV data
            window: Window for pivot point detection
            min_touches: Minimum number of touches to validate level
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            if len(data) < window * 3:
                return {'support_levels': [], 'resistance_levels': []}
            
            # Find local minima and maxima (pivot points)
            local_minima_idx = argrelextrema(data['Low'].values, np.less, order=window)[0]
            local_maxima_idx = argrelextrema(data['High'].values, np.greater, order=window)[0]
            
            # Extract pivot lows and highs
            pivot_lows = []
            pivot_highs = []
            
            for idx in local_minima_idx:
                if idx < len(data):
                    pivot_lows.append({
                        'index': idx,
                        'price': data['Low'].iloc[idx],
                        'date': data.index[idx]
                    })
            
            for idx in local_maxima_idx:
                if idx < len(data):
                    pivot_highs.append({
                        'index': idx,
                        'price': data['High'].iloc[idx],
                        'date': data.index[idx]
                    })
            
            # Cluster similar price levels (within 1% tolerance)
            def cluster_levels(pivots, tolerance=0.01):
                if not pivots:
                    return []
                
                pivots.sort(key=lambda x: x['price'])
                clusters = []
                current_cluster = [pivots[0]]
                
                for i in range(1, len(pivots)):
                    price_diff = abs(pivots[i]['price'] - current_cluster[0]['price']) / current_cluster[0]['price']
                    if price_diff <= tolerance:
                        current_cluster.append(pivots[i])
                    else:
                        if len(current_cluster) >= min_touches:
                            avg_price = np.mean([p['price'] for p in current_cluster])
                            clusters.append({
                                'level': avg_price,
                                'touches': len(current_cluster),
                                'strength': len(current_cluster),
                                'last_touch': max(current_cluster, key=lambda x: x['index'])['index']
                            })
                        current_cluster = [pivots[i]]
                
                # Don't forget the last cluster
                if len(current_cluster) >= min_touches:
                    avg_price = np.mean([p['price'] for p in current_cluster])
                    clusters.append({
                        'level': avg_price,
                        'touches': len(current_cluster),
                        'strength': len(current_cluster),
                        'last_touch': max(current_cluster, key=lambda x: x['index'])['index']
                    })
                
                return clusters
            
            support_levels = cluster_levels(pivot_lows)
            resistance_levels = cluster_levels(pivot_highs)
            
            # Sort by strength (number of touches) and recency
            support_levels.sort(key=lambda x: (x['strength'], x['last_touch']), reverse=True)
            resistance_levels.sort(key=lambda x: (x['strength'], x['last_touch']), reverse=True)
            
            return {
                'support_levels': support_levels[:5],  # Top 5 support levels
                'resistance_levels': resistance_levels[:5]  # Top 5 resistance levels
            }
            
        except Exception as e:
            logger.error(f"Error finding support/resistance levels: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def check_consolidation_near_level(self, data: pd.DataFrame, level: float, window: int = 10, tolerance: float = 0.02) -> bool:
        """
        Check if price has been consolidating near a support/resistance level.
        
        Args:
            data: DataFrame with OHLCV data
            level: Price level to check
            window: Number of periods to check
            tolerance: Price tolerance as percentage
            
        Returns:
            Boolean indicating if price is consolidating near level
        """
        try:
            if len(data) < window:
                return False
            
            recent_data = data.tail(window)
            recent_closes = recent_data['Close'].values
            
            # Calculate how many prices are within tolerance of the level
            within_tolerance = 0
            for close in recent_closes:
                price_diff = abs(close - level) / level
                if price_diff <= tolerance:
                    within_tolerance += 1
            
            # Consider it consolidation if at least 60% of recent closes are near the level
            consolidation_ratio = within_tolerance / window
            return consolidation_ratio >= 0.6
            
        except Exception as e:
            logger.error(f"Error checking consolidation: {e}")
            return False
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Support/Resistance Breakout signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional signal columns
        """
        try:
            if len(data) < 50:  # Need sufficient data for pivot analysis
                logger.warning(f"{self.name}: Insufficient data for analysis")
                data['sr_breakout_signal'] = 0
                return data
            
            # Find support and resistance levels
            sr_levels = self.find_support_resistance_levels(data)
            
            # Calculate volume moving average
            data['volume_ma_20'] = data['Volume'].rolling(window=20, min_periods=10).mean()
            
            # Initialize signal column
            data['sr_breakout_signal'] = 0
            
            # Check for breakouts
            for i in range(20, len(data)):
                current_close = data['Close'].iloc[i]
                current_high = data['High'].iloc[i]
                current_low = data['Low'].iloc[i]
                current_volume = data['Volume'].iloc[i]
                avg_volume = data['volume_ma_20'].iloc[i]
                
                if pd.isna(avg_volume):
                    continue
                
                # Volume confirmation: at least 1.5x average volume
                volume_confirmation = current_volume >= (avg_volume * 1.5)
                
                # Check resistance breakouts (bullish)
                for resistance in sr_levels['resistance_levels']:
                    resistance_level = resistance['level']
                    
                    # Price must break above resistance with volume
                    if current_close > resistance_level and volume_confirmation:
                        # Additional confirmation: strong close (closing in top 75% of daily range)
                        daily_range = current_high - current_low
                        if daily_range > 0:
                            close_position = (current_close - current_low) / daily_range
                            
                            if close_position >= 0.75:
                                # Check if there was prior consolidation near this level
                                recent_data = data.iloc[max(0, i-10):i]
                                if self.check_consolidation_near_level(recent_data, resistance_level):
                                    data.loc[data.index[i], 'sr_breakout_signal'] = 1
                                    logger.debug(f"{self.name}: BUY signal - breakout above resistance {resistance_level:.2f} at {current_close:.2f}")
                                    break  # Only one signal per bar
                
                # Check support breakdowns (bearish)
                for support in sr_levels['support_levels']:
                    support_level = support['level']
                    
                    # Price must break below support with volume
                    if current_close < support_level and volume_confirmation:
                        # Additional confirmation: weak close (closing in bottom 25% of daily range)
                        daily_range = current_high - current_low
                        if daily_range > 0:
                            close_position = (current_close - current_low) / daily_range
                            
                            if close_position <= 0.25:
                                # Check if there was prior consolidation near this level
                                recent_data = data.iloc[max(0, i-10):i]
                                if self.check_consolidation_near_level(recent_data, support_level):
                                    data.loc[data.index[i], 'sr_breakout_signal'] = -1
                                    logger.debug(f"{self.name}: SELL signal - breakdown below support {support_level:.2f} at {current_close:.2f}")
                                    break  # Only one signal per bar
            
            return data
            
        except Exception as e:
            logger.error(f"Error in {self.name} calculation: {e}")
            data['sr_breakout_signal'] = 0
            return data
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Support/Resistance Breakout strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY, -1 for SELL, 0 for HOLD
        """
        try:
            if len(data) < 50:
                return 0
            
            # Calculate signals
            data_with_signals = self.calculate_signals(data)
            
            # Get the latest signal
            latest_signal = data_with_signals['sr_breakout_signal'].iloc[-1]
            
            return latest_signal
            
        except Exception as e:
            logger.error(f"Error running {self.name}: {e}")
            return 0
    
    def get_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate signal strength based on level significance and breakout quality.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            float: Signal strength between 0 and 1
        """
        try:
            if len(data) < 50:
                return 0.0
            
            # Find current support/resistance levels
            sr_levels = self.find_support_resistance_levels(data)
            
            latest_close = data['Close'].iloc[-1]
            latest_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20, min_periods=10).mean().iloc[-1]
            
            max_strength = 0.0
            
            # Check strength against resistance levels
            for resistance in sr_levels['resistance_levels']:
                if latest_close > resistance['level']:
                    # Volume strength
                    volume_strength = min(1.0, (latest_volume / avg_volume - 1.0) / 2.0) if avg_volume > 0 else 0.0
                    
                    # Level strength (based on number of touches)
                    level_strength = min(1.0, resistance['strength'] / 5.0)  # Normalize to 0-1
                    
                    # Breakout magnitude
                    breakout_strength = min(1.0, (latest_close - resistance['level']) / resistance['level'] * 10)
                    
                    overall_strength = (volume_strength * 0.4) + (level_strength * 0.3) + (breakout_strength * 0.3)
                    max_strength = max(max_strength, overall_strength)
            
            # Check strength against support levels
            for support in sr_levels['support_levels']:
                if latest_close < support['level']:
                    # Volume strength
                    volume_strength = min(1.0, (latest_volume / avg_volume - 1.0) / 2.0) if avg_volume > 0 else 0.0
                    
                    # Level strength (based on number of touches)
                    level_strength = min(1.0, support['strength'] / 5.0)  # Normalize to 0-1
                    
                    # Breakdown magnitude
                    breakdown_strength = min(1.0, (support['level'] - latest_close) / support['level'] * 10)
                    
                    overall_strength = (volume_strength * 0.4) + (level_strength * 0.3) + (breakdown_strength * 0.3)
                    max_strength = max(max_strength, overall_strength)
            
            return max_strength
            
        except Exception as e:
            logger.error(f"Error calculating signal strength for {self.name}: {e}")
            return 0.0
