"""
Fibonacci Retracement Strategy
File: scripts/strategies/fibonacci_retracement.py

This strategy uses Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%, 78.6%)
to identify potential support and resistance areas during pullbacks within a trend.
Traders look for bounce opportunities at key Fibonacci levels.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from utils.volume_analysis import get_enhanced_volume_confirmation
from utils.logger import setup_logging

logger = setup_logging()


class FibonacciRetracementStrategy(BaseStrategy):
    """
    Fibonacci Retracement Strategy for swing trading.
    
    Logic:
    1. Identify significant trend (swing high to swing low)
    2. Calculate Fibonacci retracement levels
    3. Look for price bounces at key levels (38.2%, 50%, 61.8%)
    4. Enter in direction of main trend after bounce confirmation
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Fibonacci_Retracement"
        self.description = "Pullback entries at Fibonacci retracement levels"
        
        # Fibonacci retracement levels
        self.fib_levels = {
            '23.6': 0.236,
            '38.2': 0.382,
            '50.0': 0.500,
            '61.8': 0.618,
            '78.6': 0.786
        }
        
        # Key levels for trading (most significant)
        self.key_fib_levels = [0.382, 0.500, 0.618]
    
    def find_swing_points(self, data: pd.DataFrame, window: int = 10) -> dict:
        """
        Find significant swing highs and lows for Fibonacci calculation.
        
        Args:
            data: DataFrame with OHLCV data
            window: Window for swing point detection
            
        Returns:
            Dictionary with swing high and swing low information
        """
        try:
            if len(data) < window * 3:
                return None
            
            # Calculate recent high and low (last 20-50 periods)
            lookback_period = min(50, len(data) - 1)
            recent_data = data.tail(lookback_period)
            
            # Find the highest high and lowest low in recent period
            swing_high_idx = recent_data['High'].idxmax()
            swing_low_idx = recent_data['Low'].idxmin()
            
            swing_high_price = recent_data.loc[swing_high_idx, 'High']
            swing_low_price = recent_data.loc[swing_low_idx, 'Low']
            
            # Determine trend direction based on which came first
            swing_high_pos = list(recent_data.index).index(swing_high_idx)
            swing_low_pos = list(recent_data.index).index(swing_low_idx)
            
            # Calculate Fibonacci levels
            price_range = swing_high_price - swing_low_price
            
            if price_range <= 0:
                return None
            
            fib_levels = {}
            
            # For uptrend (swing low to swing high)
            if swing_low_pos < swing_high_pos:
                trend_direction = 'uptrend'
                for level_name, level_ratio in self.fib_levels.items():
                    fib_levels[level_name] = swing_high_price - (price_range * level_ratio)
            else:
                # For downtrend (swing high to swing low) 
                trend_direction = 'downtrend'
                for level_name, level_ratio in self.fib_levels.items():
                    fib_levels[level_name] = swing_low_price + (price_range * level_ratio)
            
            return {
                'trend_direction': trend_direction,
                'swing_high': swing_high_price,
                'swing_low': swing_low_price,
                'swing_high_idx': swing_high_idx,
                'swing_low_idx': swing_low_idx,
                'price_range': price_range,
                'fib_levels': fib_levels
            }
            
        except Exception as e:
            logger.error(f"Error finding swing points: {e}")
            return None
    
    def check_bounce_at_fib_level(self, data: pd.DataFrame, current_idx: int, fib_level: float, trend_direction: str, tolerance: float = 0.005) -> bool:
        """
        Check if price bounced at a Fibonacci level.
        
        Args:
            data: DataFrame with OHLCV data
            current_idx: Current bar index
            fib_level: Fibonacci level price
            trend_direction: 'uptrend' or 'downtrend'
            tolerance: Price tolerance as percentage
            
        Returns:
            Boolean indicating if there was a bounce
        """
        try:
            if current_idx < 2:
                return False
            
            current_close = data['Close'].iloc[current_idx]
            prev_close = data['Close'].iloc[current_idx - 1]
            prev2_close = data['Close'].iloc[current_idx - 2] if current_idx >= 2 else prev_close
            
            current_low = data['Low'].iloc[current_idx]
            current_high = data['High'].iloc[current_idx]
            
            # Check if price touched the Fibonacci level within tolerance
            level_touched = False
            
            if trend_direction == 'uptrend':
                # In uptrend, look for bounce off support (Fib level acts as support)
                if current_low <= fib_level * (1 + tolerance) and current_low >= fib_level * (1 - tolerance):
                    level_touched = True
                    # Confirm bounce: current close should be higher than the low and showing recovery
                    bounce_confirmed = (current_close > current_low * 1.005 and 
                                      current_close > prev_close)
                else:
                    bounce_confirmed = False
            else:
                # In downtrend, look for bounce off resistance (Fib level acts as resistance)  
                if current_high >= fib_level * (1 - tolerance) and current_high <= fib_level * (1 + tolerance):
                    level_touched = True
                    # Confirm bounce: current close should be lower than the high and showing rejection
                    bounce_confirmed = (current_close < current_high * 0.995 and 
                                      current_close < prev_close)
                else:
                    bounce_confirmed = False
            
            return level_touched and bounce_confirmed
            
        except Exception as e:
            logger.error(f"Error checking bounce at Fib level: {e}")
            return False
    
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Fibonacci Retracement trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional signal columns
        """
        try:
            if len(data) < 30:  # Need sufficient data for swing analysis
                logger.warning(f"{self.name}: Insufficient data for analysis")
                data['fib_retracement_signal'] = 0
                return data
            
            # Initialize signal column
            data['fib_retracement_signal'] = 0
            
            # Calculate moving averages for trend confirmation
            data['sma_20'] = data['Close'].rolling(window=20, min_periods=10).mean()
            data['sma_50'] = data['Close'].rolling(window=50, min_periods=25).mean()
            
            # Calculate volume moving average
            data['volume_ma_20'] = data['Volume'].rolling(window=20, min_periods=10).mean()
            
            # Analyze each bar starting from sufficient history
            for i in range(25, len(data)):
                # Find swing points up to current bar
                current_data = data.iloc[:i+1]  # Data up to current bar (no future data)
                swing_info = self.find_swing_points(current_data)
                
                if swing_info is None:
                    continue
                
                current_close = data['Close'].iloc[i]
                current_volume = data['Volume'].iloc[i]
                avg_volume = data['volume_ma_20'].iloc[i]
                sma_20 = data['sma_20'].iloc[i]
                sma_50 = data['sma_50'].iloc[i]
                
                if pd.isna(avg_volume) or pd.isna(sma_20) or pd.isna(sma_50):
                    continue
                
                trend_direction = swing_info['trend_direction']
                
                # Check for bounces at key Fibonacci levels
                for level_ratio in self.key_fib_levels:
                    level_name = f"{level_ratio*100:.1f}"
                    if level_name in swing_info['fib_levels']:
                        fib_level = swing_info['fib_levels'][level_name]
                        
                        # Check for bounce at this level
                        if self.check_bounce_at_fib_level(data, i, fib_level, trend_direction):
                            
                            # Enhanced volume confirmation
                            volume_info = get_enhanced_volume_confirmation(current_data, signal_type=trend_direction)
                            volume_factor = volume_info['factor']
                            
                            if trend_direction == 'uptrend':
                                # Bullish signal: bounce in uptrend + trend confirmation
                                trend_confirmation = sma_20 > sma_50  # Uptrend confirmed
                                
                                if volume_factor >= 1.0 and trend_confirmation:
                                    data.loc[data.index[i], 'fib_retracement_signal'] = 1
                                    logger.debug(f"{self.name}: BUY signal - bounce at {level_name}% Fib level ({fib_level:.2f}) with volume factor {volume_factor}")
                                    break  # Only one signal per bar
                                    
                            elif trend_direction == 'downtrend':
                                # Bearish signal: bounce in downtrend + trend confirmation
                                trend_confirmation = sma_20 < sma_50  # Downtrend confirmed
                                
                                if volume_factor >= 1.0 and trend_confirmation:
                                    data.loc[data.index[i], 'fib_retracement_signal'] = -1
                                    logger.debug(f"{self.name}: SELL signal - bounce at {level_name}% Fib level ({fib_level:.2f}) with volume factor {volume_factor}")
                                    break  # Only one signal per bar
            
            return data
            
        except Exception as e:
            logger.error(f"Error in {self.name} calculation: {e}")
            data['fib_retracement_signal'] = 0
            return data
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Fibonacci Retracement strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY, -1 for SELL, 0 for HOLD
        """
        try:
            if len(data) < 30:
                return 0
            
            # Calculate signals
            data_with_signals = self.calculate_signals(data)
            
            # Get the latest signal
            latest_signal = data_with_signals['fib_retracement_signal'].iloc[-1]
            
            # Additional validation
            if latest_signal != 0:
                latest_close = data['Close'].iloc[-1]
                latest_volume = data['Volume'].iloc[-1]
                avg_volume = data['Volume'].rolling(window=20, min_periods=10).mean().iloc[-1]
                
                # Confirm trend direction with moving averages
                sma_20 = data['Close'].rolling(window=20, min_periods=10).mean().iloc[-1]
                sma_50 = data['Close'].rolling(window=50, min_periods=25).mean().iloc[-1]
                
                if not pd.isna(sma_20) and not pd.isna(sma_50):
                    if latest_signal == 1 and sma_20 > sma_50:  # Bullish in uptrend
                        return 1
                    elif latest_signal == -1 and sma_20 < sma_50:  # Bearish in downtrend
                        return -1
                    else:
                        logger.debug(f"{self.name}: Signal filtered out due to trend conflict")
                        return 0
                else:
                    return int(latest_signal)  # Accept signal if we can't confirm trend
            
            return 0
            
        except Exception as e:
            logger.error(f"Error running {self.name}: {e}")
            return 0
    
    def get_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate signal strength based on Fibonacci level significance and bounce quality.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            float: Signal strength between 0 and 1
        """
        try:
            if len(data) < 30:
                return 0.0
            
            # Find current swing setup
            swing_info = self.find_swing_points(data)
            
            if swing_info is None:
                return 0.0
            
            latest_close = data['Close'].iloc[-1]
            latest_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20, min_periods=10).mean().iloc[-1]
            
            max_strength = 0.0
            
            # Check proximity and bounce quality at key Fibonacci levels
            for level_ratio in self.key_fib_levels:
                level_name = f"{level_ratio*100:.1f}"
                if level_name in swing_info['fib_levels']:
                    fib_level = swing_info['fib_levels'][level_name]
                    
                    # Calculate distance from Fibonacci level
                    distance_from_fib = abs(latest_close - fib_level) / fib_level
                    proximity_strength = max(0.0, 1.0 - (distance_from_fib * 50))  # Strong if within 2%
                    
                    # Volume strength
                    volume_strength = min(1.0, (latest_volume / avg_volume - 0.8) / 1.2) if avg_volume > 0 else 0.0
                    
                    # Fibonacci level significance (61.8% and 50% are stronger)
                    if level_ratio == 0.618:
                        level_significance = 1.0  # Golden ratio - strongest
                    elif level_ratio == 0.500:
                        level_significance = 0.9  # 50% retracement - very strong
                    elif level_ratio == 0.382:
                        level_significance = 0.8  # 38.2% - strong
                    else:
                        level_significance = 0.6
                    
                    # Combine factors
                    level_strength = (proximity_strength * 0.4) + (volume_strength * 0.3) + (level_significance * 0.3)
                    max_strength = max(max_strength, level_strength)
            
            return max_strength
            
        except Exception as e:
            logger.error(f"Error calculating signal strength for {self.name}: {e}")
            return 0.0
