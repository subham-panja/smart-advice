"""
Volume Breakout Strategy
File: scripts/strategies/volume_breakout.py

This strategy identifies breakouts confirmed by significant volume spikes.
When price breaks above resistance or below support with 2x+ average volume,
it signals a potential strong move in the breakout direction.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from utils.logger import setup_logging

logger = setup_logging()


class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume Breakout Strategy for swing trading.
    
    Entry Conditions:
    - Price breaks above recent high (20-day) OR below recent low (20-day)
    - Volume is at least 2x the 20-day average volume
    - Price closes above/below the breakout level (confirmation)
    
    Exit Conditions:
    - Price moves 5% in favor OR 3% against
    - Volume drops below average for 3+ consecutive days
    """
    
    def __init__(self):
        super().__init__()
        self.name = "Volume_Breakout"
        self.description = "Volume-confirmed breakout above resistance or below support"
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Breakout trading signals.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with additional signal columns
        """
        try:
            if len(data) < 25:  # Need at least 25 days for calculations
                logger.warning(f"{self.name}: Insufficient data for analysis")
                data['volume_breakout_signal'] = 0
                return data
            
            # Calculate volume metrics
            data['volume_ma_20'] = data['Volume'].rolling(window=20, min_periods=10).mean()
            data['volume_ratio'] = data['Volume'] / data['volume_ma_20']
            
            # Calculate price levels (20-day high/low)
            data['resistance_20'] = data['High'].rolling(window=20, min_periods=10).max()
            data['support_20'] = data['Low'].rolling(window=20, min_periods=10).min()
            
            # Shift resistance/support to avoid look-ahead bias
            data['resistance_20'] = data['resistance_20'].shift(1)
            data['support_20'] = data['support_20'].shift(1)
            
            # Calculate average true range for volatility adjustment
            data['high_low'] = data['High'] - data['Low']
            data['high_close'] = np.abs(data['High'] - data['Close'].shift(1))
            data['low_close'] = np.abs(data['Low'] - data['Close'].shift(1))
            data['atr'] = data[['high_low', 'high_close', 'low_close']].max(axis=1)
            data['atr_14'] = data['atr'].rolling(window=14, min_periods=7).mean()
            
            # Initialize signal column
            data['volume_breakout_signal'] = 0
            
            for i in range(21, len(data)):  # Start from index 21 to have enough history
                current_close = data['Close'].iloc[i]
                current_volume = data['Volume'].iloc[i]
                volume_avg = data['volume_ma_20'].iloc[i]
                resistance = data['resistance_20'].iloc[i]
                support = data['support_20'].iloc[i]
                
                # Skip if we don't have valid data
                if pd.isna(volume_avg) or pd.isna(resistance) or pd.isna(support):
                    continue
                    
                # Volume condition: at least 2x average volume
                volume_spike = current_volume >= (volume_avg * 2.0)
                
                if volume_spike:
                    # Bullish breakout: Close above 20-day high
                    if current_close > resistance:
                        # Additional confirmation: close is in upper 75% of daily range
                        daily_range = data['High'].iloc[i] - data['Low'].iloc[i]
                        close_position = (current_close - data['Low'].iloc[i]) / daily_range if daily_range > 0 else 0
                        
                        if close_position >= 0.75:  # Strong close near high
                            data.loc[data.index[i], 'volume_breakout_signal'] = 1
                            logger.debug(f"{self.name}: BUY signal at {current_close} with volume {current_volume:.0f} (avg: {volume_avg:.0f})")
                    
                    # Bearish breakdown: Close below 20-day low
                    elif current_close < support:
                        # Additional confirmation: close is in lower 25% of daily range
                        daily_range = data['High'].iloc[i] - data['Low'].iloc[i]
                        close_position = (current_close - data['Low'].iloc[i]) / daily_range if daily_range > 0 else 0
                        
                        if close_position <= 0.25:  # Weak close near low
                            data.loc[data.index[i], 'volume_breakout_signal'] = -1
                            logger.debug(f"{self.name}: SELL signal at {current_close} with volume {current_volume:.0f} (avg: {volume_avg:.0f})")
            
            # Clean up temporary columns
            columns_to_drop = ['high_low', 'high_close', 'low_close', 'atr']
            data = data.drop(columns=columns_to_drop, errors='ignore')
            
            return data
            
        except Exception as e:
            logger.error(f"Error in {self.name} calculation: {e}")
            data['volume_breakout_signal'] = 0
            return data
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core Volume Breakout strategy logic.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY, -1 for SELL, 0 for HOLD (raw signal, before volume filtering)
        """
        try:
            if len(data) < 25:
                return 0
            
            # Calculate signals
            data_with_signals = self.calculate_signals(data)
            
            # Get and return the latest raw signal
            latest_signal = data_with_signals['volume_breakout_signal'].iloc[-1]
            return int(latest_signal)
            
        except Exception as e:
            logger.error(f"Error running {self.name}: {e}")
            return 0
    
    def get_signal_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate signal strength based on volume ratio and breakout magnitude.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            float: Signal strength between 0 and 1
        """
        try:
            if len(data) < 25:
                return 0.0
            
            latest_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(window=20, min_periods=10).mean().iloc[-1]
            latest_close = data['Close'].iloc[-1]
            
            # Calculate volume strength (0.0 to 1.0)
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1.0
            volume_strength = min(1.0, (volume_ratio - 1.0) / 2.0)  # Normalize to 0-1
            
            # Calculate breakout strength
            resistance = data['High'].rolling(window=20, min_periods=10).max().iloc[-2]  # Previous high
            support = data['Low'].rolling(window=20, min_periods=10).min().iloc[-2]      # Previous low
            
            breakout_strength = 0.0
            if not pd.isna(resistance) and latest_close > resistance:
                # Bullish breakout strength
                breakout_magnitude = (latest_close - resistance) / resistance
                breakout_strength = min(1.0, breakout_magnitude * 20)  # Scale to 0-1
            elif not pd.isna(support) and latest_close < support:
                # Bearish breakout strength
                breakout_magnitude = (support - latest_close) / support
                breakout_strength = min(1.0, breakout_magnitude * 20)  # Scale to 0-1
            
            # Combine volume and breakout strength
            overall_strength = (volume_strength * 0.6) + (breakout_strength * 0.4)
            
            return overall_strength
            
        except Exception as e:
            logger.error(f"Error calculating signal strength for {self.name}: {e}")
            return 0.0
