"""
OBV Bullish Divergence Strategy
File: scripts/strategies/obv_bullish_divergence.py

This strategy identifies bullish divergences between price and On-Balance Volume (OBV).
A bullish divergence occurs when price makes lower lows but OBV makes higher lows.
"""

import pandas as pd
import numpy as np
import talib as ta
from scripts.strategies.base_strategy import BaseStrategy


class OBV_Bullish_Divergence(BaseStrategy):
    """
    Strategy based on OBV bullish divergences.
    
    Bullish divergence signals:
    - Price makes lower lows while OBV makes higher lows
    - Indicates potential upward price reversal
    - Volume is supporting a bullish move despite price weakness
    """
    
    def __init__(self, params=None):
        """
        Initialize the OBV Bullish Divergence strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - lookback_period: Period to look for divergences (default: 20)
                   - min_pivot_distance: Minimum distance between pivots (default: 5)
                   - divergence_threshold: Minimum divergence strength (default: 0.02)
                   - confirmation_periods: Periods to confirm divergence (default: 3)
        """
        super().__init__(params)
        self.lookback_period = self.get_parameter('lookback_period', 20)
        self.min_pivot_distance = self.get_parameter('min_pivot_distance', 5)
        self.divergence_threshold = self.get_parameter('divergence_threshold', 0.02)  # 2%
        self.confirmation_periods = self.get_parameter('confirmation_periods', 3)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the OBV Bullish Divergence strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        min_periods = self.lookback_period + self.min_pivot_distance + 10
        if not self.validate_data(data, min_periods=min_periods):
            self.log_signal(-1, "Insufficient data for OBV divergence analysis", data)
            return -1
        
        try:
            close = data['Close'].values
            volume = data['Volume'].values.astype(float)  # Ensure float type for TA-Lib
            
            # Calculate OBV
            obv = ta.OBV(close, volume)
            
            if len(obv) < self.lookback_period or np.isnan(obv[-1]):
                self.log_signal(-1, "Insufficient OBV data", data)
                return -1
            
            # Find recent lows in both price and OBV
            recent_close = close[-self.lookback_period:]
            recent_obv = obv[-self.lookback_period:]
            
            # Find price lows (local minima)
            price_lows = self._find_local_minima(recent_close, self.min_pivot_distance)
            
            # Find OBV lows
            obv_lows = self._find_local_minima(recent_obv, self.min_pivot_distance)
            
            if len(price_lows) < 2 or len(obv_lows) < 2:
                self.log_signal(-1, f"Insufficient pivots: price lows {len(price_lows)}, OBV lows {len(obv_lows)}", data)
                return -1
            
            # Check for bullish divergence
            divergence_found = False
            divergence_strength = 0
            
            # Compare the two most recent lows
            if len(price_lows) >= 2 and len(obv_lows) >= 2:
                # Get the two most recent lows
                latest_price_low_idx = price_lows[-1]
                prev_price_low_idx = price_lows[-2]
                
                latest_obv_low_idx = obv_lows[-1]
                prev_obv_low_idx = obv_lows[-2]
                
                # Check if they are reasonably aligned (within acceptable range)
                price_alignment = abs(latest_price_low_idx - latest_obv_low_idx)
                if price_alignment <= self.min_pivot_distance:
                    
                    # Get the actual values
                    latest_price_low = recent_close[latest_price_low_idx]
                    prev_price_low = recent_close[prev_price_low_idx]
                    
                    latest_obv_low = recent_obv[latest_obv_low_idx]
                    prev_obv_low = recent_obv[prev_obv_low_idx]
                    
                    # Check for divergence: price lower low, OBV higher low
                    price_decline = (latest_price_low - prev_price_low) / prev_price_low
                    obv_improvement = (latest_obv_low - prev_obv_low) / abs(prev_obv_low) if prev_obv_low != 0 else 0
                    
                    # Bullish divergence condition
                    if price_decline < -self.divergence_threshold and obv_improvement > 0:
                        divergence_found = True
                        divergence_strength = abs(price_decline) + obv_improvement
                        
                        self.log_signal(1, f"Bullish OBV divergence: Price declined {price_decline*100:.2f}%, OBV improved {obv_improvement*100:.2f}%", data)
                        return 1
            
            # Check for alternative divergence patterns
            if not divergence_found:
                # Look for divergence with current price vs OBV trend
                current_close = close[-1]
                current_obv = obv[-1]
                
                # Compare current values with recent lows
                if len(price_lows) >= 1 and len(obv_lows) >= 1:
                    recent_price_low = recent_close[price_lows[-1]]
                    recent_obv_low = recent_obv[obv_lows[-1]]
                    
                    # Check if we're near recent lows but showing divergence
                    price_from_low = (current_close - recent_price_low) / recent_price_low
                    obv_from_low = (current_obv - recent_obv_low) / abs(recent_obv_low) if recent_obv_low != 0 else 0
                    
                    # Near price low but OBV showing strength
                    if (price_from_low < 0.03 and  # Within 3% of recent low
                        obv_from_low > 0.05):      # OBV improved by more than 5%
                        
                        # Additional confirmation: check OBV trend
                        recent_obv_trend = np.mean(obv[-self.confirmation_periods:]) - np.mean(obv[-self.confirmation_periods*2:-self.confirmation_periods])
                        if recent_obv_trend > 0:
                            self.log_signal(1, f"OBV strength near price low: price {price_from_low*100:.1f}% from low, OBV +{obv_from_low*100:.1f}%", data)
                            return 1
            
            # Check for general OBV momentum
            if len(obv) >= 10:
                obv_momentum = self._calculate_momentum(obv[-10:])
                price_momentum = self._calculate_momentum(close[-10:])
                
                # Positive OBV momentum with weak/negative price momentum
                if obv_momentum > 0.01 and price_momentum < 0.005:
                    momentum_divergence = obv_momentum - price_momentum
                    if momentum_divergence > 0.01:  # Significant momentum divergence
                        self.log_signal(1, f"OBV momentum divergence: OBV +{obv_momentum*100:.2f}%, Price +{price_momentum*100:.2f}%", data)
                        return 1
            
            # No bullish divergence found
            self.log_signal(-1, "No bullish OBV divergence detected", data)
            return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in OBV divergence analysis: {str(e)}", data)
            return -1
    
    def _find_local_minima(self, data: np.ndarray, min_distance: int) -> list:
        """
        Find local minima in the data with minimum distance between them.
        
        Args:
            data: Data array
            min_distance: Minimum distance between minima
            
        Returns:
            List of indices where local minima occur
        """
        minima = []
        
        for i in range(min_distance, len(data) - min_distance):
            # Check if current point is lower than surrounding points
            is_minimum = True
            current_value = data[i]
            
            # Check left side
            for j in range(max(0, i - min_distance), i):
                if data[j] <= current_value:
                    is_minimum = False
                    break
            
            if not is_minimum:
                continue
                
            # Check right side
            for j in range(i + 1, min(len(data), i + min_distance + 1)):
                if data[j] <= current_value:
                    is_minimum = False
                    break
            
            if is_minimum:
                minima.append(i)
        
        return minima
    
    def _calculate_momentum(self, data: np.ndarray) -> float:
        """
        Calculate momentum as the slope of linear regression.
        
        Args:
            data: Price or indicator data
            
        Returns:
            Momentum value (normalized)
        """
        if len(data) < 2:
            return 0.0
        
        x = np.arange(len(data))
        # Simple linear regression slope
        slope = np.polyfit(x, data, 1)[0]
        
        # Normalize by the mean value
        mean_value = np.mean(data)
        if mean_value != 0:
            return slope / mean_value
        else:
            return 0.0
