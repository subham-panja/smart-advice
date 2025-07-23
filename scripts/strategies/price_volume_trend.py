"""
Price Volume Trend Strategy
File: scripts/strategies/price_volume_trend.py

This strategy uses the Price Volume Trend (PVT) indicator to identify
accumulation/distribution patterns and generate buy/sell signals.
"""

import pandas as pd
import numpy as np
from scripts.strategies.base_strategy import BaseStrategy


class Price_Volume_Trend(BaseStrategy):
    """
    Strategy based on Price Volume Trend (PVT) indicator.
    
    PVT Formula:
    PVT = Previous PVT + (Volume * (Close - Previous Close) / Previous Close)
    
    Signals:
    - Rising PVT with rising prices: Bullish signal
    - Falling PVT with falling prices: Bearish signal
    - PVT divergences: Reversal signals
    - PVT crossovers with moving averages: Trend signals
    """
    
    def __init__(self, params=None):
        """
        Initialize the Price Volume Trend strategy.
        
        Args:
            params: Dictionary with strategy parameters
                   - ma_period: Moving average period for PVT (default: 14)
                   - trend_periods: Periods to determine trend direction (default: 5)
                   - divergence_periods: Periods to look for divergences (default: 10)
                   - min_pvt_change: Minimum PVT change for significant signal (default: 1000)
        """
        super().__init__(params)
        self.ma_period = self.get_parameter('ma_period', 14)
        self.trend_periods = self.get_parameter('trend_periods', 5)
        self.divergence_periods = self.get_parameter('divergence_periods', 10)
        self.min_pvt_change = self.get_parameter('min_pvt_change', 1000)
    
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Price Volume Trend strategy.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        min_periods = max(self.ma_period, self.divergence_periods) + 5
        if not self.validate_data(data, min_periods=min_periods):
            self.log_signal(-1, "Insufficient data for PVT analysis", data)
            return -1
        
        try:
            close = data['Close'].values
            volume = data['Volume'].values
            
            if len(close) < 2:
                self.log_signal(-1, "Need at least 2 periods for PVT calculation", data)
                return -1
            
            # Calculate PVT
            pvt = np.zeros(len(close))
            pvt[0] = 0  # Start with 0
            
            for i in range(1, len(close)):
                if close[i-1] != 0:  # Avoid division by zero
                    price_change_ratio = (close[i] - close[i-1]) / close[i-1]
                    pvt[i] = pvt[i-1] + (volume[i] * price_change_ratio)
                else:
                    pvt[i] = pvt[i-1]
            
            current_pvt = pvt[-1]
            prev_pvt = pvt[-2]
            
            # Calculate PVT moving average
            if len(pvt) >= self.ma_period:
                pvt_ma = np.convolve(pvt, np.ones(self.ma_period)/self.ma_period, mode='valid')
                if len(pvt_ma) > 0:
                    current_pvt_ma = pvt_ma[-1]
                    prev_pvt_ma = pvt_ma[-2] if len(pvt_ma) > 1 else current_pvt_ma
                else:
                    current_pvt_ma = current_pvt
                    prev_pvt_ma = prev_pvt
            else:
                current_pvt_ma = current_pvt
                prev_pvt_ma = prev_pvt
            
            # Analyze PVT trend
            recent_pvt = pvt[-self.trend_periods:]
            recent_prices = close[-self.trend_periods:]
            
            pvt_trend = np.polyfit(range(len(recent_pvt)), recent_pvt, 1)[0]  # Linear trend slope
            price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            
            # Normalize trends
            pvt_trend_normalized = pvt_trend / abs(np.mean(recent_pvt)) if np.mean(recent_pvt) != 0 else 0
            price_trend_normalized = price_trend / np.mean(recent_prices)
            
            # Check PVT magnitude for significance
            pvt_change = abs(current_pvt - prev_pvt)
            if pvt_change < self.min_pvt_change and abs(current_pvt) > self.min_pvt_change:
                # Small change in large PVT value - percentage based check
                pvt_change_pct = pvt_change / abs(current_pvt)
                if pvt_change_pct < 0.01:  # Less than 1% change
                    self.log_signal(-1, f"Insignificant PVT change: {pvt_change_pct*100:.2f}%", data)
                    return -1
            
            # Generate signals
            
            # 1. PVT and Price trend alignment
            if pvt_trend_normalized > 0.001 and price_trend_normalized > 0:
                # Both PVT and price trending up
                trend_strength = min(abs(pvt_trend_normalized), abs(price_trend_normalized)) * 1000
                if trend_strength > 1:
                    self.log_signal(1, f"Bullish PVT alignment: PVT trend {pvt_trend_normalized*1000:.2f}, price trend {price_trend_normalized*100:.2f}%", data)
                    return 1
            
            elif pvt_trend_normalized < -0.001 and price_trend_normalized < 0:
                # Both PVT and price trending down
                self.log_signal(-1, f"Bearish PVT alignment: PVT trend {pvt_trend_normalized*1000:.2f}, price trend {price_trend_normalized*100:.2f}%", data)
                return -1
            
            # 2. PVT crossover signals
            if current_pvt > current_pvt_ma and prev_pvt <= prev_pvt_ma:
                # PVT crosses above its moving average
                crossover_strength = (current_pvt - current_pvt_ma) / abs(current_pvt_ma) if current_pvt_ma != 0 else 0
                if abs(crossover_strength) > 0.05:  # 5% crossover
                    self.log_signal(1, f"Bullish PVT crossover: {crossover_strength*100:.2f}% above MA", data)
                    return 1
            
            elif current_pvt < current_pvt_ma and prev_pvt >= prev_pvt_ma:
                # PVT crosses below its moving average
                crossover_strength = (current_pvt_ma - current_pvt) / abs(current_pvt_ma) if current_pvt_ma != 0 else 0
                if abs(crossover_strength) > 0.05:  # 5% crossover
                    self.log_signal(-1, f"Bearish PVT crossover: {crossover_strength*100:.2f}% below MA", data)
                    return -1
            
            # 3. PVT divergence analysis
            if len(pvt) >= self.divergence_periods:
                # Check for bullish divergence: price making lower lows, PVT making higher lows
                recent_price_min_idx = np.argmin(recent_prices)
                recent_pvt_min_idx = np.argmin(recent_pvt)
                
                # Look for divergence pattern
                if len(close) >= self.divergence_periods * 2:
                    older_prices = close[-self.divergence_periods*2:-self.divergence_periods]
                    older_pvt = pvt[-self.divergence_periods*2:-self.divergence_periods]
                    
                    older_price_min = np.min(older_prices)
                    older_pvt_min = np.min(older_pvt)
                    recent_price_min = np.min(recent_prices)
                    recent_pvt_min = np.min(recent_pvt)
                    
                    # Bullish divergence: price lower low, PVT higher low
                    if (recent_price_min < older_price_min * 0.98 and  # Price made significantly lower low
                        recent_pvt_min > older_pvt_min * 1.02):        # PVT made higher low
                        
                        price_decline = (recent_price_min - older_price_min) / older_price_min
                        pvt_improvement = (recent_pvt_min - older_pvt_min) / abs(older_pvt_min) if older_pvt_min != 0 else 0
                        
                        self.log_signal(1, f"Bullish PVT divergence: price {price_decline*100:.2f}%, PVT +{pvt_improvement*100:.2f}%", data)
                        return 1
                    
                    # Bearish divergence: price higher high, PVT lower high
                    older_price_max = np.max(older_prices)
                    older_pvt_max = np.max(older_pvt)
                    recent_price_max = np.max(recent_prices)
                    recent_pvt_max = np.max(recent_pvt)
                    
                    if (recent_price_max > older_price_max * 1.02 and  # Price made higher high
                        recent_pvt_max < older_pvt_max * 0.98):        # PVT made lower high
                        
                        self.log_signal(-1, f"Bearish PVT divergence: price higher high, PVT lower high", data)
                        return -1
            
            # 4. Current PVT position analysis
            if current_pvt > current_pvt_ma:
                # PVT above its moving average
                pvt_strength = (current_pvt - current_pvt_ma) / abs(current_pvt_ma) if current_pvt_ma != 0 else 0
                if pvt_trend_normalized > 0:
                    self.log_signal(1, f"PVT bullish: {pvt_strength*100:.2f}% above MA, rising trend", data)
                    return 1
                else:
                    self.log_signal(-1, f"PVT mixed: above MA but declining trend", data)
                    return -1
            else:
                # PVT below its moving average
                pvt_weakness = (current_pvt_ma - current_pvt) / abs(current_pvt_ma) if current_pvt_ma != 0 else 0
                if pvt_trend_normalized < 0:
                    self.log_signal(-1, f"PVT bearish: {pvt_weakness*100:.2f}% below MA, falling trend", data)
                    return -1
                else:
                    # Below MA but rising - potential recovery
                    if pvt_trend_normalized > 0.001:
                        self.log_signal(1, f"PVT recovery: below MA but rising trend {pvt_trend_normalized*1000:.2f}", data)
                        return 1
                    else:
                        self.log_signal(-1, f"PVT weak: below MA with flat trend", data)
                        return -1
            
        except Exception as e:
            self.log_signal(-1, f"Error in PVT analysis: {str(e)}", data)
            return -1
