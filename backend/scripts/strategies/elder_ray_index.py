"""
Elder Ray Index Strategy
File: scripts/strategies/elder_ray_index.py

This strategy uses the Elder Ray Index to identify buying/selling pressure and 
generate signals based on bullish/bearish power.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
import talib as ta


class Elder_Ray_Index(BaseStrategy):
    """
    Strategy based on Elder Ray Index (bullish and bearish power).

    Signals:
    - Bullish Power: Close is above EMA, and High - EMA is notable.
    - Bearish Power: Close is below EMA, and EMA - Low is notable.
    - Combined signals can indicate strong trends or reversals.
    """

    def __init__(self, params=None):
        """
        Initialize the Elder Ray Index strategy.

        Args:
            params: Dictionary with strategy parameters
                   - period: EMA calculation period (default: 13)
                   - threshold: Threshold factor for significant bullish/bearish power (default: 0.1)
        """
        super().__init__(params)
        self.period = self.get_parameter('period', 13)
        self.threshold = self.get_parameter('threshold', 0.1)

    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the Elder Ray Index strategy.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            int: 1 for BUY signal, -1 for SELL/NO_BUY signal
        """
        if not self.validate_data(data, min_periods=self.period + 10):
            self.log_signal(-1, "Insufficient data for Elder Ray analysis", data)
            return -1

        try:
            # Calculate EMA
            close_prices = data['Close'].values
            high_prices = data['High'].values
            low_prices = data['Low'].values

            ema = ta.EMA(close_prices, timeperiod=self.period)

            if len(ema) < 5 or np.isnan(ema[-1]):
                self.log_signal(-1, "Insufficient EMA data", data)
                return -1

            current_ema = ema[-1]
            prev_ema = ema[-2]

            # Calculate Bullish and Bearish Power
            bullish_power = high_prices[-1] - current_ema
            bearish_power = current_ema - low_prices[-1]

            # Calculate thresholds
            avg_trading_range = np.mean(high_prices - low_prices)
            significant_bp = avg_trading_range * self.threshold

            # Buy signal
            if bullish_power > significant_bp and close_prices[-1] > current_ema:
                self.log_signal(1, f"Bullish Power: {bullish_power:.2f} > significant {significant_bp:.2f}, EMA({current_ema:.2f})", data)
                return 1

            # Sell signal
            if bearish_power > significant_bp and close_prices[-1] < current_ema:
                self.log_signal(-1, f"Bearish Power: {bearish_power:.2f} > significant {significant_bp:.2f}, EMA({current_ema:.2f})", data)
                return -1

            # Neutral signal
            self.log_signal(-1, f"Neutral: Bullish {bullish_power:.2f}, Bearish {bearish_power:.2f}, EMA({current_ema:.2f})", data)
            return -1

        except Exception as e:
            self.log_signal(-1, f"Error in Elder Ray analysis: {str(e)}", data)
            return -1

