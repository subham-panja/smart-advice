"""
Pocket Pivot Entry Strategy
File: scripts/strategies/pocket_pivot_entry.py

Based on Gil Morales and Chris Kacher's volume signature.
Buy Signal: Volume is greater than any down-day volume in the previous 10 days
while price is near/above key moving averages.
"""

import pandas as pd
import talib as ta

from .base_strategy import BaseStrategy


class Pocket_Pivot_Entry(BaseStrategy):
    def __init__(self, params=None):
        super().__init__(params)
        self.lookback = self.get_parameter("lookback", 10)
        self.sma_fast = self.get_parameter("sma_fast", 10)
        self.sma_slow = self.get_parameter("sma_slow", 50)

    def _execute_strategy_logic(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> int:
        if not self.validate_data(data, min_periods=max(self.sma_slow, self.lookback) + 5):
            return -1

        try:
            close = data["Close"].values
            volume = data["Volume"].values

            # 1. Moving Average Context
            sma10 = ta.SMA(close, timeperiod=self.sma_fast)
            sma50 = ta.SMA(close, timeperiod=self.sma_slow)

            curr_price = close[-1]
            if curr_price < sma10[-1] or curr_price < sma50[-1]:
                return -1  # Must be above key averages

            # 2. Volume Signature
            # Find the max volume on a "Down Day" in the last 10 days
            # A down day is where Close < Prev Close
            max_down_vol = 0
            for i in range(-self.lookback - 1, -1):
                if close[i] < close[i - 1]:
                    if volume[i] > max_down_vol:
                        max_down_vol = volume[i]

            curr_vol = volume[-1]
            if curr_vol > max_down_vol:
                self.log_signal(
                    1, f"Pocket Pivot detected: Volume ({curr_vol:,.0f}) > Max Down Vol ({max_down_vol:,.0f})", data
                )
                return 1

            return -1

        except Exception:
            return -1
