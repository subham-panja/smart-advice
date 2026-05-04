"""
MACD Signal Crossover Strategy
File: scripts/strategies/macd_signal_crossover.py

This strategy uses MACD (Moving Average Convergence Divergence) signal line crossovers
to generate buy and sell signals.
"""

import pandas as pd
import talib as ta

from .base_strategy import BaseStrategy


class MACD_Signal_Crossover(BaseStrategy):
    """
    MACD Signal Crossover Strategy.

    Buy Signal: MACD line crosses above signal line
    Sell Signal: MACD line crosses below signal line
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.fast_period = self.get_parameter("fast_period", 12)
        self.slow_period = self.get_parameter("slow_period", 26)
        self.signal_period = self.get_parameter("signal_period", 9)
        self.check_divergence = self.get_parameter("check_divergence", False)
        self.divergence_bars = self.get_parameter("divergence_bars", 10)
        self.block_on_bearish_divergence = self.get_parameter("block_on_bearish_divergence", False)

    def _execute_strategy_logic(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> int:
        """
        Execute the MACD signal crossover strategy logic.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            int: 1 for buy signal, -1 for sell/no signal
        """
        # Validate data
        min_periods = self.slow_period + self.signal_period
        if not self.validate_data(data, min_periods=min_periods):
            return -1

        try:
            # Calculate MACD using TA-Lib
            close_prices = data["Close"].values
            macd_line, signal_line, histogram = ta.MACD(
                close_prices, fastperiod=self.fast_period, slowperiod=self.slow_period, signalperiod=self.signal_period
            )

            # Check if we have valid MACD values
            if pd.isna(macd_line[-1]) or pd.isna(signal_line[-1]) or pd.isna(macd_line[-2]) or pd.isna(signal_line[-2]):
                self.log_signal(-1, "Insufficient data for MACD calculation", data)
                return -1

            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            previous_macd = macd_line[-2]
            previous_signal = signal_line[-2]
            current_histogram = histogram[-1]

            # Check for bearish divergence (price makes higher high but MACD makes lower high)
            if self.check_divergence:
                lookback = min(self.divergence_bars, len(close_prices))
                if lookback >= 5:
                    recent_closes = close_prices[-lookback:]

                    # Find price high and corresponding MACD high
                    price_high_idx = recent_closes.argmax()
                    macd_at_high = macd_line[-lookback + price_high_idx]

                    # Check if there's a more recent lower MACD high
                    if price_high_idx < lookback - 1:
                        later_macd_highs = macd_line[-lookback + price_high_idx + 1 : -1]
                        if len(later_macd_highs) > 0 and later_macd_highs.max() < macd_at_high:
                            if self.block_on_bearish_divergence:
                                reason = "BLOCKED: Bearish MACD divergence detected (price HH but MACD LH)"
                                self.log_signal(-1, reason, data, symbol=symbol)
                                return -1
                            else:
                                self.logger.debug(f"[{symbol}] Bearish MACD divergence warning (not blocking)")

            # Buy signal: MACD crosses above signal line
            if previous_macd <= previous_signal and current_macd > current_signal:
                reason = (
                    f"MACD bullish crossover: MACD ({current_macd:.4f}) crosses above signal ({current_signal:.4f})"
                )
                self.log_signal(1, reason, data, symbol=symbol)
                return 1

            # Sell signal: MACD crosses below signal line
            elif previous_macd >= previous_signal and current_macd < current_signal:
                reason = (
                    f"MACD bearish crossover: MACD ({current_macd:.4f}) crosses below signal ({current_signal:.4f})"
                )
                self.log_signal(-1, reason, data, symbol=symbol)
                return -1

            # Check if MACD is above signal line (bullish)
            elif current_macd > current_signal:
                # Additional check: prefer positive histogram (strengthening momentum)
                if current_histogram > 0:
                    reason = f"MACD bullish: MACD ({current_macd:.4f}) above signal ({current_signal:.4f}), positive histogram"
                    self.log_signal(1, reason, data, symbol=symbol)
                    return 1
                else:
                    reason = f"MACD bullish but weakening: MACD ({current_macd:.4f}) above signal ({current_signal:.4f}), negative histogram"
                    self.log_signal(1, reason, data, symbol=symbol)
                    return 1

            # MACD is below signal line (bearish)
            else:
                reason = f"MACD bearish: MACD ({current_macd:.4f}) below signal ({current_signal:.4f})"
                self.log_signal(-1, reason, data, symbol=symbol)
                return -1

        except Exception as e:
            self.log_signal(-1, f"Error in MACD calculation: {str(e)}", data)
            return -1
