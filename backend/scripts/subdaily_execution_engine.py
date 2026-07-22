"""
Sub-Daily (4H / 1H) Execution Engine
=====================================

Provides multi-timeframe precision execution:
1. Resampling 60-minute candles to 4-Hour or 75-Minute bars.
2. Sub-daily intraday entry confirmation.
3. Intraday trailing stop-loss stepping based on sub-daily swing lows.
4. Intraday pyramiding evaluation on sub-daily ATR triggers.
5. Intraday exit triggers for immediate risk containment.
"""

import logging
from typing import Any, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def resample_to_4h(df_60m: pd.DataFrame) -> pd.DataFrame:
    """Resamples 60-minute OHLCV data into 4-Hour bars."""
    if df_60m is None or df_60m.empty:
        return pd.DataFrame()

    resampled = (
        df_60m.resample("4h")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )
    return resampled


def calculate_subdaily_swing_low(df_sub: pd.DataFrame, window: int = 6) -> float:
    """Calculates the recent sub-daily swing low over the specified bar window."""
    if df_sub is None or len(df_sub) < window:
        return 0.0
    return float(df_sub["Low"].tail(window).min())


def check_subdaily_pyramid_trigger(
    pos: Any,
    subdaily_df: pd.DataFrame,
    date: pd.Timestamp,
    atr: float,
    pyramid_cfg: dict,
) -> Optional[Tuple[float, float]]:
    """Evaluates if sub-daily candles triggered a pyramid step add.

    Returns (trigger_price, trigger_time) or None.
    """
    if subdaily_df is None or subdaily_df.empty:
        return None

    steps = pyramid_cfg.get("steps", [])
    if pos.adds_count >= len(steps):
        return None

    step = steps[pos.adds_count]
    trigger_mult = step.get("trigger_step_atr", 1.5)
    required_price = pos.last_add_price + (trigger_mult * atr)

    date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
    day_bars = subdaily_df.loc[date_str] if date_str in subdaily_df.index else subdaily_df.tail(6)

    for idx, row in day_bars.iterrows():
        if row["Close"] >= required_price or row["High"] >= required_price:
            return float(row["Close"]), idx

    return None


def calculate_subdaily_trailing_sl(
    pos: Any,
    subdaily_df: pd.DataFrame,
    atr: float,
    current_sl: float,
    buffer_atr_mult: float = 0.5,
) -> float:
    """Trails stop-loss directly below the recent 4H/1H swing low."""
    if subdaily_df is None or len(subdaily_df) < 4:
        return current_sl

    recent_low = calculate_subdaily_swing_low(subdaily_df, window=6)
    if recent_low <= 0:
        return current_sl

    proposed_sl = recent_low - (buffer_atr_mult * atr)
    return max(current_sl, proposed_sl)


def check_subdaily_intraday_exit(
    pos: Any,
    subdaily_df: pd.DataFrame,
    date: pd.Timestamp,
) -> Optional[Tuple[str, float]]:
    """Checks sub-daily intraday candles for instant Stop Loss or Target breaches.

    Returns (exit_reason, exit_price) or None.
    """
    if subdaily_df is None or subdaily_df.empty:
        return None

    date_str = str(date.date()) if hasattr(date, "date") else str(date)[:10]
    day_bars = subdaily_df.loc[date_str] if date_str in subdaily_df.index else subdaily_df.tail(6)

    for idx, row in day_bars.iterrows():
        if pos.current_stop_loss > 0 and row["Low"] <= pos.current_stop_loss:
            return "SUBDAILY_STOP_LOSS", pos.current_stop_loss

    return None
