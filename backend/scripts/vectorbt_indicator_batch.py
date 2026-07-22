"""
Vectorbt-based batch indicator computation.

Computes all TA-Lib indicators once for all symbols across all dates
using vectorbt's 2D array support, replacing per-symbol per-date TA-Lib calls.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import vectorbt as vbt


@dataclass
class ComputedIndicators:
    """Holds all pre-computed indicators as 2D DataFrames (dates x symbols)."""

    symbols: List[str]
    dates: pd.DatetimeIndex

    # Price series
    close: pd.DataFrame = field(repr=False)
    high: pd.DataFrame = field(repr=False)
    low: pd.DataFrame = field(repr=False)
    open_price: pd.DataFrame = field(repr=False)
    volume: pd.DataFrame = field(repr=False)

    # TA-Lib indicators
    adx: pd.DataFrame = field(repr=False)
    pdi: pd.DataFrame = field(repr=False)  # PLUS_DI
    mdi: pd.DataFrame = field(repr=False)  # MINUS_DI
    sma_20: pd.DataFrame = field(repr=False)
    sma_50: pd.DataFrame = field(repr=False)
    sma_150: pd.DataFrame = field(repr=False)
    sma_200: pd.DataFrame = field(repr=False)
    atr_14: pd.DataFrame = field(repr=False)
    macd: pd.DataFrame = field(repr=False)
    macd_signal: pd.DataFrame = field(repr=False)
    macd_hist: pd.DataFrame = field(repr=False)
    rsi_14: pd.DataFrame = field(repr=False)
    rsi_9: pd.DataFrame = field(repr=False)
    bb_upper: pd.DataFrame = field(repr=False)
    bb_middle: pd.DataFrame = field(repr=False)
    bb_lower: pd.DataFrame = field(repr=False)
    ema_21: pd.DataFrame = field(repr=False)
    obv: pd.DataFrame = field(repr=False)

    # Derived / weekly
    weekly_sma_10: Optional[pd.DataFrame] = field(repr=False, default=None)
    weekly_sma_30: Optional[pd.DataFrame] = field(repr=False, default=None)
    weekly_rsi_14: Optional[pd.DataFrame] = field(repr=False, default=None)
    vwap_20: Optional[pd.DataFrame] = field(repr=False, default=None)


class IndicatorStore:
    """O(1) lookup for pre-computed indicators by (symbol, date)."""

    def __init__(self, indicators: ComputedIndicators):
        self._indicators = indicators
        self._symbol_to_col: Dict[str, str] = {}
        for i, sym in enumerate(indicators.symbols):
            self._symbol_to_col[sym] = sym  # columns use symbol name directly

        # Flatten each DataFrame into {symbol: {date: value}} for fast lookup
        self._series_cache: Dict[str, pd.Series] = {}
        for attr_name in [
            "close",
            "high",
            "low",
            "open_price",
            "volume",
            "adx",
            "pdi",
            "mdi",
            "sma_20",
            "sma_50",
            "sma_150",
            "sma_200",
            "atr_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "rsi_14",
            "rsi_9",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "ema_21",
            "obv",
            "weekly_sma_10",
            "weekly_sma_30",
            "weekly_rsi_14",
            "vwap_20",
        ]:
            df = getattr(indicators, attr_name, None)
            if df is not None:
                self._series_cache[attr_name] = df

    def get(self, symbol: str, indicator: str, date: pd.Timestamp) -> float:
        """Get a single indicator value for a symbol at a specific date."""
        df = self._series_cache.get(indicator)
        if df is None:
            return np.nan
        col = self._symbol_to_col.get(symbol, symbol)
        if col not in df.columns:
            return np.nan
        try:
            return df.at[date, col]
        except (KeyError, TypeError):
            return np.nan

    def get_series(
        self,
        symbol: str,
        indicator: str,
        end_date: pd.Timestamp,
        lookback: Optional[int] = None,
    ) -> pd.Series:
        """Get historical series for a symbol up to end_date, optionally limited to lookback bars."""
        df = self._series_cache.get(indicator)
        if df is None:
            return pd.Series(dtype=float)
        col = self._symbol_to_col.get(symbol, symbol)
        if col not in df.columns:
            return pd.Series(dtype=float)
        try:
            series = df.loc[:end_date, col]
        except (KeyError, TypeError):
            return pd.Series(dtype=float)
        if lookback is not None and len(series) > lookback:
            series = series.iloc[-lookback:]
        return series

    def get_full_series(self, symbol: str, indicator: str) -> pd.Series:
        """Get the full historical series for a symbol and indicator."""
        df = self._series_cache.get(indicator)
        if df is None:
            return pd.Series(dtype=float)
        col = self._symbol_to_col.get(symbol, symbol)
        if col not in df.columns:
            return pd.Series(dtype=float)
        return df[col]


def _align_to_common_dates(
    symbols_data: Dict[str, pd.DataFrame],
) -> tuple[pd.DatetimeIndex, Dict[str, pd.DataFrame]]:
    """Align all symbols to a common date index, forward-filling gaps within each symbol's range."""
    all_dates = set()
    for df in symbols_data.values():
        all_dates.update(df.index)
    common_dates = pd.DatetimeIndex(sorted(all_dates))

    aligned = {}
    for sym, df in symbols_data.items():
        cols = ["Open", "High", "Low", "Close", "Volume"]
        reindexed = df[cols].reindex(common_dates).ffill()
        last_real_date = df.index[-1]
        reindexed.loc[common_dates > last_real_date] = np.nan
        aligned[sym] = reindexed

    return common_dates, aligned


def compute_all_indicators(
    symbols_data: Dict[str, pd.DataFrame],
    strategy_config: Optional[Dict[str, Any]] = None,
) -> ComputedIndicators:
    """Compute all indicators for all symbols in one vectorized pass.

    Args:
        symbols_data: {symbol: OHLCV DataFrame} with columns Open/High/Low/Close/Volume
        strategy_config: strategy config dict (used to determine which
            weekly indicators are needed based on MTF_GATE settings)

    Returns:
        ComputedIndicators with 2D DataFrames (dates x symbols)
    """
    common_dates, aligned = _align_to_common_dates(symbols_data)
    symbols = list(symbols_data.keys())

    if not symbols or len(common_dates) < 2:
        raise ValueError(
            f"Need at least 2 common dates and 1 symbol, got {len(common_dates)} dates, {len(symbols)} symbols"
        )

    # Stack into 2D DataFrames
    close = pd.concat({s: aligned[s]["Close"] for s in symbols}, axis=1)
    high = pd.concat({s: aligned[s]["High"] for s in symbols}, axis=1)
    low = pd.concat({s: aligned[s]["Low"] for s in symbols}, axis=1)
    open_price = pd.concat({s: aligned[s]["Open"] for s in symbols}, axis=1)
    volume = pd.concat({s: aligned[s]["Volume"] for s in symbols}, axis=1).astype(float)

    # Flatten column MultiIndex to single-level (symbol names)
    close.columns = symbols
    high.columns = symbols
    low.columns = symbols
    open_price.columns = symbols
    volume.columns = symbols

    # --- TA-Lib indicators via vbt.talib (batch across all symbols) ---

    # ADX / DI
    adx = vbt.talib("ADX").run(high, low, close, timeperiod=14).real
    pdi = vbt.talib("PLUS_DI").run(high, low, close, timeperiod=14).real
    mdi = vbt.talib("MINUS_DI").run(high, low, close, timeperiod=14).real
    adx.columns = symbols
    pdi.columns = symbols
    mdi.columns = symbols

    # SMAs
    sma_20 = vbt.talib("SMA").run(close, timeperiod=20).real
    sma_50 = vbt.talib("SMA").run(close, timeperiod=50).real
    sma_150 = vbt.talib("SMA").run(close, timeperiod=150).real
    sma_200 = vbt.talib("SMA").run(close, timeperiod=200).real
    sma_20.columns = symbols
    sma_50.columns = symbols
    sma_150.columns = symbols
    sma_200.columns = symbols

    # ATR
    atr_14 = vbt.talib("ATR").run(high, low, close, timeperiod=14).real
    atr_14.columns = symbols

    # MACD
    macd_result = vbt.talib("MACD").run(close, fastperiod=12, slowperiod=26, signalperiod=9)
    macd = macd_result.macd
    macd_signal = macd_result.macdsignal
    macd_hist = macd_result.macdhist
    macd.columns = symbols
    macd_signal.columns = symbols
    macd_hist.columns = symbols

    # RSI
    rsi_14 = vbt.talib("RSI").run(close, timeperiod=14).real
    rsi_9 = vbt.talib("RSI").run(close, timeperiod=9).real
    rsi_14.columns = symbols
    rsi_9.columns = symbols

    # BBANDS
    bb_result = vbt.talib("BBANDS").run(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_upper = bb_result.upperband
    bb_middle = bb_result.middleband
    bb_lower = bb_result.lowerband
    bb_upper.columns = symbols
    bb_middle.columns = symbols
    bb_lower.columns = symbols

    # EMA(21)
    ema_21 = vbt.talib("EMA").run(close, timeperiod=21).real
    ema_21.columns = symbols

    # OBV
    obv = vbt.talib("OBV").run(close, volume).real
    obv.columns = symbols

    # --- Weekly indicators (for MTF_GATE) ---
    need_weekly = False
    if strategy_config:
        gates_cfg = strategy_config.get("swing_trading_gates", {})
        need_weekly = gates_cfg.get("MTF_GATE", {}).get("enabled", False)

    weekly_sma_10 = None
    weekly_sma_30 = None
    weekly_rsi_14 = None
    vwap_20 = None

    if need_weekly:
        weekly_sma_10, weekly_sma_30, weekly_rsi_14, vwap_20 = _compute_weekly_indicators(
            close, high, low, volume, common_dates
        )

    return ComputedIndicators(
        symbols=symbols,
        dates=common_dates,
        close=close,
        high=high,
        low=low,
        open_price=open_price,
        volume=volume,
        adx=adx,
        pdi=pdi,
        mdi=mdi,
        sma_20=sma_20,
        sma_50=sma_50,
        sma_150=sma_150,
        sma_200=sma_200,
        atr_14=atr_14,
        macd=macd,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        rsi_14=rsi_14,
        rsi_9=rsi_9,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        ema_21=ema_21,
        obv=obv,
        weekly_sma_10=weekly_sma_10,
        weekly_sma_30=weekly_sma_30,
        weekly_rsi_14=weekly_rsi_14,
        vwap_20=vwap_20,
    )


def _compute_weekly_indicators(
    close: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    volume: pd.DataFrame,
    common_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute weekly-resampled indicators and map back to daily dates using fully vectorized 2D operations."""

    # Resample all 2D DataFrames to weekly at once (eliminates 1300-stock for loop)
    w_close = close.resample("W").last()

    # Vectorized TA-Lib runs on all symbols at once
    sma10 = vbt.talib("SMA").run(w_close, timeperiod=10).real
    sma30 = vbt.talib("SMA").run(w_close, timeperiod=30).real
    rsi14 = vbt.talib("RSI").run(w_close, timeperiod=14).real

    # Reindex back to daily, ffill
    weekly_sma_10 = sma10.reindex(common_dates, method="ffill")
    weekly_sma_30 = sma30.reindex(common_dates, method="ffill")
    weekly_rsi_14 = rsi14.reindex(common_dates, method="ffill")

    # VWAP (daily data) - fully vectorized 2D operations
    typical_price = (high + low + close) / 3
    cum_vp = (typical_price * volume).rolling(20, min_periods=10).sum()
    cum_vol = volume.rolling(20, min_periods=10).sum()
    vwap_20 = cum_vp / cum_vol

    weekly_sma_10.columns = close.columns
    weekly_sma_30.columns = close.columns
    weekly_rsi_14.columns = close.columns
    vwap_20.columns = close.columns

    return weekly_sma_10, weekly_sma_30, weekly_rsi_14, vwap_20
