import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import talib as ta

logger = logging.getLogger(__name__)


class SwingTradingSignalAnalyzer:
    """Analyzes stocks for swing trading using Trend, Volatility, and Volume gates, driven by strategy config."""

    def __init__(self):
        pass

    def _get_series(
        self,
        symbol: str,
        df: pd.DataFrame,
        indicator_store: Optional[Any],
        name: str,
        lookback: Optional[int] = None,
    ) -> pd.Series:
        """Get a historical series of an indicator, from store or TA-Lib."""
        if indicator_store is not None:
            return indicator_store.get_series(symbol, name, df.index[-1], lookback)
        return pd.Series(dtype=float)

    def analyze_swing_opportunity(
        self,
        symbol: str,
        df: pd.DataFrame,
        strategy_config: Dict[str, Any],
        indicator_store: Optional[Any] = None,
        market_breadth_ok: bool = True,
    ) -> Dict[str, Any]:
        # Basic Setup
        gates_cfg = strategy_config["swing_trading_gates"]
        t_cfg = gates_cfg.get("TREND_GATE", {}).get("params", {})
        v_cfg = gates_cfg.get("VOLATILITY_GATE", {}).get("params", {})
        vol_cfg = gates_cfg.get("VOLUME_GATE", {}).get("params", {})

        sma_p = t_cfg.get("sma_period", 50)
        min_data = max(sma_p, 100)
        if len(df) < min_data:
            return {
                "symbol": symbol,
                "all_gates_passed": False,
                "gates": {"trend": False, "volume": False, "volatility": False},
            }

        c = df["Close"].iloc[-1]

        # 1. GATE CHECKS (Hard Constraints)
        # -------------------------------
        # Trend Gate (always required if present)
        if indicator_store is not None:
            adx = indicator_store.get(symbol, "adx", df.index[-1])
            pdi = indicator_store.get(symbol, "pdi", df.index[-1])
            mdi = indicator_store.get(symbol, "mdi", df.index[-1])
            sma = indicator_store.get(symbol, "sma_50", df.index[-1])
        else:
            adx_series = ta.ADX(df["High"], df["Low"], df["Close"], 14)
            adx = adx_series.iloc[-1]
            pdi, mdi = (
                ta.PLUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
                ta.MINUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
            )
            sma = ta.SMA(df["Close"], sma_p).iloc[-1]

        trend_ok = adx > t_cfg["adx_min"]
        # DI alignment check (optional - some strategies like HMA retracement may not need it)
        if t_cfg.get("require_di_alignment", True):
            trend_ok = trend_ok and pdi > mdi
        if t_cfg.get("require_price_above_sma", True):
            trend_ok = trend_ok and c > sma

        # SMA Stack Check: 50 > 150 > 200 (Minervini/Weinstein requirement)
        if t_cfg.get("require_sma_stack"):
            if indicator_store is not None:
                sma50 = indicator_store.get(symbol, "sma_50", df.index[-1])
                sma150 = indicator_store.get(symbol, "sma_150", df.index[-1])
                sma200 = indicator_store.get(symbol, "sma_200", df.index[-1])
            else:
                sma50 = ta.SMA(df["Close"], 50).iloc[-1]
                sma150 = ta.SMA(df["Close"], 150).iloc[-1]
                sma200 = ta.SMA(df["Close"], 200).iloc[-1]
            trend_ok = trend_ok and (sma50 > sma150 > sma200)

        # Volume Gate (respect enabled flag)
        vol_ok = True  # Default pass if disabled
        if gates_cfg.get("VOLUME_GATE", {}).get("enabled", False):
            vol_cfg = gates_cfg["VOLUME_GATE"]["params"]
            v_mean = df["Volume"].tail(20).mean()
            v_latest = df["Volume"].iloc[-1]
            min_vol_ratio = vol_cfg.get("min_volume_ratio", 0.8)
            vol_ok = v_latest >= v_mean * min_vol_ratio

            # OBV trend check (accumulation detection)
            if "obv_trend_lookback" in vol_cfg:
                obv_lookback = vol_cfg["obv_trend_lookback"]
                if indicator_store is not None:
                    obv_recent = self._get_series(symbol, df, indicator_store, "obv", lookback=obv_lookback)
                else:
                    obv = pd.Series(0.0, index=df.index)
                    for i in range(1, len(df)):
                        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                            obv.iloc[i] = obv.iloc[i - 1] + df["Volume"].iloc[i]
                        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                            obv.iloc[i] = obv.iloc[i - 1] - df["Volume"].iloc[i]
                        else:
                            obv.iloc[i] = obv.iloc[i - 1]
                    obv_recent = obv.tail(obv_lookback).dropna()
                if len(obv_recent) >= 5:
                    x = np.arange(len(obv_recent))
                    slope = np.polyfit(x, obv_recent.values, 1)[0]
                    if vol_cfg.get("obv_required", True):
                        vol_ok = vol_ok and (slope > 0)

        # Volatility Gate - require ATR in range (not too low, not too high)
        volatility_ok = True
        if gates_cfg["VOLATILITY_GATE"]["enabled"]:
            lb = v_cfg["lookback_days"]
            max_pctile = v_cfg.get("max_percentile", 60)  # Upper bound: e.g., 60
            min_pctile = v_cfg.get("min_percentile", 20)  # Lower bound: e.g., 20
            if indicator_store is not None:
                atr_recent = self._get_series(symbol, df, indicator_store, "atr_14", lookback=lb)
                atr_recent = atr_recent.dropna()
                if len(atr_recent) > 0:
                    current_atr = atr_recent.iloc[-1]
            else:
                atr = ta.ATR(df["High"], df["Low"], df["Close"], 14)
                atr_recent = atr.iloc[-lb:].dropna()
                if len(atr_recent) > 0:
                    current_atr = atr.iloc[-1]
            if len(atr_recent) > 0:
                pctile = (atr_recent < current_atr).sum() / len(atr_recent) * 100
                # ATR must be between min and max percentile (enough momentum, not crash-level)
                volatility_ok = min_pctile <= pctile <= max_pctile

        gates = {"trend": trend_ok, "volume": vol_ok, "volatility": volatility_ok}

        # MTF_GATE - Multi-Timeframe Analysis (Weekly trend confirmation)
        if gates_cfg.get("MTF_GATE", {}).get("enabled", False):
            mtf_cfg = gates_cfg["MTF_GATE"]["params"]
            mtf_ok = True

            if mtf_cfg.get("weekly_trend_check", False):
                try:
                    if indicator_store is not None:
                        # Use pre-computed weekly indicators
                        sma_fast_val = indicator_store.get(symbol, "weekly_sma_10", df.index[-1])
                        sma_slow_val = indicator_store.get(symbol, "weekly_sma_30", df.index[-1])
                        weekly_rsi_val = indicator_store.get(symbol, "weekly_rsi_14", df.index[-1])
                        vwap_val = indicator_store.get(symbol, "vwap_20", df.index[-1])

                        if not np.isnan(sma_fast_val) and not np.isnan(sma_slow_val):
                            if sma_fast_val <= sma_slow_val:
                                mtf_ok = False
                        else:
                            mtf_ok = False

                        rsi_alignment_min = mtf_cfg.get("rsi_alignment_min", 60)
                        if np.isnan(weekly_rsi_val) or weekly_rsi_val < rsi_alignment_min:
                            mtf_ok = False

                        if mtf_cfg.get("vwap_confirmation", False):
                            if np.isnan(vwap_val) or df["Close"].iloc[-1] < vwap_val:
                                mtf_ok = False
                    else:
                        # Fetch weekly data for trend confirmation
                        weekly_data = (
                            df.resample("W")
                            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                            .dropna()
                        )

                        if len(weekly_data) >= 30:
                            # Weekly SMA crossover check
                            sma_fast = ta.SMA(weekly_data["Close"], mtf_cfg.get("weekly_sma_fast", 10))
                            sma_slow = ta.SMA(weekly_data["Close"], mtf_cfg.get("weekly_sma_slow", 30))

                            # Fast SMA should be above slow SMA (weekly uptrend)
                            if sma_fast.iloc[-1] <= sma_slow.iloc[-1]:
                                mtf_ok = False

                            # Weekly RSI alignment check
                            rsi_alignment_min = mtf_cfg.get("rsi_alignment_min", 60)
                            weekly_rsi = ta.RSI(weekly_data["Close"], 14)
                            if weekly_rsi.iloc[-1] < rsi_alignment_min:
                                mtf_ok = False

                            # VWAP confirmation - stock should be above VWAP (institutional accumulation)
                            if mtf_cfg.get("vwap_confirmation", False):
                                typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
                                cumulative_vp = (typical_price * df["Volume"]).rolling(20, min_periods=10).sum()
                                cumulative_vol = df["Volume"].rolling(20, min_periods=10).sum()
                                vwap = cumulative_vp / cumulative_vol
                                if not vwap.empty and df["Close"].iloc[-1] < vwap.iloc[-1]:
                                    mtf_ok = False
                except Exception as e:
                    logger.debug(f"MTF_GATE check failed for {symbol}: {e}")
                    mtf_ok = False  # Fail safe - if MTF check fails, gate fails

            if not mtf_ok:
                return {
                    "symbol": symbol,
                    "all_gates_passed": False,
                    "gates": {**gates, "mtf": False},
                    "reason": "Multi-timeframe analysis failed - weekly trend not confirmed",
                }

        all_gates_passed = all(gates.values())

        # 52-Week High Proximity Check (Minervini/Darvas/CANSLIM)
        proximity_pct = strategy_config["recommendation_thresholds"].get("proximity_to_52_week_high_pct", 100.0)
        if len(df) >= 252:
            high_52w = df["High"].tail(252).max()
            if high_52w > 0 and ((high_52w - c) / high_52w * 100) > proximity_pct:
                return {
                    "symbol": symbol,
                    "all_gates_passed": False,
                    "gates": gates,
                    "reason": f"Price > {proximity_pct}% below 52-week high",
                }

        if not all_gates_passed:
            return {"symbol": symbol, "all_gates_passed": False, "gates": gates, "reason": "Gates failed"}

        # Market Breadth Filter - block buys during broad market sell-offs
        if not market_breadth_ok:
            return {
                "symbol": symbol,
                "all_gates_passed": False,
                "gates": gates,
                "reason": "Market breadth weak - broad market sell-off detected",
            }

        # 2. INDICATOR SIGNALS (Hard & Bonus)
        # ----------------------------------
        signals = {}
        strat_cfg = strategy_config["strategy_config"]

        # MACD
        if strat_cfg.get("MACD_Signal_Crossover", {}).get("enabled", False):
            macd_cfg = strat_cfg["MACD_Signal_Crossover"]
            if indicator_store is not None:
                macd_val = indicator_store.get(symbol, "macd", df.index[-1])
                macdsignal_val = indicator_store.get(symbol, "macd_signal", df.index[-1])
            else:
                macd, macdsignal, histogram = ta.MACD(
                    df["Close"],
                    macd_cfg.get("fast_period", 12),
                    macd_cfg.get("slow_period", 26),
                    macd_cfg.get("signal_period", 9),
                )
                macd_val = macd.iloc[-1]
                macdsignal_val = macdsignal.iloc[-1]
            # For signal scanning, just check MACD > Signal line
            # Strict histogram checks (bars below zero, first green) are handled in entry patterns
            signals["MACD"] = 1 if macd_val > macdsignal_val else 0

        # RSI
        if strat_cfg.get("RSI_Overbought_Oversold", {}).get("enabled", False):
            if indicator_store is not None:
                rsi = indicator_store.get(symbol, "rsi_14", df.index[-1])
            else:
                rsi = ta.RSI(df["Close"], 14).iloc[-1]
            signals["RSI"] = 1 if rsi > 50 else 0

        # RSI Strength Cross (Nitin strategy)
        if strat_cfg.get("RSI_Strength_Cross", {}).get("enabled", False):
            rsi_cfg = strat_cfg["RSI_Strength_Cross"]
            rsi_period = rsi_cfg.get("rsi_period", 9)
            wma_period = rsi_cfg.get("slow_wma_period", 21)
            if indicator_store is not None:
                rsi_short = indicator_store.get(symbol, "rsi_9", df.index[-1])
                # WMA of RSI not pre-computed; compute from series
                rsi_short_series = self._get_series(symbol, df, indicator_store, "rsi_9")
                rsi_wma_val = (
                    rsi_short_series.rolling(wma_period, min_periods=1).mean().iloc[-1]
                    if len(rsi_short_series) > 0
                    else rsi_short
                )
            else:
                rsi_short = ta.RSI(df["Close"], rsi_period)
                rsi_wma = ta.WMA(rsi_short, wma_period)
                rsi_short = rsi_short.iloc[-1]
                rsi_wma_val = rsi_wma.iloc[-1]
            # RSI should be above its WMA (bullish momentum)
            signals["RSI_Strength"] = 1 if rsi_short > rsi_wma_val else 0

        # Bollinger Bands
        if strat_cfg.get("Bollinger_Band_Squeeze", {}).get("enabled", False):
            if indicator_store is not None:
                bb_middle_val = indicator_store.get(symbol, "bb_middle", df.index[-1])
            else:
                upper, middle, lower = ta.BBANDS(df["Close"], 20, 2, 2)
                bb_middle_val = middle.iloc[-1]
            signals["BBANDS"] = 1 if c > bb_middle_val else 0

        # ADX Strength
        if strat_cfg.get("ADX_Trend_Strength", {}).get("enabled", False):
            signals["ADX_Strength"] = 1 if adx > strat_cfg["ADX_Trend_Strength"]["threshold"] else 0

        # Candlestick Patterns
        candle_pats = []
        if strat_cfg.get("Candlestick_Patterns", {}).get("enabled", False):
            pat_cfg = strat_cfg["Candlestick_Patterns"]["patterns"]
            signals["Candlesticks"] = 0

            if pat_cfg.get("bullish_engulfing"):
                if ta.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1] == 100:
                    signals["Candlesticks"] = 1
                    candle_pats.append("Engulfing")

            if pat_cfg.get("hammer") and signals["Candlesticks"] == 0:
                if ta.CDLHAMMER(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1] == 100:
                    signals["Candlesticks"] = 1
                    candle_pats.append("Hammer")

            if pat_cfg.get("morning_star") and signals["Candlesticks"] == 0:
                if ta.CDLMORNINGSTAR(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1] == 100:
                    signals["Candlesticks"] = 1
                    candle_pats.append("Morning Star")

        # 3. ENTRY PATTERNS (One-of-many Triggers)
        # ---------------------------------------
        patterns_cfg = strategy_config.get("entry_patterns", [])
        entry_signals = {}

        for pat in patterns_cfg:
            if not pat.get("enabled"):
                continue

            pat_name = pat["name"]
            entry_signals[pat_name] = 0

            if pat_name == "pullback_to_ema":
                if indicator_store is not None:
                    ema = indicator_store.get(symbol, "ema_21", df.index[-1])
                    rsi = indicator_store.get(symbol, "rsi_14", df.index[-1])
                else:
                    ema = ta.EMA(df["Close"], pat["ema_period"]).iloc[-1]
                    rsi = ta.RSI(df["Close"], 14).iloc[-1]
                # Check if price is within configured distance of EMA and RSI is in range
                max_distance = pat.get("max_distance_pct", 3.0) / 100.0
                is_near_ema = abs(c - ema) / ema < max_distance
                rsi_min, rsi_max = pat["rsi_range"]

                # Check bullish candle requirement
                bullish_candle_ok = True
                if pat.get("bullish_candle_required", False):
                    bullish_candle_ok = df["Close"].iloc[-1] > df["Open"].iloc[-1]

                if is_near_ema and rsi_min <= rsi <= rsi_max and bullish_candle_ok:
                    entry_signals[pat_name] = 1

            elif pat_name == "bollinger_squeeze_breakout":
                if indicator_store is not None:
                    bb_upper_series = indicator_store.get_full_series(symbol, "bb_upper")
                    bb_middle_series = indicator_store.get_full_series(symbol, "bb_middle")
                    bb_lower_series = indicator_store.get_full_series(symbol, "bb_lower")
                    upper, middle, lower = bb_upper_series, bb_middle_series, bb_lower_series
                else:
                    upper, middle, lower = ta.BBANDS(df["Close"], pat["bb_period"], pat["bb_std"], pat["bb_std"])
                bandwidth = (upper - lower) / middle
                # Squeeze if bandwidth < threshold
                is_squeeze = bandwidth.iloc[-2] < pat["squeeze_threshold"]
                # Breakout if price crosses upper band today
                is_breakout = c > upper.iloc[-1]

                # Check retest requirement - if enabled, price must retest the breakout level
                retest_ok = True
                if pat.get("retest_required", False):
                    # Check if price pulled back to breakout level and bounced
                    if len(df) >= 5:
                        recent_lows = df["Low"].iloc[-5:].min()
                        retest_ok = recent_lows >= upper.iloc[-5]

                # Check squeeze duration - if too long, invalid setup
                squeeze_duration_ok = True
                max_duration = pat.get("max_squeeze_duration_days", 20)
                if len(df) >= max_duration:
                    # Count consecutive days of low volatility
                    recent_bandwidth = bandwidth.iloc[-max_duration:]
                    avg_bandwidth = recent_bandwidth.mean()
                    squeeze_duration_ok = avg_bandwidth < pat["squeeze_threshold"] * 1.5

                if is_squeeze and is_breakout and retest_ok and squeeze_duration_ok:
                    entry_signals[pat_name] = 1

            elif pat_name == "macd_zero_cross":
                if indicator_store is not None:
                    macd_line = indicator_store.get_full_series(symbol, "macd")
                    histogram = indicator_store.get_full_series(symbol, "macd_hist")
                else:
                    macd_line, _, histogram = ta.MACD(df["Close"], pat["fast"], pat["slow"], pat["signal"])
                # Cross from below zero to above zero
                cross_ok = macd_line.iloc[-2] < 0 and macd_line.iloc[-1] > 0

                # Above zero only - if enabled, MACD must already be positive
                if pat.get("above_zero_only", False):
                    cross_ok = cross_ok and macd_line.iloc[-1] > 0

                # Require histogram expansion - momentum must be increasing
                histogram_ok = True
                if pat.get("require_histogram_expansion", False):
                    if len(histogram) >= 3:
                        histogram_ok = histogram.iloc[-1] > histogram.iloc[-2] > histogram.iloc[-3]

                if cross_ok and histogram_ok:
                    entry_signals[pat_name] = 1

            elif pat_name == "higher_low_structure":
                # Check for higher lows structure (rising support)
                lows = df["Low"].rolling(window=pat["pivot_lookback"]).min().dropna()
                if len(lows) > 10:
                    recent_lows = lows.iloc[-10:].unique()
                    if len(recent_lows) >= pat["min_swings"]:
                        structure_ok = recent_lows[-1] > recent_lows[-2]

                        # Volume confirmation - if enabled, volume should support the structure
                        volume_ok = True
                        if pat.get("require_volume_confirmation", False):
                            vol_ma20 = df["Volume"].tail(20).mean()
                            volume_ok = df["Volume"].iloc[-1] >= vol_ma20 * 0.8

                        if structure_ok and volume_ok:
                            entry_signals[pat_name] = 1

            elif pat_name == "volatility_contraction":
                if indicator_store is not None:
                    atr_series = indicator_store.get_full_series(symbol, "atr_14")
                else:
                    atr_series = ta.ATR(df["High"], df["Low"], df["Close"], 14)

                # Check min contractions - ATR should be decreasing over multiple periods
                min_contractions = pat.get("min_contractions", 2)
                contraction_count = 0
                for i in range(1, min_contractions + 1):
                    if atr_series.iloc[-i] < atr_series.iloc[-(i + 1)]:
                        contraction_count += 1

                contraction_ok = contraction_count >= min_contractions

                # Volume dry-up - if enabled, volume should decrease with volatility
                volume_dry_ok = True
                if pat.get("volume_dry_up_required", False):
                    vol_ma20 = df["Volume"].tail(20).mean()
                    volume_dry_ok = df["Volume"].iloc[-1] < vol_ma20 * 0.8

                # Max ATR % of price - filter out extremely volatile stocks
                max_atr_pct = pat.get("max_atr_pct_of_price", 3.0) / 100.0
                atr_pct_ok = (atr_series.iloc[-1] / c) < max_atr_pct

                if contraction_ok and volume_dry_ok and atr_pct_ok:
                    entry_signals[pat_name] = 1

            elif pat_name == "twenty_day_high_breakout":
                high_20 = df["High"].iloc[-21:-1].max()
                if c > high_20:
                    if pat.get("volume_confirm"):
                        vol_ma20 = df["Volume"].tail(20).mean()
                        if df["Volume"].iloc[-1] >= vol_ma20 * pat.get("min_volume_multiplier", 1.5):
                            entry_signals[pat_name] = 1
                    else:
                        entry_signals[pat_name] = 1

            elif pat_name == "nr7_volatility_squeeze":
                lb = pat.get("lookback", 7)
                if len(df) >= lb + 1:
                    ranges = df["High"] - df["Low"]
                    current_range = ranges.iloc[-1]
                    min_range = ranges.tail(lb).min()
                    if current_range <= min_range * 1.001:
                        if pat.get("volume_dry_up"):
                            vol_ma20 = df["Volume"].tail(20).mean()
                            if df["Volume"].iloc[-1] < vol_ma20:
                                entry_signals[pat_name] = 1
                        else:
                            entry_signals[pat_name] = 1

            elif pat_name == "triple_confirm_hma_retracement":
                # Triple Confirm HMA Retracement Entry
                # 1. HMA(30) > HMA(44) - uptrend
                # 2. Price between or above HMA(30) and HMA(44) - retracement zone
                # 3. MACD histogram turning positive (momentum shift)
                # 4. RSI WMA cross above threshold (momentum confirmation)
                hma_fast_period = pat.get("hma_fast", 30)
                hma_slow_period = pat.get("hma_slow", 44)
                price_zone = pat.get("price_zone_requirement", "between_or_above_hma")

                if len(df) >= hma_slow_period + 10:
                    # Calculate HMA (Hull Moving Average)
                    def calc_hma(series, period):
                        half_period = period // 2
                        sqrt_period = int(period**0.5)
                        ema_half = ta.EMA(series, half_period)
                        ema_full = ta.EMA(series, period)
                        diff = 2 * ema_half - ema_full
                        return ta.EMA(diff, sqrt_period)

                    hma_fast = calc_hma(df["Close"], hma_fast_period)
                    hma_slow = calc_hma(df["Close"], hma_slow_period)

                    hma_f = hma_fast.iloc[-1]
                    hma_s = hma_slow.iloc[-1]

                    # 1. HMA alignment - fast above slow (uptrend)
                    if hma_f <= hma_s:
                        entry_signals[pat_name] = 0
                    else:
                        # 2. Price zone check
                        price_zone_ok = False
                        if price_zone == "between_or_above_hma":
                            price_zone_ok = hma_s <= c <= hma_f * 1.02 or c > hma_f
                        elif price_zone == "near_hma_fast":
                            price_zone_ok = abs(c - hma_f) / hma_f < 0.02

                        if not price_zone_ok:
                            entry_signals[pat_name] = 0
                        else:
                            # 3. MACD confirmation (if required)
                            macd_confirmed = True
                            if pat.get("macd_confirmed", False):
                                macd_line, signal_line, histogram = ta.MACD(df["Close"], 21, 39, 9)
                                # Histogram should be turning positive or already positive
                                macd_confirmed = histogram.iloc[-1] > 0 or (
                                    histogram.iloc[-1] > histogram.iloc[-2] and histogram.iloc[-2] < 0
                                )

                            if not macd_confirmed:
                                entry_signals[pat_name] = 0
                            else:
                                # 4. RSI WMA confirmation (if required)
                                rsi_wma_confirmed = True
                                if pat.get("rsi_wma_confirmed", False):
                                    rsi_9 = ta.RSI(df["Close"], 9)
                                    rsi_wma = ta.WMA(rsi_9, 21)
                                    # RSI should be above its WMA (momentum positive)
                                    rsi_wma_confirmed = rsi_9.iloc[-1] > rsi_wma.iloc[-1]

                                if rsi_wma_confirmed:
                                    entry_signals[pat_name] = 1

        # Add entry patterns to signals
        signals.update(entry_signals)

        # RSI Momentum Filter - require RSI > threshold for all entry patterns
        # This filters out dead stocks with no upward momentum
        if entry_signals and any(v == 1 for v in entry_signals.values()):
            if indicator_store is not None:
                rsi_14_series = indicator_store.get_full_series(symbol, "rsi_14")
            else:
                rsi_14_series = ta.RSI(df["Close"], 14)
            rsi_current = rsi_14_series.iloc[-1]
            rsi_momentum_min = strategy_config.get("rsi_momentum_filter", {}).get("min_rsi", 50)
            rsi_rising = strategy_config.get("rsi_momentum_filter", {}).get("require_rising", False)
            rsi_ok = rsi_current >= rsi_momentum_min
            if rsi_rising and len(rsi_14_series) >= 6:
                rsi_ok = rsi_ok and (rsi_current > rsi_14_series.iloc[-5])
            if not rsi_ok:
                for pat_name in entry_signals:
                    if entry_signals[pat_name] == 1:
                        entry_signals[pat_name] = 0  # Clear all triggered patterns

        # 4. APPLY BONUS/HARD LOGIC
        # -------------------------
        final_reasons = []
        # Check standard indicators
        for name, config_key in [
            ("MACD", "MACD_Signal_Crossover"),
            ("RSI", "RSI_Overbought_Oversold"),
            ("BBANDS", "Bollinger_Band_Squeeze"),
            ("ADX_Strength", "ADX_Trend_Strength"),
            ("Candlesticks", "Candlestick_Patterns"),
        ]:
            if name not in signals:
                continue

            is_bonus = strat_cfg.get(config_key, {}).get("is_bonus", False)
            if signals[name] == 0 and not is_bonus:
                return {
                    "symbol": symbol,
                    "all_gates_passed": False,
                    "gates": gates,
                    "reason": f"Hard requirement failed: {name}",
                }

            if signals[name] == 1:
                final_reasons.append(name if name != "Candlesticks" else f"Candle({', '.join(candle_pats)})")

        # Check if at least one entry pattern is triggered (Mandatory)
        active_entry_patterns = [p for p in entry_signals.values() if p == 1]
        if not active_entry_patterns and entry_signals:
            # Entry patterns are bonus signals - don't block if gates passed
            pass

        for p_name, p_val in entry_signals.items():
            if p_val == 1:
                final_reasons.append(f"Pattern({p_name})")

        # Calculate Technical Score
        all_signals = [s for s in signals.values()]
        technical_score = sum(all_signals) / len(all_signals) if all_signals else 0.0

        return {
            "symbol": symbol,
            "all_gates_passed": True,
            "gates": gates,
            "recommendation": "BUY"
            if technical_score >= strategy_config["recommendation_thresholds"]["technical_minimum"]
            else "HOLD",
            "technical_score": technical_score,
            "reason": " + ".join(final_reasons) if final_reasons else "Weak Signals",
        }
