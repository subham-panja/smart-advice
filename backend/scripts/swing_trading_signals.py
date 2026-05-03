"""
Swing Trading Signal Generator
==============================

Analyzes stocks for swing trading using configurable Trend, Volatility, and Volume gates.
Driven entirely by strategy JSON config — no hardcoded thresholds.

Supports:
- Multi-gate checks (TREND, VOLATILITY, VOLUME) with enable/disable flags
- VCP (Volatility Contraction Pattern) with full Minervini parameters
- Entry patterns: pullback_to_ema, bollinger_squeeze, macd_cross, higher_low,
  volatility_contraction, 20_day_high_breakout, nr7_squeeze
- RSI momentum filter for entry quality
- Bonus vs hard indicator logic
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd
import talib as ta

logger = logging.getLogger(__name__)


class SwingTradingSignalAnalyzer:
    """Analyzes stocks for swing trading using gates and entry patterns from strategy config."""

    def analyze_swing_opportunity(
        self, symbol: str, df: pd.DataFrame, strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run full swing analysis: gates → indicators → entry patterns → score."""
        gates_cfg = strategy_config.get("swing_trading_gates", {})

        # Get configurable indicator periods
        indicator_periods = strategy_config.get("indicator_periods", {})
        adx_period = indicator_periods.get("adx", 14)
        rsi_period = indicator_periods.get("rsi", 14)
        atr_period = indicator_periods.get("atr", 14)
        bb_period = indicator_periods.get("bb", 20)
        macd_fast = indicator_periods.get("macd_fast", 12)
        macd_slow = indicator_periods.get("macd_slow", 26)
        macd_signal_p = indicator_periods.get("macd_signal", 9)

        # Determine required data length
        max_required = 252  # 52-week high needs 252 days
        for gate_name in ["TREND_GATE", "VOLATILITY_GATE", "VOLUME_GATE"]:
            gate_params = gates_cfg.get(gate_name, {}).get("params", {})
            if "sma_period" in gate_params:
                max_required = max(max_required, gate_params["sma_period"])
            if "lookback_days" in gate_params:
                max_required = max(max_required, gate_params["lookback_days"])

        if len(df) < max_required:
            return {
                "symbol": symbol,
                "all_gates_passed": False,
                "gates": {"trend": False, "volume": False, "volatility": False},
                "reason": f"Insufficient data: {len(df)} < {max_required}",
            }

        c = df["Close"].iloc[-1]

        # ========== 1. GATE CHECKS (Hard Constraints) ==========

        # --- TREND GATE ---
        trend_cfg = gates_cfg.get("TREND_GATE", {})
        trend_params = trend_cfg.get("params", {})
        trend_enabled = trend_cfg.get("enabled", True)

        adx_series = ta.ADX(df["High"], df["Low"], df["Close"], adx_period)
        adx = adx_series.iloc[-1]
        pdi = ta.PLUS_DI(df["High"], df["Low"], df["Close"], adx_period).iloc[-1]
        mdi = ta.MINUS_DI(df["High"], df["Low"], df["Close"], adx_period).iloc[-1]
        sma_period = trend_params.get("sma_period", 50)
        sma = ta.SMA(df["Close"], sma_period).iloc[-1]

        trend_ok = True
        if trend_enabled:
            adx_min = trend_params.get("adx_min", 20)
            adx_max = trend_params.get("adx_max", 80)
            trend_ok = adx_min <= adx <= adx_max

            if trend_params.get("require_di_plus_above_di_minus"):
                min_di_diff = trend_params.get("min_di_diff", 0)
                trend_ok = trend_ok and (pdi - mdi >= min_di_diff)

            if trend_params.get("require_price_above_sma"):
                trend_ok = trend_ok and c > sma

            if trend_params.get("require_sma_stack"):
                sma50 = ta.SMA(df["Close"], 50).iloc[-1]
                sma150 = ta.SMA(df["Close"], 150).iloc[-1]
                sma200 = ta.SMA(df["Close"], 200).iloc[-1]
                trend_ok = trend_ok and (sma50 > sma150 > sma200)

                # Check that SMA200 is rising (Minervini requirement)
                if len(df) >= 220:
                    sma200_prev = ta.SMA(df["Close"].iloc[:-20], 200).iloc[-1]
                    trend_ok = trend_ok and (sma200 > sma200_prev)

        # --- VOLUME GATE ---
        vol_cfg = gates_cfg.get("VOLUME_GATE", {})
        vol_params = vol_cfg.get("params", {})
        vol_enabled = vol_cfg.get("enabled", True)

        v_mean = df["Volume"].tail(20).mean()
        v_latest = df["Volume"].iloc[-1]
        min_vol_ratio = vol_params.get("min_volume_ratio", 0.8)
        vol_ok = True
        if vol_enabled:
            vol_ok = v_latest >= v_mean * min_vol_ratio

            # OBV trend check (accumulation detection)
            if "obv_trend_lookback" in vol_params:
                obv_lookback = vol_params["obv_trend_lookback"]
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
                    logic_op = vol_params.get("logic_operator", "AND")
                    if logic_op == "AND":
                        vol_ok = vol_ok and (slope > 0)
                    # If OR, vol_ok stays true if volume ratio passes

        # --- VOLATILITY GATE ---
        volat_cfg = gates_cfg.get("VOLATILITY_GATE", {})
        volat_params = volat_cfg.get("params", {})
        volat_enabled = volat_cfg.get("enabled", True)

        volatility_ok = True
        if volat_enabled:
            atr = ta.ATR(df["High"], df["Low"], df["Close"], atr_period)
            lb = volat_params.get("lookback_days", 100)
            max_pctile = volat_params.get("max_percentile", 60)
            min_pctile = volat_params.get("min_percentile", 20)
            atr_recent = atr.iloc[-lb:].dropna()
            if len(atr_recent) > 0:
                current_atr = atr.iloc[-1]
                pctile = (atr_recent < current_atr).sum() / len(atr_recent) * 100
                # ATR must be between min and max percentile
                volatility_ok = min_pctile <= pctile <= max_pctile

        gates = {"trend": trend_ok, "volume": vol_ok, "volatility": volatility_ok}

        # Check if all gates must pass (require_all_gates flag)
        require_all_gates = strategy_config.get("recommendation_thresholds", {}).get("require_all_gates", True)
        if require_all_gates:
            all_gates_passed = all(gates.values())
        else:
            # At least trend gate must pass
            all_gates_passed = trend_ok

        # --- 52-WEEK HIGH PROXIMITY (Minervini/Darvas/CANSLIM) ---
        proximity_pct = strategy_config.get("recommendation_thresholds", {}).get("proximity_to_52_week_high_pct", 25.0)
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
            failed = [k for k, v in gates.items() if not v]
            return {
                "symbol": symbol,
                "all_gates_passed": False,
                "gates": gates,
                "reason": f"Gates failed: {', '.join(failed)}",
            }

        # ========== 2. INDICATOR SIGNALS (Hard & Bonus) ==========
        signals = {}
        strat_cfg = strategy_config.get("strategy_config", {})

        # MACD
        if strat_cfg.get("MACD_Signal_Crossover", {}).get("enabled"):
            macd, macdsignal, _ = ta.MACD(df["Close"], macd_fast, macd_slow, macd_signal_p)
            signals["MACD"] = 1 if macd.iloc[-1] > macdsignal.iloc[-1] else 0

        # RSI
        if strat_cfg.get("RSI_Overbought_Oversold", {}).get("enabled"):
            rsi = ta.RSI(df["Close"], rsi_period).iloc[-1]
            overbought = strat_cfg["RSI_Overbought_Oversold"].get("overbought", 70)
            oversold = strat_cfg["RSI_Overbought_Oversold"].get("oversold", 30)
            # Signal if RSI is in bullish zone (not overbought, above oversold)
            signals["RSI"] = 1 if oversold < rsi < overbought else 0

        # Bollinger Bands
        if strat_cfg.get("Bollinger_Band_Squeeze", {}).get("enabled"):
            bb_period_cfg = strat_cfg["Bollinger_Band_Squeeze"].get("period", bb_period)
            bb_std = strat_cfg["Bollinger_Band_Squeeze"].get("std", 2.0)
            upper, middle, lower = ta.BBANDS(df["Close"], bb_period_cfg, bb_std, bb_std)
            signals["BBANDS"] = 1 if c > middle.iloc[-1] else 0

        # ADX Strength
        if strat_cfg.get("ADX_Trend_Strength", {}).get("enabled"):
            adx_threshold = strat_cfg["ADX_Trend_Strength"].get("threshold", 20)
            signals["ADX_Strength"] = 1 if adx > adx_threshold else 0

        # Volume Breakout
        if strat_cfg.get("Volume_Breakout", {}).get("enabled"):
            vol_mult = strat_cfg["Volume_Breakout"].get("volume_multiplier", 1.5)
            vol_lookback = strat_cfg["Volume_Breakout"].get("lookback", 20)
            vol_ma = df["Volume"].tail(vol_lookback).mean()
            signals["Volume_Breakout"] = 1 if v_latest >= vol_ma * vol_mult else 0

        # Candlestick Patterns
        candle_pats = []
        if strat_cfg.get("Candlestick_Patterns", {}).get("enabled"):
            pat_cfg = strat_cfg["Candlestick_Patterns"].get("patterns", {})
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

        # Relative Strength (already computed elsewhere, but mark as passed if enabled)
        if strat_cfg.get("Relative_Strength_Comparison", {}).get("enabled"):
            signals["RS"] = 1  # RS is pre-filtered in stock selection

        # OBV (bonus if enabled)
        if strat_cfg.get("On_Balance_Volume", {}).get("enabled"):
            signals["OBV"] = 1 if vol_ok else 0  # OBV already checked in volume gate

        # ========== 3. ENTRY PATTERNS ==========
        patterns_cfg = strategy_config.get("entry_patterns", [])
        entry_signals = {}

        for pat in patterns_cfg:
            if not pat.get("enabled"):
                continue

            pat_name = pat["name"]
            entry_signals[pat_name] = 0

            if pat_name == "pullback_to_ema":
                ema_period = pat.get("ema_period", 21)
                ema = ta.EMA(df["Close"], ema_period).iloc[-1]
                rsi = ta.RSI(df["Close"], rsi_period).iloc[-1]
                max_dist_pct = pat.get("max_distance_from_ema_pct", 1.5) / 100.0
                is_near_ema = abs(c - ema) / ema < max_dist_pct
                rsi_range = pat.get("rsi_range", [40, 60])
                rsi_min, rsi_max = rsi_range
                if is_near_ema and rsi_min <= rsi <= rsi_max:
                    entry_signals[pat_name] = 1

            elif pat_name == "bollinger_squeeze_breakout":
                bb_p = pat.get("bb_period", bb_period)
                bb_std = pat.get("bb_std", 2.0)
                squeeze_thresh = pat.get("squeeze_threshold", 0.10)
                upper, middle, lower = ta.BBANDS(df["Close"], bb_p, bb_std, bb_std)
                bandwidth = (upper - lower) / middle
                is_squeeze = bandwidth.iloc[-2] < squeeze_thresh
                is_breakout = c > upper.iloc[-1]

                if pat.get("require_volume_confirm", False):
                    vol_mult = pat.get("min_volume_multiplier", 1.5)
                    vol_ma = df["Volume"].tail(20).mean()
                    is_breakout = is_breakout and v_latest >= vol_ma * vol_mult

                if is_squeeze and is_breakout:
                    entry_signals[pat_name] = 1

            elif pat_name == "macd_zero_cross":
                macd, _, _ = ta.MACD(df["Close"], pat.get("fast", 12), pat.get("slow", 26), pat.get("signal", 9))
                if macd.iloc[-2] < 0 and macd.iloc[-1] > 0:
                    entry_signals[pat_name] = 1

            elif pat_name == "higher_low_structure":
                lookback = pat.get("pivot_lookback", 5)
                min_swings = pat.get("min_swings", 3)
                lows = df["Low"].rolling(window=lookback).min().dropna()
                if len(lows) > 10:
                    recent_lows = lows.iloc[-10:].unique()
                    if len(recent_lows) >= min_swings:
                        if recent_lows[-1] > recent_lows[-2]:
                            entry_signals[pat_name] = 1

            elif pat_name == "volatility_contraction":
                # Full VCP pattern detection using all configured parameters
                atr = ta.ATR(df["High"], df["Low"], df["Close"], atr_period)

                # 1. ATR must be decreasing over configured days
                atr_decrease_days = pat.get("atr_decrease_days", 5)
                if len(atr) < atr_decrease_days + 1:
                    continue
                atr_decreasing = atr.iloc[-1] < atr.iloc[-atr_decrease_days]

                # 2. ATR as % of price must be within limit
                max_atr_pct = pat.get("max_atr_pct_of_price", 4.0)
                current_atr_pct = (atr.iloc[-1] / c) * 100
                atr_pct_ok = current_atr_pct <= max_atr_pct

                # 3. Check for multiple contractions (VCP shape)
                min_contractions = pat.get("min_contractions", 2)
                max_contractions = pat.get("max_contractions", 5)
                max_depth_pct = pat.get("max_contraction_depth_pct", 25.0)
                min_duration = pat.get("min_contraction_duration_days", 3)
                max_duration = pat.get("max_contraction_duration_days", 90)

                # Count contractions by finding local highs and measuring pullbacks
                close_series = df["Close"].tail(max_duration)
                contractions = 0

                if len(close_series) >= min_duration * min_contractions:
                    # Find rolling highs and measure subsequent pullbacks
                    window = min_duration
                    highs = close_series.rolling(window=window).max()
                    for i in range(window * 2, len(close_series)):
                        if highs.iloc[i] > highs.iloc[i - 1] and highs.iloc[i - 1] == highs.iloc[i - 2]:
                            # Peak detected, measure pullback
                            peak = highs.iloc[i]
                            subsequent_lows = close_series.iloc[i : i + max_duration]
                            if len(subsequent_lows) > 0:
                                pullback_pct = ((peak - subsequent_lows.min()) / peak) * 100
                                if pullback_pct <= max_depth_pct:
                                    contractions += 1

                    contractions_ok = min_contractions <= contractions <= max_contractions
                else:
                    contractions_ok = False

                # 4. Volume dry-up during contractions (if required)
                volume_dry_up = pat.get("volume_dry_up_required", True)
                vol_dry_ok = True
                if volume_dry_up:
                    vol_ma = df["Volume"].tail(20).mean()
                    recent_vol = df["Volume"].tail(5).mean()
                    vol_dry_ok = recent_vol < vol_ma * 0.8

                # 5. Require VCP shape (if configured)
                require_vcp = pat.get("require_vcp_shape", True)
                if require_vcp:
                    vcp_ok = atr_decreasing and atr_pct_ok and contractions_ok and vol_dry_ok
                else:
                    vcp_ok = atr_decreasing and atr_pct_ok

                if vcp_ok:
                    entry_signals[pat_name] = 1

            elif pat_name == "twenty_day_high_breakout":
                high_20 = df["High"].iloc[-21:-1].max()
                if c > high_20:
                    if pat.get("volume_confirm"):
                        vol_ma20 = df["Volume"].tail(20).mean()
                        vol_mult = pat.get("min_volume_multiplier", 1.5)
                        if df["Volume"].iloc[-1] >= vol_ma20 * vol_mult:
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

        # ========== 4. RSI MOMENTUM FILTER ==========
        # Filters out dead stocks with no upward momentum at entry
        rsi_filter_cfg = strategy_config.get("rsi_momentum_filter", {})
        rsi_filter_enabled = rsi_filter_cfg.get("enabled", True)

        if rsi_filter_enabled and entry_signals and any(v == 1 for v in entry_signals.values()):
            rsi_series = ta.RSI(df["Close"], rsi_period)
            rsi_current = rsi_series.iloc[-1]
            rsi_momentum_min = rsi_filter_cfg.get("min_rsi", 50)
            rsi_rising = rsi_filter_cfg.get("require_rising", False)
            rsi_ok = rsi_current >= rsi_momentum_min
            if rsi_rising and len(rsi_series) >= 6:
                rsi_ok = rsi_ok and (rsi_current > rsi_series.iloc[-5])
            if not rsi_ok:
                for pat_name in entry_signals:
                    if entry_signals[pat_name] == 1:
                        entry_signals[pat_name] = 0  # Clear all triggered patterns

        # Add entry patterns to signals dict
        signals.update(entry_signals)

        # ========== 5. APPLY BONUS/HARD LOGIC ==========
        final_reasons = []
        indicator_config_map = [
            ("MACD", "MACD_Signal_Crossover"),
            ("RSI", "RSI_Overbought_Oversold"),
            ("BBANDS", "Bollinger_Band_Squeeze"),
            ("ADX_Strength", "ADX_Trend_Strength"),
            ("Volume_Breakout", "Volume_Breakout"),
            ("Candlesticks", "Candlestick_Patterns"),
        ]

        for name, config_key in indicator_config_map:
            if name not in signals:
                continue

            cfg = strat_cfg.get(config_key, {})
            is_bonus = cfg.get("is_bonus", False)
            if signals[name] == 0 and not is_bonus:
                return {
                    "symbol": symbol,
                    "all_gates_passed": False,
                    "gates": gates,
                    "reason": f"Hard requirement failed: {name}",
                }

            if signals[name] == 1:
                if name == "Candlesticks" and candle_pats:
                    final_reasons.append(f"Candle({', '.join(candle_pats)})")
                else:
                    final_reasons.append(name)

        # Check if at least one entry pattern is triggered
        for p_name, p_val in entry_signals.items():
            if p_val == 1:
                final_reasons.append(f"Pattern({p_name})")

        # ========== 6. CALCULATE TECHNICAL SCORE ==========
        all_signals = [s for s in signals.values()]
        technical_score = sum(all_signals) / len(all_signals) if all_signals else 0.0

        rec_thresholds = strategy_config.get("recommendation_thresholds", {})
        tech_min = rec_thresholds.get("technical_minimum", 0.35)

        return {
            "symbol": symbol,
            "all_gates_passed": True,
            "gates": gates,
            "recommendation": "BUY" if technical_score >= tech_min else "HOLD",
            "technical_score": technical_score,
            "reason": " + ".join(final_reasons) if final_reasons else "Weak Signals",
        }
