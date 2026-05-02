import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

logger = logging.getLogger(__name__)


class SwingTradingSignalAnalyzer:
    """Analyzes stocks for swing trading using Trend, Volatility, and Volume gates, driven by strategy config."""

    def __init__(self):
        pass

    def analyze_swing_opportunity(
        self, symbol: str, df: pd.DataFrame, strategy_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Basic Setup
        gates_cfg = strategy_config["swing_trading_gates"]
        t_cfg = gates_cfg["TREND_GATE"]["params"]
        v_cfg = gates_cfg["VOLATILITY_GATE"]["params"]
        vol_cfg = gates_cfg["VOLUME_GATE"]["params"]

        sma_p = t_cfg["sma_period"]
        if len(df) < max(sma_p, 100):
            return {
                "symbol": symbol,
                "all_gates_passed": False,
                "gates": {"trend": False, "volume": False, "volatility": False},
            }

        c = df["Close"].iloc[-1]

        # 1. GATE CHECKS (Hard Constraints)
        # -------------------------------
        # Trend Gate
        adx_series = ta.ADX(df["High"], df["Low"], df["Close"], 14)
        adx = adx_series.iloc[-1]
        pdi, mdi = (
            ta.PLUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
            ta.MINUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
        )
        sma = ta.SMA(df["Close"], sma_p).iloc[-1]

        trend_ok = adx > t_cfg["adx_min"] and pdi > mdi
        if t_cfg.get("require_price_above_sma"):
            trend_ok = trend_ok and c > sma

        # SMA Stack Check: 50 > 150 > 200 (Minervini/Weinstein requirement)
        if t_cfg.get("require_sma_stack"):
            sma50 = ta.SMA(df["Close"], 50).iloc[-1]
            sma150 = ta.SMA(df["Close"], 150).iloc[-1]
            sma200 = ta.SMA(df["Close"], 200).iloc[-1]
            trend_ok = trend_ok and (sma50 > sma150 > sma200)

        # Volume Gate
        v_mean = df["Volume"].tail(20).mean()
        v_latest = df["Volume"].iloc[-1]
        vol_ok = v_latest > v_mean * (1 + vol_cfg["zscore_threshold"])

        # Volatility Gate
        volatility_ok = True
        if gates_cfg["VOLATILITY_GATE"]["enabled"]:
            atr = ta.ATR(df["High"], df["Low"], df["Close"], 14)
            lb = v_cfg["lookback_days"]
            p_min, p_max = v_cfg["min_percentile"], v_cfg["max_percentile"]
            atr_recent = atr.iloc[-lb:].dropna()
            if len(atr_recent) > 0:
                current_atr = atr.iloc[-1]
                pctile = (atr_recent < current_atr).sum() / len(atr_recent) * 100
                volatility_ok = p_min <= pctile <= p_max

        gates = {"trend": trend_ok, "volume": vol_ok, "volatility": volatility_ok}
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

        # 2. INDICATOR SIGNALS (Hard & Bonus)
        # ----------------------------------
        signals = {}
        strat_cfg = strategy_config["strategy_config"]

        # MACD
        if strat_cfg["MACD_Signal_Crossover"]["enabled"]:
            macd, macdsignal, _ = ta.MACD(df["Close"], 12, 26, 9)
            signals["MACD"] = 1 if macd.iloc[-1] > macdsignal.iloc[-1] else 0

        # RSI
        if strat_cfg["RSI_Overbought_Oversold"]["enabled"]:
            rsi = ta.RSI(df["Close"], 14).iloc[-1]
            signals["RSI"] = 1 if rsi > 50 else 0

        # Bollinger Bands
        if strat_cfg["Bollinger_Band_Squeeze"]["enabled"]:
            upper, middle, lower = ta.BBANDS(df["Close"], 20, 2, 2)
            signals["BBANDS"] = 1 if c > middle.iloc[-1] else 0

        # ADX Strength
        if strat_cfg["ADX_Trend_Strength"]["enabled"]:
            signals["ADX_Strength"] = 1 if adx > strat_cfg["ADX_Trend_Strength"]["threshold"] else 0

        # Candlestick Patterns
        candle_pats = []
        if strat_cfg["Candlestick_Patterns"]["enabled"]:
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
                ema = ta.EMA(df["Close"], pat["ema_period"]).iloc[-1]
                rsi = ta.RSI(df["Close"], 14).iloc[-1]
                # Check if price is within 1% of EMA and RSI is in range
                is_near_ema = abs(c - ema) / ema < 0.01
                rsi_min, rsi_max = pat["rsi_range"]
                if is_near_ema and rsi_min <= rsi <= rsi_max:
                    entry_signals[pat_name] = 1

            elif pat_name == "bollinger_squeeze_breakout":
                upper, middle, lower = ta.BBANDS(df["Close"], pat["bb_period"], pat["bb_std"], pat["bb_std"])
                bandwidth = (upper - lower) / middle
                # Squeeze if bandwidth < threshold
                is_squeeze = bandwidth.iloc[-2] < pat["squeeze_threshold"]
                # Breakout if price crosses upper band today
                is_breakout = c > upper.iloc[-1]
                if is_squeeze and is_breakout:
                    entry_signals[pat_name] = 1

            elif pat_name == "macd_zero_cross":
                macd, _, _ = ta.MACD(df["Close"], pat["fast"], pat["slow"], pat["signal"])
                # Cross from below zero to above zero
                if macd.iloc[-2] < 0 and macd.iloc[-1] > 0:
                    entry_signals[pat_name] = 1

            elif pat_name == "higher_low_structure":
                # Simplified check for rising lows in the last X swings
                lows = df["Low"].rolling(window=pat["pivot_lookback"]).min().dropna()
                if len(lows) > 10:
                    recent_lows = lows.iloc[-10:].unique()
                    if len(recent_lows) >= pat["min_swings"]:
                        if recent_lows[-1] > recent_lows[-2]:
                            entry_signals[pat_name] = 1

            elif pat_name == "volatility_contraction":
                atr = ta.ATR(df["High"], df["Low"], df["Close"], 14)
                # Check if ATR is decreasing over the last 5 days
                if atr.iloc[-1] < atr.iloc[-5]:
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

        # Add entry patterns to signals
        signals.update(entry_signals)

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

            is_bonus = strat_cfg[config_key].get("is_bonus", False)
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
