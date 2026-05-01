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
            volatility_ok = p_min <= (atr.iloc[-lb:] < atr.iloc[-1]).sum() <= p_max

        gates = {"trend": trend_ok, "volume": vol_ok, "volatility": volatility_ok}
        all_gates_passed = all(gates.values())

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

        # 3. APPLY BONUS/HARD LOGIC
        # -------------------------
        final_reasons = []
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

        # Calculate Technical Score
        enabled_signals = [s for s in signals.values()]
        technical_score = sum(enabled_signals) / len(enabled_signals) if enabled_signals else 0.0

        return {
            "symbol": symbol,
            "all_gates_passed": True,
            "gates": gates,  # Restored for compatibility
            "recommendation": "BUY" if technical_score >= 0.5 else "HOLD",
            "technical_score": technical_score,
            "reason": " + ".join(final_reasons) if final_reasons else "Weak Signals",
        }
