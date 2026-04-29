import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

from config import SWING_PATTERNS, SWING_TRADING_GATES

logger = logging.getLogger(__name__)


class SwingTradingSignalAnalyzer:
    """Analyzes stocks for swing trading using Trend, Volatility, and Volume gates."""

    def __init__(self):
        self.t_cfg = SWING_TRADING_GATES.get("TREND_GATE", {}).get("params", {})
        self.v_cfg = SWING_TRADING_GATES.get("VOLATILITY_GATE", {}).get("params", {})
        self.vol_cfg = SWING_TRADING_GATES.get("VOLUME_GATE", {}).get("params", {})

    def analyze_swing_opportunity(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        sma_p = self.t_cfg.get("sma_period", 200)
        if len(df) < sma_p:
            return {"symbol": symbol, "all_gates_passed": False}

        c = df["Close"].iloc[-1]

        # Trend Gate
        adx = ta.ADX(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        pdi, mdi = (
            ta.PLUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
            ta.MINUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
        )
        sma = ta.SMA(df["Close"], sma_p).iloc[-1]
        trend_ok = adx > self.t_cfg.get("adx_min", 20) and c > sma and pdi > mdi

        # Volume Gate
        v_mean = df["Volume"].tail(20).mean()
        vol_ok = df["Volume"].iloc[-1] > v_mean * (1 + self.vol_cfg.get("zscore_threshold", 0.2))

        # Volatility Gate
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14)
        lb = self.v_cfg.get("lookback_days", 100)
        p_min, p_max = self.v_cfg.get("min_percentile", 20), self.v_cfg.get("max_percentile", 80)
        v_ok = p_min <= (atr.iloc[-lb:] < atr.iloc[-1]).sum() <= p_max

        gates = {"trend": trend_ok, "volume": vol_ok, "volatility": v_ok}
        all_ok = all(gates.values())

        res = {"symbol": symbol, "all_gates_passed": all_ok, "gates": gates}
        if all_ok:
            p_cfg = next((p for p in SWING_PATTERNS.get("entry_patterns", []) if p["name"] == "pullback_to_ema"), {})
            if p_cfg.get("enabled"):
                ema = ta.EMA(df["Close"], p_cfg.get("ema_period", 20)).iloc[-1]
                if df["Low"].iloc[-1] <= ema * 1.02 and c > ema:
                    res.update({"recommendation": "BUY", "pattern": "EMA_Pullback"})

        return res
