import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

from config import STOCK_FILTERING, SWING_PATTERNS

logger = logging.getLogger(__name__)


class TradeLogic:
    """Calculates entry/exit points and trade timing."""

    def analyze(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"symbol": symbol, "recommendation": "HOLD"}

        c = df["Close"].iloc[-1]
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        sma20, sma50 = ta.SMA(df["Close"], 20).iloc[-1], ta.SMA(df["Close"], 50).iloc[-1]
        rsi = ta.RSI(df["Close"], 14).iloc[-1]

        # Determine Recommendation
        rec = "HOLD"
        min_rsi = STOCK_FILTERING.get("require_rsi_above", 60.0)
        if c > sma20 > sma50 and rsi > min_rsi:
            rec = "BUY"

        # Trade Plan
        rules = SWING_PATTERNS.get("exit_rules", {})
        sl = c - (atr * rules.get("atr_stop_multiplier", 1.5))
        tp = c + (atr * rules.get("target_1_atr", 3.0))

        return {
            "symbol": symbol,
            "buy_price": round(c, 2),
            "sell_price": round(tp, 2),
            "stop_loss": round(sl, 2),
            "recommendation": rec,
            "confidence": round(0.7 if rec == "BUY" else 0.4, 2),
            "risk_reward_ratio": round((tp - c) / (c - sl), 2) if c > sl else 0,
        }
