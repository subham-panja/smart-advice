import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

logger = logging.getLogger(__name__)


class TradeLogic:
    """Calculates entry/exit points and trade timing."""

    def analyze(self, symbol: str, df: pd.DataFrame, app_config: Dict[str, Any]) -> Dict[str, Any]:
        if df.empty:
            return {"symbol": symbol, "recommendation": "HOLD"}

        c = df["Close"].iloc[-1]
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]

        # Determine Recommendation (Legacy logic, mainly for trade plan structure)
        rec = "HOLD"

        # Trade Plan
        exit_rules = app_config["exit_rules"]
        sl_multiplier = exit_rules["atr_stop_multiplier"]

        # Fetch T1 from targets (Required)
        targets = exit_rules["targets"]
        t1_mult = targets[0]["atr_multiplier"]

        sl = c - (atr * sl_multiplier)
        tp = c + (atr * t1_mult)

        return {
            "symbol": symbol,
            "buy_price": round(c, 2),
            "sell_price": round(tp, 2),
            "stop_loss": round(sl, 2),
            "recommendation": rec,
            "confidence": round(0.7 if rec == "BUY" else 0.4, 2),
            "risk_reward_ratio": round((tp - c) / (c - sl), 2) if c > sl else 0,
        }
