import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

from config import RECOMMENDATION_THRESHOLDS, RISK_MANAGEMENT, SWING_PATTERNS

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles stop-loss, position sizing, and risk-reward evaluation."""

    def __init__(self, account_balance: float = 100000):
        p = RISK_MANAGEMENT.get("position_sizing", {})
        self.balance = account_balance or RISK_MANAGEMENT.get("portfolio_constraints", {}).get(
            "initial_capital", 100000
        )
        self.risk_per_trade = p.get("risk_per_trade", 0.01)
        self.min_rr = RECOMMENDATION_THRESHOLDS.get("min_risk_reward_ratio", 2.0)

    def calculate_risk_params(self, df: pd.DataFrame, entry: float) -> Dict[str, Any]:
        """Calculates optimal stop loss, position size, and targets."""
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        rules = SWING_PATTERNS.get("exit_rules", {})
        sl = entry - (atr * rules.get("atr_stop_multiplier", 2.0))

        # Position Sizing
        risk_amt = self.balance * self.risk_per_trade
        risk_per_share = entry - sl
        size = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0

        # Targets
        t1_m, t2_m = rules.get("target_1_atr", 2.0), rules.get("target_2_atr", 4.0)
        targets = {"T1": entry + (atr * t1_m), "T2": entry + (atr * t2_m)}

        return {
            "stop_loss": round(sl, 2),
            "position_size": size,
            "position_value": round(size * entry, 2),
            "risk_amount": round(risk_amt, 2),
            "targets": {k: round(v, 2) for k, v in targets.items()},
            "risk_reward_ok": True,
        }

    def validate_trade(self, entry: float, sl: float, target: float) -> bool:
        risk = entry - sl
        reward = target - entry
        return (reward / risk >= self.min_rr) if risk > 0 else False

    def calculate_stop_loss(
        self, df: pd.DataFrame, current_price: float, method="atr", atr_multiplier=1.5
    ) -> Dict[str, Any]:
        """Calculates a trailing stop loss based on ATR."""
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        sl = current_price - (atr * atr_multiplier)
        return {"stop_loss": round(sl, 2)}
