import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

from config import RECOMMENDATION_THRESHOLDS, RISK_MANAGEMENT, SWING_PATTERNS, TRADING_OPTIONS

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles stop-loss, position sizing, and risk-reward evaluation."""

    def __init__(self, account_balance: float = None):
        # Strict Config Source of Truth
        self.sizing_cfg = RISK_MANAGEMENT.get("position_sizing", {})
        self.exit_cfg = SWING_PATTERNS.get("exit_rules", {})

        # Pull values directly from config with no hardcoded local defaults
        self.balance = account_balance or TRADING_OPTIONS.get("initial_capital")
        self.risk_per_trade_pct = self.sizing_cfg.get("risk_per_trade")
        self.max_pos_pct = self.sizing_cfg.get("max_position_pct")
        self.min_rr = RECOMMENDATION_THRESHOLDS.get("min_risk_reward_ratio")

    def calculate_risk_params(self, df: pd.DataFrame, entry: float) -> Dict[str, Any]:
        """Calculates optimal stop loss, position size, and targets based strictly on config."""
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]

        # 1. Stop Loss Calculation (Config Driven)
        sl_multiplier = self.exit_cfg.get("atr_stop_multiplier")
        sl = entry - (atr * sl_multiplier)
        risk_per_share = entry - sl

        if risk_per_share <= 0:
            return {"position_size": 0, "risk_reward_ok": False}

        # 2. Position Sizing based on Risk Amount (1% of Config Balance)
        risk_amt = self.balance * self.risk_per_trade_pct
        size_based_on_risk = int(risk_amt / risk_per_share)

        # 3. Position Sizing based on Capital Cap (10% of Config Balance)
        max_capital_allowed = self.balance * self.max_pos_pct
        size_based_on_capital = int(max_capital_allowed / entry)

        # Final Size: Smallest of Risk-Limit or Capital-Limit
        size = min(size_based_on_risk, size_based_on_capital)

        # 4. Targets (Config Driven)
        t1_m = self.exit_cfg.get("target_1_atr")
        t2_m = self.exit_cfg.get("target_2_atr")
        targets = {"T1": entry + (atr * t1_m), "T2": entry + (atr * t2_m)}

        # Validate Risk Reward
        reward = targets["T1"] - entry
        rr_ratio = reward / risk_per_share
        rr_ok = rr_ratio >= self.min_rr

        return {
            "stop_loss": round(sl, 2),
            "position_size": size,
            "position_value": round(size * entry, 2),
            "risk_amount": round(size * risk_per_share, 2),
            "allocation_pct": round(((size * entry) / self.balance) * 100, 2),
            "targets": {k: round(v, 2) for k, v in targets.items()},
            "risk_reward_ok": rr_ok,
            "rr_ratio": round(rr_ratio, 2),
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
