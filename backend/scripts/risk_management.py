import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

from config import TRADING_OPTIONS

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles stop-loss, position sizing, and risk-reward evaluation."""

    def __init__(self, account_balance: float = None):
        self.balance = account_balance if account_balance is not None else TRADING_OPTIONS["initial_capital"]

    def calculate_risk_params(self, df: pd.DataFrame, entry: float, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculates optimal stop loss, position size, and targets based on global and strategy config."""
        from config import RISK_MANAGEMENT

        sizing_cfg = RISK_MANAGEMENT["position_sizing"]
        exit_rules = app_config["exit_rules"]
        rec_thresholds = app_config["recommendation_thresholds"]

        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]

        # 1. Stop Loss Calculation (Strict)
        sl_multiplier = exit_rules["atr_stop_multiplier"]
        sl = entry - (atr * sl_multiplier)
        risk_per_share = entry - sl

        if risk_per_share <= 0:
            return {"position_size": 0, "risk_reward_ok": False}

        # 2. Position Sizing based on Risk Amount
        risk_per_trade_pct = sizing_cfg["risk_per_trade"]
        risk_amt = self.balance * risk_per_trade_pct
        size_based_on_risk = int(risk_amt / risk_per_share)

        # 3. Position Sizing based on Capital Cap
        max_pos_pct = sizing_cfg["max_position_pct"]
        max_capital_allowed = self.balance * max_pos_pct
        size_based_on_capital = int(max_capital_allowed / entry)

        # Final Size: Smallest of Risk-Limit or Capital-Limit
        size = min(size_based_on_risk, size_based_on_capital)

        # 4. Targets (Strictly from List)
        target_list = exit_rules["targets"]
        targets = {
            "T1": entry + (atr * target_list[0]["atr_multiplier"]),
            "T2": entry + (atr * target_list[1]["atr_multiplier"]),
        }

        # Validate Risk Reward
        min_rr = rec_thresholds["min_risk_reward_ratio"]
        reward = targets["T1"] - entry
        rr_ratio = reward / risk_per_share
        rr_ok = rr_ratio >= min_rr

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

    def calculate_stop_loss(
        self, df: pd.DataFrame, current_price: float, method="atr", atr_multiplier=1.5
    ) -> Dict[str, Any]:
        """Calculates a trailing stop loss based on ATR."""
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        sl = current_price - (atr * atr_multiplier)
        return {"stop_loss": round(sl, 2)}
