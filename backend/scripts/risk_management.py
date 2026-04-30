import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

from config import TRADING_OPTIONS

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles stop-loss, position sizing, and risk-reward evaluation."""

    def __init__(self, account_balance: float = None):
        self.balance = account_balance or TRADING_OPTIONS.get("initial_capital", 100000.0)

    def calculate_risk_params(
        self, df: pd.DataFrame, entry: float, app_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Calculates optimal stop loss, position size, and targets based strictly on config."""
        if not app_config:
            return {"position_size": 0, "risk_reward_ok": False}

        risk_mgmt = app_config.get("risk_management", {})
        sizing_cfg = risk_mgmt.get("position_sizing", {})
        exit_rules = app_config.get("exit_rules", {})
        rec_thresholds = app_config.get("recommendation_thresholds", {})

        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]

        # 1. Stop Loss Calculation (Config Driven)
        sl_multiplier = exit_rules.get("atr_stop_multiplier", 1.5)
        sl = entry - (atr * sl_multiplier)
        risk_per_share = entry - sl

        if risk_per_share <= 0:
            return {"position_size": 0, "risk_reward_ok": False}

        # 2. Position Sizing based on Risk Amount (e.g. 1% of Config Balance)
        risk_per_trade_pct = sizing_cfg.get("risk_per_trade", 0.01)
        risk_amt = self.balance * risk_per_trade_pct
        size_based_on_risk = int(risk_amt / risk_per_share)

        # 3. Position Sizing based on Capital Cap (e.g. 10% of Config Balance)
        max_pos_pct = sizing_cfg.get("max_position_pct", 0.10)
        max_capital_allowed = self.balance * max_pos_pct
        size_based_on_capital = int(max_capital_allowed / entry)

        # Final Size: Smallest of Risk-Limit or Capital-Limit
        size = min(size_based_on_risk, size_based_on_capital)

        # 4. Targets (Config Driven)
        target_list = exit_rules.get("targets", [])
        targets = {}
        if len(target_list) >= 1:
            targets["T1"] = entry + (atr * target_list[0].get("atr_multiplier", 2.0))
        else:
            targets["T1"] = entry + (atr * 2.0)

        if len(target_list) >= 2:
            targets["T2"] = entry + (atr * target_list[1].get("atr_multiplier", 4.0))
        else:
            targets["T2"] = entry + (atr * 4.0)

        # Validate Risk Reward
        min_rr = rec_thresholds.get("min_risk_reward_ratio", 2.0)
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
