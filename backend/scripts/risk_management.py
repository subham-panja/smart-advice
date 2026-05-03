import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

# Config now comes from strategy JSON

logger = logging.getLogger(__name__)


class RiskManager:
    """Handles stop-loss, position sizing, and risk-reward evaluation."""

    def __init__(self, account_balance: float = None):
        # Default balance; actual value comes from strategy trading_config.initial_capital
        self.balance = account_balance if account_balance is not None else 100000.0

    def _get_strategy_risk_config(self, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk config from strategy config, falling back to global defaults."""
        # Global defaults (strategy config takes precedence)
        global_sizing = {"risk_per_trade": 0.01, "max_position_pct": 0.10}
        strategy_risk = app_config.get("risk_management", {})

        return {
            "risk_per_trade": strategy_risk.get("risk_per_trade_pct", global_sizing.get("risk_per_trade", 0.01)),
            "max_risk_per_trade": strategy_risk.get(
                "max_risk_per_trade_pct",
                strategy_risk.get("risk_per_trade_pct", global_sizing.get("risk_per_trade", 0.01)),
            ),
            "max_position_pct": strategy_risk.get("max_position_pct", global_sizing.get("max_position_pct", 0.20)),
            "volatility_scaling": strategy_risk.get("volatility_scaling", {}),
        }

    def _calculate_volatility_scaled_risk(
        self, df: pd.DataFrame, atr: float, max_risk_pct: float, vol_scale_cfg: Dict[str, Any]
    ) -> float:
        """Scale risk_per_trade_pct based on current ATR vs historical ATR percentile.

        When volatility is low (low ATR percentile) → use full max_risk_pct.
        When volatility is high (high ATR percentile) → reduce risk proportionally.

        Returns the scaled risk_per_trade_pct.
        """
        if not vol_scale_cfg.get("enabled", False):
            return max_risk_pct

        lookback = vol_scale_cfg.get("atr_lookback_days", 100)
        low_pctile = vol_scale_cfg.get("low_volatility_percentile", 25)
        high_pctile = vol_scale_cfg.get("high_volatility_percentile", 75)
        min_multiplier = vol_scale_cfg.get("min_risk_multiplier", 0.6)

        if len(df) < lookback + 14:
            return max_risk_pct

        atr_series = ta.ATR(df["High"], df["Low"], df["Close"], 14)
        atr_history = atr_series.dropna().iloc[-lookback:]

        if len(atr_history) < lookback or atr_history.max() == atr_history.min():
            return max_risk_pct

        current_percentile = (atr_history <= atr).mean() * 100

        if current_percentile <= low_pctile:
            scaled_risk = max_risk_pct
        elif current_percentile >= high_pctile:
            scaled_risk = max_risk_pct * min_multiplier
        else:
            pct_range = high_pctile - low_pctile
            fraction = (current_percentile - low_pctile) / pct_range if pct_range > 0 else 0
            scaled_risk = max_risk_pct * (1.0 - fraction * (1.0 - min_multiplier))

        return round(scaled_risk, 4)

    def calculate_risk_params(
        self, df: pd.DataFrame, entry: float, app_config: Dict[str, Any], regime_status: str = "BULL"
    ) -> Dict[str, Any]:
        """Calculates optimal stop loss, position size, and targets based on global and strategy config."""
        exit_rules = app_config["exit_rules"]
        rec_thresholds = app_config["recommendation_thresholds"]
        risk_cfg = app_config.get("risk_management", {})

        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]

        # 1. Stop Loss Calculation - support different stop loss types
        stop_loss_type = risk_cfg.get("stop_loss_type", "ATR")

        if stop_loss_type == "ATR":
            sl_multiplier = exit_rules["atr_stop_multiplier"]
            sl = entry - (atr * sl_multiplier)
        elif stop_loss_type == "PERCENTAGE":
            # Use regime-adaptive percentage stop
            regime_risk_cfg = risk_cfg.get("regime_adaptive_risk", {})
            regime_key = regime_status.lower() if regime_status != "UNKNOWN" else "bull"
            if regime_key in regime_risk_cfg:
                stop_pct = regime_risk_cfg[regime_key].get("stop_loss_pct", 8.0) / 100.0
            else:
                stop_pct = exit_rules.get("oneil_stop_loss_pct", 8.0) / 100.0
            sl = entry * (1 - stop_pct)
        elif stop_loss_type == "SUPPORT":
            # Use recent swing low as stop
            swing_low = df["Low"].tail(20).min()
            sl = swing_low * 0.98  # 2% buffer below swing low
        else:
            # Default to ATR
            sl_multiplier = exit_rules["atr_stop_multiplier"]
            sl = entry - (atr * sl_multiplier)

        risk_per_share = entry - sl

        if risk_per_share <= 0:
            return {"position_size": 0, "risk_reward_ok": False}

        # 2. Position Sizing based on Risk Amount
        risk_cfg_extract = self._get_strategy_risk_config(app_config)
        vol_scale_cfg = risk_cfg_extract["volatility_scaling"]

        # Support different position sizing methods
        position_sizing_method = risk_cfg.get("position_sizing", "risk_based")

        if position_sizing_method == "risk_based":
            max_risk_pct = risk_cfg_extract["max_risk_per_trade"]
            effective_risk_pct = self._calculate_volatility_scaled_risk(df, atr, max_risk_pct, vol_scale_cfg)

            # Regime-adaptive risk (from strategy config)
            regime_risk_cfg = app_config.get("risk_management", {}).get("regime_adaptive_risk", {})
            regime_key = regime_status.lower() if regime_status != "UNKNOWN" else "bull"
            if regime_key in regime_risk_cfg:
                regime_risk = regime_risk_cfg[regime_key]
                effective_risk_pct = regime_risk.get("risk_per_trade_pct", effective_risk_pct) / 100.0

            risk_amt = self.balance * effective_risk_pct
            size_based_on_risk = int(risk_amt / risk_per_share)
            size = size_based_on_risk
        elif position_sizing_method == "fixed_pct":
            # Fixed percentage of capital per trade
            fixed_pct = risk_cfg.get("risk_per_trade_pct", 2.0) / 100.0
            allocation = self.balance * fixed_pct
            size = int(allocation / entry)
        elif position_sizing_method == "volatility_weighted":
            # Weight position size by inverse of volatility
            vol_scale_cfg = risk_cfg.get("volatility_scaling", {})
            max_risk_pct = risk_cfg_extract["max_risk_per_trade"]
            effective_risk_pct = self._calculate_volatility_scaled_risk(df, atr, max_risk_pct, vol_scale_cfg)
            risk_amt = self.balance * effective_risk_pct
            size = int(risk_amt / risk_per_share)
        else:
            # Default to risk_based
            max_risk_pct = risk_cfg_extract["max_risk_per_trade"]
            effective_risk_pct = self._calculate_volatility_scaled_risk(df, atr, max_risk_pct, vol_scale_cfg)
            risk_amt = self.balance * effective_risk_pct
            size = int(risk_amt / risk_per_share)

        # 3. Position Sizing based on Capital Cap
        max_pos_pct = risk_cfg_extract["max_position_pct"] / 100.0  # Convert from percentage (10.0) to decimal (0.10)
        max_capital_allowed = self.balance * max_pos_pct
        size_based_on_capital = int(max_capital_allowed / entry)

        # Final Size: Smallest of Risk-Limit or Capital-Limit
        size = min(size, size_based_on_capital)

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
            "effective_risk_pct": round(effective_risk_pct * 100, 2),
        }

    def calculate_stop_loss(
        self, df: pd.DataFrame, current_price: float, method="atr", atr_multiplier=1.5
    ) -> Dict[str, Any]:
        """Calculates a trailing stop loss based on ATR."""
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14).iloc[-1]
        sl = current_price - (atr * atr_multiplier)
        return {"stop_loss": round(sl, 2)}
