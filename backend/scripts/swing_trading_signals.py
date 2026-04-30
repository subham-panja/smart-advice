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
        self, symbol: str, df: pd.DataFrame, strategy_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        if not strategy_config:
            return {"symbol": symbol, "all_gates_passed": False}

        gates_cfg = strategy_config.get("swing_trading_gates", {})
        t_cfg = gates_cfg.get("TREND_GATE", {}).get("params", {})
        v_cfg = gates_cfg.get("VOLATILITY_GATE", {}).get("params", {})
        vol_cfg = gates_cfg.get("VOLUME_GATE", {}).get("params", {})

        sma_p = t_cfg.get("sma_period", 200)
        if len(df) < max(sma_p, 100):
            return {"symbol": symbol, "all_gates_passed": False}

        c = df["Close"].iloc[-1]

        # Trend Gate
        adx_series = ta.ADX(df["High"], df["Low"], df["Close"], 14)
        adx = adx_series.iloc[-1]
        adx_prev = adx_series.iloc[-2]

        pdi, mdi = (
            ta.PLUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
            ta.MINUS_DI(df["High"], df["Low"], df["Close"], 14).iloc[-1],
        )
        sma = ta.SMA(df["Close"], sma_p).iloc[-1]

        # Primary Trend Logic
        # Note: require_price_above_sma check added for flexibility
        require_above_sma = t_cfg.get("require_price_above_sma", True)
        trend_ok = adx > t_cfg.get("adx_min", 15) and pdi > mdi
        if require_above_sma:
            trend_ok = trend_ok and c > sma

        # Slope Check: If enabled, ADX must be rising to confirm momentum is building
        if trend_ok and t_cfg.get("adx_slope_check", True):
            trend_ok = adx > adx_prev

        # Volume Gate
        is_ep_mode = "EP" in strategy_config.get("name", "").upper()
        v_mean = df["Volume"].tail(20).mean()
        v_latest = df["Volume"].iloc[-1]

        # EP MODE: Check for massive spike in last 5 days
        has_recent_spike = False
        if is_ep_mode:
            # Look for volume spike in any of the last 5 days (excluding today)
            recent_v = df["Volume"].tail(6).iloc[:-1]
            baseline_v = df["Volume"].rolling(window=50).mean().tail(6).iloc[:-1]
            # Use multiplier from JSON if available, else default to 3.0
            spike_mult = 3.0
            has_recent_spike = any(recent_v > baseline_v * spike_mult)

        # Standard logic OR (EP Mode + Recent Spike)
        vol_ok = v_latest > v_mean * (1 + vol_cfg.get("zscore_threshold", 0.2))
        if is_ep_mode and has_recent_spike:
            # Patel/Bonde Rule: Buy the dry-up after the spike
            vol_ok = True

        # Volatility Gate
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14)
        lb = v_cfg.get("lookback_days", 100)
        p_min, p_max = v_cfg.get("min_percentile", 20), v_cfg.get("max_percentile", 80)

        # Avoid gate if disabled
        volatility_ok = True
        if gates_cfg.get("VOLATILITY_GATE", {}).get("enabled", False):
            volatility_ok = p_min <= (atr.iloc[-lb:] < atr.iloc[-1]).sum() <= p_max

        gates = {"trend": trend_ok, "volume": vol_ok, "volatility": volatility_ok}
        all_ok = all(gates.values())

        res = {"symbol": symbol, "all_gates_passed": all_ok, "gates": gates}
        if all_ok:
            entry_patterns = strategy_config.get("entry_patterns", [])
            p_cfg = next((p for p in entry_patterns if p["name"] == "pullback_to_ema"), {})
            if p_cfg.get("enabled", True):
                ema_p = p_cfg.get("ema_period", 10)
                ema = ta.EMA(df["Close"], ema_p).iloc[-1]
                # Catch explosive momentum on EMA
                if df["Low"].iloc[-1] <= ema * 1.03 and c > ema * 0.98:
                    res.update({"recommendation": "BUY", "pattern": "EMA_Pullback"})

        return res
