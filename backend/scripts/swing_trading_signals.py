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
        gates_cfg = strategy_config["swing_trading_gates"]
        t_cfg = gates_cfg["TREND_GATE"]["params"]
        v_cfg = gates_cfg["VOLATILITY_GATE"]["params"]
        vol_cfg = gates_cfg["VOLUME_GATE"]["params"]

        sma_p = t_cfg["sma_period"]
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
        require_above_sma = t_cfg["require_price_above_sma"]
        trend_ok = adx > t_cfg["adx_min"] and pdi > mdi
        if require_above_sma:
            trend_ok = trend_ok and c > sma

        # Slope Check
        if trend_ok and t_cfg["adx_slope_check"]:
            trend_ok = adx > adx_prev

        # Volume Gate
        is_ep_mode = "EP" in strategy_config["name"].upper()
        v_mean = df["Volume"].tail(20).mean()
        v_latest = df["Volume"].iloc[-1]

        # EP MODE Catalyst check
        has_recent_spike = False
        if is_ep_mode:
            # Look for the multiplier in stock_filters or volume_analysis_config
            vol_spike_cfg = next(
                (f for f in strategy_config.get("stock_filters", []) if f["type"] == "volume_spike_lookup"), {}
            )
            spike_mult = vol_spike_cfg.get("multiplier", 2.0)
            lookback = vol_spike_cfg.get("lookback_days", 5)
            ma_p = vol_spike_cfg.get("ma_period", 50)

            recent_v = df["Volume"].tail(lookback + 1).iloc[:-1]
            baseline_v = df["Volume"].rolling(window=ma_p).mean().tail(lookback + 1).iloc[:-1]
            has_recent_spike = any(recent_v > baseline_v * spike_mult)

        # Standard logic OR (EP Mode + Recent Spike)
        vol_ok = v_latest > v_mean * (1 + vol_cfg["zscore_threshold"])
        if is_ep_mode and has_recent_spike:
            vol_ok = True

        # Volatility Gate
        atr = ta.ATR(df["High"], df["Low"], df["Close"], 14)
        lb = v_cfg["lookback_days"]
        p_min, p_max = v_cfg["min_percentile"], v_cfg["max_percentile"]

        volatility_ok = True
        if gates_cfg["VOLATILITY_GATE"]["enabled"]:
            volatility_ok = p_min <= (atr.iloc[-lb:] < atr.iloc[-1]).sum() <= p_max

        gates = {"trend": trend_ok, "volume": vol_ok, "volatility": volatility_ok}
        all_ok = all(gates.values())

        res = {"symbol": symbol, "all_gates_passed": all_ok, "gates": gates}

        if all_ok:
            entry_patterns = strategy_config["entry_patterns"]
            p_cfg = next((p for p in entry_patterns if p["name"] == "pullback_to_ema"), {"enabled": False})

            if p_cfg["enabled"]:
                ema_p = p_cfg["ema_period"]
                ema = ta.EMA(df["Close"], ema_p).iloc[-1]
                if df["Low"].iloc[-1] <= ema * 1.03 and c > ema * 0.98:
                    res.update({"recommendation": "BUY", "pattern": "EMA_Pullback"})

        # Construct Analysis Reason
        reasons = []
        if gates["trend"]:
            reasons.append("Trend OK")
        if gates["volume"]:
            reasons.append("Volume Spike")
        if gates["volatility"]:
            reasons.append("Volatility Squeeze")
        if res.get("pattern"):
            reasons.append(f"Pattern: {res['pattern']}")

        res["reason"] = " + ".join(reasons) if all_ok else "Gates failed"
        return res
