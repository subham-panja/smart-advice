"""
Intraday Technical Indicators & Multi-Timeframe Confluence Module
================================================================
Calculates Central Pivot Range (CPR), Intraday VWAP, and Opening Range Breakout (ORB).
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


def calculate_cpr(
    prev_high: float,
    prev_low: float,
    prev_close: float,
    narrow_threshold_pct: float = 0.4,
) -> Dict[str, Union[float, bool]]:
    """
    Calculate Central Pivot Range (CPR) and standard floor pivots from previous day's OHLC.

    Formula:
        Pivot = (High + Low + Close) / 3
        Bottom Central (BC) = (High + Low) / 2
        Top Central (TC) = (Pivot - BC) + Pivot
        R1 = 2 * Pivot - Low,  S1 = 2 * Pivot - High
        R2 = Pivot + (High - Low), S2 = Pivot - (High - Low)
    """
    if prev_high <= 0 or prev_low <= 0 or prev_close <= 0:
        raise ValueError("Prices must be positive numbers")

    pivot = (prev_high + prev_low + prev_close) / 3.0
    bc_raw = (prev_high + prev_low) / 2.0
    tc_raw = (pivot - bc_raw) + pivot

    tc = max(tc_raw, bc_raw)
    bc = min(tc_raw, bc_raw)

    cpr_width_pct = (abs(tc - bc) / pivot) * 100.0 if pivot > 0 else 0.0
    is_narrow = cpr_width_pct <= narrow_threshold_pct

    r1 = (2.0 * pivot) - prev_low
    s1 = (2.0 * pivot) - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)

    return {
        "pivot": round(pivot, 2),
        "tc": round(tc, 2),
        "bc": round(bc, 2),
        "cpr_width_pct": round(cpr_width_pct, 3),
        "is_narrow": bool(is_narrow),
        "r1": round(r1, 2),
        "s1": round(s1, 2),
        "r2": round(r2, 2),
        "s2": round(s2, 2),
    }


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate intraday Volume-Weighted Average Price (VWAP).
    Resets at the start of each trading session if multiple days are present.
    """
    if df.empty:
        return pd.Series(dtype=float)

    high = df["High"] if "High" in df.columns else df["Close"]
    low = df["Low"] if "Low" in df.columns else df["Close"]
    close = df["Close"]
    volume = df["Volume"]

    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume

    # If datetime index with multiple days, group by date for intraday reset
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index.normalize().unique()) > 1:
        dates = df.index.normalize()
        cum_tp_vol = tp_vol.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
    else:
        cum_tp_vol = tp_vol.cumsum()
        cum_vol = volume.cumsum()

    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
    return vwap.ffill().bfill()


def calculate_orb(
    df: pd.DataFrame,
    start_time_str: str = "09:15",
    end_time_str: str = "09:30",
) -> Dict[str, float]:
    """
    Extract the Opening Range (ORB) High and Low during the opening session.
    Default: first 15 minutes of the trading day (09:15 to 09:30 IST).
    """
    if df.empty:
        return {"orb_high": 0.0, "orb_low": 0.0, "orb_range": 0.0}

    # If datetime index, slice to opening window
    if isinstance(df.index, pd.DatetimeIndex):
        mask = (df.index.strftime("%H:%M") >= start_time_str) & (df.index.strftime("%H:%M") <= end_time_str)
        orb_bars = df.loc[mask]
    else:
        # Fallback for mock/test data with fewer bars (first 3 bars of 5m = 15m)
        orb_bars = df.head(3)

    if orb_bars.empty:
        orb_bars = df.head(3)

    orb_high = float(orb_bars["High"].max()) if "High" in orb_bars.columns else float(orb_bars["Close"].max())
    orb_low = float(orb_bars["Low"].min()) if "Low" in orb_bars.columns else float(orb_bars["Close"].min())

    return {
        "orb_high": round(orb_high, 2),
        "orb_low": round(orb_low, 2),
        "orb_range": round(orb_high - orb_low, 2),
    }


def evaluate_intraday_confluence(
    symbol: str,
    df_5m: pd.DataFrame,
    prev_day_ohlc: Dict[str, float],
    current_time_str: str = "10:00",
    config: Optional[dict] = None,
) -> Dict[str, any]:
    """
    Evaluates institutional confluence (CPR + VWAP + ORB) on 5-minute bars.
    Returns trade recommendation, setup classification, entry/stop-loss/target, and gate checklist.
    """
    if df_5m.empty:
        return {"symbol": symbol, "signal": "HOLD", "reason": "No data"}

    cfg = config or {}
    gates_cfg = cfg.get("intraday_gates", {})

    prev_high = prev_day_ohlc.get("High", 0.0)
    prev_low = prev_day_ohlc.get("Low", 0.0)
    prev_close = prev_day_ohlc.get("Close", 0.0)

    cpr = calculate_cpr(prev_high, prev_low, prev_close)
    vwap_series = calculate_vwap(df_5m)
    current_vwap = float(vwap_series.iloc[-1]) if not vwap_series.empty else 0.0
    orb = calculate_orb(df_5m)

    last_bar = df_5m.iloc[-1]
    current_price = float(last_bar["Close"])
    current_volume = float(last_bar.get("Volume", 0))
    avg_volume = float(df_5m["Volume"].rolling(10, min_periods=1).mean().iloc[-1]) if "Volume" in df_5m.columns else 1.0

    # 1. Market Hours Gate
    market_hours_ok = "09:30" <= current_time_str <= "14:45"
    if not market_hours_ok and gates_cfg.get("MARKET_HOURS_GATE", {}).get("enabled", True):
        return {
            "symbol": symbol,
            "signal": "HOLD",
            "reason": f"Outside active trading window ({current_time_str})",
            "all_gates_passed": False,
            "gates": {"market_hours": False, "cpr": False, "vwap": False, "orb": False},
        }

    # 2. CPR Gate
    cpr_ok = current_price >= cpr["tc"]  # Price cleanly above Top Central Pivot for Long

    # 3. VWAP Gate
    vwap_ok = current_price >= current_vwap  # Price above institutional volume anchor

    # 4. ORB Gate
    orb_ok = current_price > orb["orb_high"]  # Broke above 15m opening range high

    # Volume confirmation
    vol_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0
    vol_surge_ok = vol_ratio >= 1.25

    gates = {
        "market_hours": market_hours_ok,
        "cpr": cpr_ok,
        "vwap": vwap_ok,
        "orb": orb_ok,
    }

    all_gates_passed = market_hours_ok and cpr_ok and vwap_ok and orb_ok

    # Calculate 5-minute ATR for dynamic stop placement
    if len(df_5m) >= 5:
        tr1 = df_5m["High"] - df_5m["Low"]
        tr2 = (df_5m["High"] - df_5m["Close"].shift(1)).abs()
        tr3 = (df_5m["Low"] - df_5m["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_5m = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
    else:
        atr_5m = max(current_price * 0.005, 1.0)

    # Risk parameters
    stop_loss = max(round(current_price - (atr_5m * 1.2), 2), round(orb["orb_low"], 2))
    target_1 = round(current_price + (atr_5m * 1.8), 2)
    target_2 = round(current_price + (atr_5m * 3.0), 2)
    risk = max(current_price - stop_loss, 0.01)
    reward_t1 = target_1 - current_price
    rr_ratio = round(reward_t1 / risk, 2)

    signal = "BUY" if (all_gates_passed and vol_surge_ok) else "HOLD"
    pattern = "orb_15m_breakout" if orb_ok else ("vwap_pullback_rejection" if vwap_ok else "cpr_breakout_and_retest")

    return {
        "symbol": symbol,
        "signal": signal,
        "pattern": pattern,
        "all_gates_passed": all_gates_passed,
        "gates": gates,
        "current_price": round(current_price, 2),
        "current_vwap": round(current_vwap, 2),
        "cpr": cpr,
        "orb": orb,
        "atr_5m": round(atr_5m, 2),
        "stop_loss": stop_loss,
        "target_1": target_1,
        "target_2": target_2,
        "rr_ratio": rr_ratio,
        "volume_ratio": round(vol_ratio, 2),
        "reason": f"Pattern({pattern}) | CPR(tc={cpr['tc']}) | VWAP({round(current_vwap, 2)}) | ORB({orb['orb_high']})",
    }
