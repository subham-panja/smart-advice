"""
Vectorbt-based batch signal generation.

Replaces the per-symbol per-date nested Python loops in _compute_signals_worker
with fully vectorized boolean array operations on pre-computed indicators.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from .vectorbt_indicator_batch import ComputedIndicators


def compute_signal_matrix(
    indicators: ComputedIndicators,
    strategy_config: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute BUY/HOLD signal for all (date, symbol) pairs simultaneously.

    Args:
        indicators: Pre-computed indicators from compute_all_indicators
        strategy_config: Full strategy config dict

    Returns:
        (pass_matrix, score_matrix) where:
        - pass_matrix: bool DataFrame (dates x symbols), True if all gates pass + BUY
        - score_matrix: float DataFrame (dates x symbols), technical_score for BUY signals
    """
    gates_cfg = strategy_config.get("swing_trading_gates", {})
    strat_cfg = strategy_config.get("strategy_config", {})
    entry_patterns = strategy_config.get("entry_patterns", [])
    thresholds = strategy_config.get("recommendation_thresholds", {})

    c = indicators.close
    h = indicators.high
    low = indicators.low
    v = indicators.volume
    o = indicators.open_price

    symbols = indicators.symbols
    dates = indicators.dates

    # Initialize result matrices
    pass_matrix = pd.DataFrame(False, index=dates, columns=symbols, dtype=bool)
    score_matrix = pd.DataFrame(0.0, index=dates, columns=symbols, dtype=float)

    # --- TREND GATE ---
    t_cfg = gates_cfg.get("TREND_GATE", {}).get("params", {})
    sma_p = t_cfg.get("sma_period", 50)

    adx = indicators.adx
    pdi = indicators.pdi
    mdi = indicators.mdi

    # Use pre-computed SMA for trend gate (sma_p defaults to 50)
    if sma_p == 50:
        sma_trend = indicators.sma_50
    elif sma_p == 150:
        sma_trend = indicators.sma_150
    elif sma_p == 200:
        sma_trend = indicators.sma_200
    else:
        # For non-standard periods, compute using vbt
        import vectorbt as vbt

        sma_trend = vbt.talib("SMA").run(c, timeperiod=sma_p).real
        sma_trend.columns = symbols

    trend_ok = adx > t_cfg.get("adx_min", 15)
    if t_cfg.get("require_di_alignment", True):
        trend_ok = trend_ok & (pdi > mdi)
    if t_cfg.get("require_price_above_sma", True):
        trend_ok = trend_ok & (c > sma_trend)

    # SMA Stack: 50 > 150 > 200
    if t_cfg.get("require_sma_stack", False):
        trend_ok = trend_ok & (indicators.sma_50 > indicators.sma_150) & (indicators.sma_150 > indicators.sma_200)

    # --- VOLUME GATE ---
    vol_ok = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    if gates_cfg.get("VOLUME_GATE", {}).get("enabled", False):
        vol_cfg = gates_cfg["VOLUME_GATE"]["params"]
        v_mean = v.rolling(20).mean()
        min_vol_ratio = vol_cfg.get("min_volume_ratio", 0.8)
        vol_ok = v >= v_mean * min_vol_ratio

        # OBV trend check
        if "obv_trend_lookback" in vol_cfg:
            obv_lookback = vol_cfg["obv_trend_lookback"]
            obv = indicators.obv
            # Compute rolling slope of OBV (simplified: use rolling mean direction)
            obv_chg = obv.pct_change()
            obv_trend = obv_chg.rolling(obv_lookback).mean() > 0
            if vol_cfg.get("obv_required", True):
                vol_ok = vol_ok & obv_trend

    # --- VOLATILITY GATE ---
    vol_gate_ok = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    if gates_cfg.get("VOLATILITY_GATE", {}).get("enabled", False):
        v_cfg = gates_cfg["VOLATILITY_GATE"]["params"]
        lb = v_cfg.get("lookback_days", 100)
        min_pctile = v_cfg.get("min_percentile", 5)
        max_pctile = v_cfg.get("max_percentile", 70)

        atr = indicators.atr_14
        atr_rank = atr.rolling(lb).rank(pct=True) * 100
        vol_gate_ok = (atr_rank >= min_pctile) & (atr_rank <= max_pctile)

    # --- MTF GATE ---
    mtf_ok = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    if gates_cfg.get("MTF_GATE", {}).get("enabled", False):
        mtf_cfg = gates_cfg["MTF_GATE"]["params"]
        if mtf_cfg.get("weekly_trend_check", False):
            if indicators.weekly_sma_10 is not None and indicators.weekly_sma_30 is not None:
                mtf_ok = mtf_ok & (indicators.weekly_sma_10 > indicators.weekly_sma_30)

            if indicators.weekly_rsi_14 is not None:
                rsi_min = mtf_cfg.get("rsi_alignment_min", 60)
                mtf_ok = mtf_ok & (indicators.weekly_rsi_14 >= rsi_min)

            if mtf_cfg.get("vwap_confirmation", False) and indicators.vwap_20 is not None:
                mtf_ok = mtf_ok & (c > indicators.vwap_20)

    # --- 52-WEEK HIGH PROXIMITY ---
    proximity_ok = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)
    proximity_pct = thresholds.get("proximity_to_52_week_high_pct", 100.0)
    high_52w = c.rolling(252).max()
    proximity_ok = (high_52w > 0) & (((high_52w - c) / high_52w * 100) <= proximity_pct)

    # --- COMBINED GATES ---
    all_gates = trend_ok & vol_ok & vol_gate_ok & mtf_ok & proximity_ok

    # --- INDICATOR SIGNALS ---
    signal_count = pd.DataFrame(0, index=dates, columns=symbols, dtype=int)
    signal_hits = pd.DataFrame(0, index=dates, columns=symbols, dtype=int)

    # MACD
    if strat_cfg.get("MACD_Signal_Crossover", {}).get("enabled", False):
        macd_hit = indicators.macd > indicators.macd_signal
        is_bonus = strat_cfg["MACD_Signal_Crossover"].get("is_bonus", False)
        signal_count += 1
        signal_hits += macd_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & macd_hit

    # RSI
    if strat_cfg.get("RSI_Overbought_Oversold", {}).get("enabled", False):
        rsi_hit = indicators.rsi_14 > 50
        is_bonus = strat_cfg["RSI_Overbought_Oversold"].get("is_bonus", False)
        signal_count += 1
        signal_hits += rsi_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & rsi_hit

    # BBANDS
    if strat_cfg.get("Bollinger_Band_Squeeze", {}).get("enabled", False):
        bb_hit = c > indicators.bb_middle
        is_bonus = strat_cfg["Bollinger_Band_Squeeze"].get("is_bonus", False)
        signal_count += 1
        signal_hits += bb_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & bb_hit

    # ADX Strength
    if strat_cfg.get("ADX_Trend_Strength", {}).get("enabled", False):
        adx_hit = indicators.adx > strat_cfg["ADX_Trend_Strength"]["threshold"]
        is_bonus = strat_cfg["ADX_Trend_Strength"].get("is_bonus", False)
        signal_count += 1
        signal_hits += adx_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & adx_hit

    # --- ENTRY PATTERNS ---
    for pat in entry_patterns:
        if not pat.get("enabled"):
            continue

        pat_name = pat["name"]
        signal_count += 1

        if pat_name == "pullback_to_ema":
            ema_distance = (c - indicators.ema_21).abs() / indicators.ema_21
            max_dist = pat.get("max_distance_pct", 3.0) / 100.0
            rsi_min, rsi_max = pat["rsi_range"]
            rsi_in_range = (indicators.rsi_14 >= rsi_min) & (indicators.rsi_14 <= rsi_max)
            bullish = c > o if pat.get("bullish_candle_required", False) else True
            hit = (ema_distance < max_dist) & rsi_in_range & bullish

        elif pat_name == "bollinger_squeeze_breakout":
            bandwidth = (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle
            is_squeeze = bandwidth.shift(1) < pat["squeeze_threshold"]
            is_breakout = c > indicators.bb_upper
            hit = is_squeeze & is_breakout

        elif pat_name == "macd_zero_cross":
            cross = (indicators.macd.shift(1) < 0) & (indicators.macd > 0)
            if pat.get("above_zero_only", False):
                cross = cross & (indicators.macd > 0)
            if pat.get("require_histogram_expansion", False):
                hist_exp = (indicators.macd_hist > indicators.macd_hist.shift(1)) & (
                    indicators.macd_hist.shift(1) > indicators.macd_hist.shift(2)
                )
                cross = cross & hist_exp
            hit = cross

        elif pat_name == "higher_low_structure":
            lows = low.rolling(pat["pivot_lookback"]).min()
            structure = lows.shift(1) > lows.shift(2)
            vol_conf = v >= v.rolling(20).mean() * 0.8 if pat.get("require_volume_confirmation", False) else True
            hit = structure & vol_conf

        elif pat_name == "volatility_contraction":
            atr_dec = indicators.atr_14 < indicators.atr_14.shift(1)
            n = pat.get("min_contractions", 3)
            contractions = atr_dec.rolling(n).sum() >= n
            vol_dry = v < v.rolling(20).mean() * 0.8 if pat.get("volume_dry_up_required", False) else True
            atr_pct = indicators.atr_14 / c < pat.get("max_atr_pct_of_price", 3.0) / 100.0
            hit = contractions & vol_dry & atr_pct

        elif pat_name == "twenty_day_high_breakout":
            high_20 = c.rolling(21).max().shift(1)
            breakout = c > high_20
            if pat.get("volume_confirm", False):
                vol_mult = pat.get("min_volume_multiplier", 1.5)
                vol_ok = v >= v.rolling(20).mean() * vol_mult
                breakout = breakout & vol_ok
            hit = breakout

        elif pat_name == "nr7_volatility_squeeze":
            lb = pat.get("lookback", 7)
            ranges = h - low
            current_range = ranges
            min_range = ranges.rolling(lb).min().shift(1)
            nr7 = current_range <= min_range * 1.001
            vol_dry = v < v.rolling(20).mean() if pat.get("volume_dry_up", False) else True
            hit = nr7 & vol_dry

        else:
            hit = pd.DataFrame(False, index=dates, columns=symbols, dtype=bool)

        signal_hits += hit.astype(int)

    # --- RSI MOMENTUM FILTER ---
    rsi_filter_cfg = strategy_config.get("rsi_momentum_filter", {})
    if rsi_filter_cfg.get("enabled", False):
        min_rsi = rsi_filter_cfg.get("min_rsi", 50)
        rsi_ok = indicators.rsi_14 >= min_rsi
        if rsi_filter_cfg.get("require_rising", False):
            rsi_ok = rsi_ok & (indicators.rsi_14 > indicators.rsi_14.shift(5))
        # Clear entry patterns where RSI filter fails
        signal_hits = signal_hits.where(rsi_ok, 0)

    # --- TECHNICAL SCORE ---
    # Avoid division by zero
    total_signals = signal_count.replace(0, 1)
    technical_score = signal_hits / total_signals

    # --- BUY RECOMMENDATION ---
    tech_min = thresholds.get("technical_minimum", 0.35)
    is_buy = technical_score >= tech_min

    # Final: all gates + buy recommendation
    pass_matrix = all_gates & is_buy
    score_matrix = technical_score.where(pass_matrix, 0.0)

    return pass_matrix, score_matrix


def signal_matrix_to_dict(
    pass_matrix: pd.DataFrame,
    score_matrix: pd.DataFrame,
) -> Dict[str, Dict]:
    """Convert signal matrices to the format expected by run_with_signals.

    Returns:
        Dict[symbol, Dict[date, {"score": float, "swing_result": dict}]]
    """
    result = {}
    for symbol in pass_matrix.columns:
        symbol_signals = {}
        col_pass = pass_matrix[symbol]
        col_score = score_matrix[symbol]

        for date in pass_matrix.index:
            if col_pass.get(date, False):
                # Normalize date to tz-naive for dict key compatibility
                date_key = date.tz_localize(None) if date.tzinfo else date
                symbol_signals[date_key] = {
                    "score": float(col_score.get(date, 0.0)),
                    "swing_result": {
                        "all_gates_passed": True,
                        "recommendation": "BUY",
                        "technical_score": float(col_score.get(date, 0.0)),
                        "gates": {"trend": True, "volume": True, "volatility": True},
                    },
                }

        if symbol_signals:
            result[symbol] = symbol_signals

    return result
