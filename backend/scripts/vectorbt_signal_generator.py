"""
Vectorbt-based batch signal generation.

Replaces the per-symbol per-date nested Python loops in _compute_signals_worker
with fully vectorized boolean array operations on pre-computed indicators.
"""

from typing import Any, Dict, Tuple

import pandas as pd

from .vectorbt_indicator_batch import ComputedIndicators


def compute_stock_prefilter(
    indicators: ComputedIndicators,
    strategy_config: Dict[str, Any],
) -> pd.DataFrame:
    """Apply stock_filters as vectorbt boolean masks.

    Returns a bool DataFrame (dates x symbols) — True if stock passes ALL pre-filters on that date.
    """
    filters = strategy_config.get("stock_filters", [])

    c = indicators.close
    v = indicators.volume

    symbols = indicators.symbols
    dates = indicators.dates

    passed = pd.DataFrame(True, index=dates, columns=symbols, dtype=bool)

    for f in filters:
        f_type = f["type"]

        if f_type == "price":
            op = f["op"]
            passed = passed & ~c.isna()
            if op == "between":
                passed = passed & (c > f["min"]) & (c < f["max"])
            elif op == ">":
                passed = passed & (c > f["value"])
            elif op == "<":
                passed = passed & (c < f["value"])

        elif f_type == "volume":
            passed = passed & ~v.isna() & (v > f["value"])

        elif f_type == "rsi":
            op = f["op"]
            period = f.get("period", 14)
            rsi_val = indicators.rsi_9 if period == 9 else indicators.rsi_14
            passed = passed & ~rsi_val.isna()
            if op == "between":
                passed = passed & (rsi_val >= f["min"]) & (rsi_val <= f["max"])
            elif op == ">":
                passed = passed & (rsi_val > f["value"])
            elif op == "<":
                passed = passed & (rsi_val < f["value"])

        elif f_type == "moving_average":
            kind = f["kind"].lower()
            period = f["period"]
            target = f["target"].lower()
            op = f["op"]

            if kind == "hma" or op == "monitor":
                continue

            if target == "close":
                if period == 20:
                    sma = indicators.sma_20
                elif period == 50:
                    sma = indicators.sma_50
                elif period == 150:
                    sma = indicators.sma_150
                elif period == 200:
                    sma = indicators.sma_200
                else:
                    import vectorbt as vbt

                    sma = vbt.talib("SMA").run(c, timeperiod=period).real
                    sma.columns = symbols

                if op == ">":
                    passed = passed & ~sma.isna() & (c > sma)
                elif op == "<":
                    passed = passed & ~sma.isna() & (c < sma)

        elif f_type == "volume_spike_lookup":
            lookback = f["lookback_days"]
            multiplier = f["multiplier"]
            ma_period = f["ma_period"]

            import vectorbt as vbt

            vol_sma = vbt.talib("SMA").run(v, timeperiod=ma_period).real
            vol_sma.columns = symbols

            spike_any = pd.DataFrame(False, index=dates, columns=symbols, dtype=bool)
            for i in range(1, lookback + 1):
                past_vol = v.shift(i)
                past_sma = vol_sma.shift(i)
                spike_any = spike_any | (past_vol > past_sma * multiplier)

            passed = passed & spike_any
            passed = passed & ~v.isna() & ~vol_sma.isna()

        elif f_type == "market_cap":
            try:
                from scripts.data_fetcher import get_market_caps

                mc_cache = get_market_caps(symbols)
                min_cap = f.get("value", 0)
                mc_mask = pd.DataFrame(False, index=dates, columns=symbols, dtype=bool)
                for sym in symbols:
                    cap = mc_cache.get(sym, None)
                    if cap is not None and cap > min_cap:
                        mc_mask[sym] = True
                passed = passed & mc_mask
            except Exception:
                pass

    return passed


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

    trend_ok = adx > t_cfg.get("adx_min", 20)
    if t_cfg.get("require_di_alignment", True):
        trend_ok = trend_ok & (pdi > mdi)
    if t_cfg.get("require_price_above_sma", True):
        trend_ok = trend_ok & (c > sma_trend)
    if indicators.sma_200 is not None:
        trend_ok = trend_ok & (c > indicators.sma_200)

    # SMA Stack: 50 > 150 > 200
    if t_cfg.get("require_sma_stack", True):
        if indicators.sma_50 is not None and indicators.sma_150 is not None and indicators.sma_200 is not None:
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

    # --- INDICATOR SIGNALS (StrategyEvaluator scoring) ---
    signal_count = pd.DataFrame(0, index=dates, columns=symbols, dtype=int)
    signal_hits = pd.DataFrame(0, index=dates, columns=symbols, dtype=int)
    strategy_count = pd.DataFrame(0, index=dates, columns=symbols, dtype=int)
    strategy_hits = pd.DataFrame(0, index=dates, columns=symbols, dtype=int)

    # MACD
    if strat_cfg.get("MACD_Signal_Crossover", {}).get("enabled", False):
        macd_hit = indicators.macd > indicators.macd_signal
        is_bonus = strat_cfg["MACD_Signal_Crossover"].get("is_bonus", False)
        signal_count += 1
        strategy_count += 1
        signal_hits += macd_hit.astype(int)
        strategy_hits += macd_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & macd_hit

    # RSI
    if strat_cfg.get("RSI_Overbought_Oversold", {}).get("enabled", False):
        rsi_hit = indicators.rsi_14 > 50
        is_bonus = strat_cfg["RSI_Overbought_Oversold"].get("is_bonus", False)
        signal_count += 1
        strategy_count += 1
        signal_hits += rsi_hit.astype(int)
        strategy_hits += rsi_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & rsi_hit

    # BBANDS
    if strat_cfg.get("Bollinger_Band_Squeeze", {}).get("enabled", False):
        bb_hit = c > indicators.bb_middle
        is_bonus = strat_cfg["Bollinger_Band_Squeeze"].get("is_bonus", False)
        signal_count += 1
        strategy_count += 1
        signal_hits += bb_hit.astype(int)
        strategy_hits += bb_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & bb_hit

    # ADX Strength
    if strat_cfg.get("ADX_Trend_Strength", {}).get("enabled", False):
        adx_hit = indicators.adx > strat_cfg["ADX_Trend_Strength"]["threshold"]
        is_bonus = strat_cfg["ADX_Trend_Strength"].get("is_bonus", False)
        signal_count += 1
        strategy_count += 1
        signal_hits += adx_hit.astype(int)
        strategy_hits += adx_hit.astype(int)
        if not is_bonus:
            all_gates = all_gates & adx_hit

    # On Balance Volume (OBV) — mirrors strategies/on_balance_volume.py
    # Scoring-only signal (StrategyEvaluator), NOT a hard gate
    if strat_cfg.get("On_Balance_Volume", {}).get("enabled", False):
        obv_lookback = strat_cfg["On_Balance_Volume"].get("lookback_period", 3)
        obv = indicators.obv
        obv_trend = obv - obv.shift(obv_lookback)
        price_trend = c - c.shift(obv_lookback)
        obv_hit = (obv_trend > 0) & (price_trend > 0)
        obv_hit = obv_hit | ((obv_trend > 0) & (price_trend <= 0))
        signal_count += 1
        strategy_count += 1
        signal_hits += obv_hit.astype(int)
        strategy_hits += obv_hit.astype(int)

    # Pocket Pivot Entry — mirrors strategies/pocket_pivot_entry.py
    # Scoring-only signal (StrategyEvaluator), NOT a hard gate
    if strat_cfg.get("Pocket_Pivot_Entry", {}).get("enabled", False):
        pp_lookback = strat_cfg["Pocket_Pivot_Entry"].get("lookback", 10)
        pp_sma_fast = strat_cfg["Pocket_Pivot_Entry"].get("sma_fast", 10)
        pp_sma_slow = strat_cfg["Pocket_Pivot_Entry"].get("sma_slow", 50)

        if pp_sma_fast == 20:
            sma_fast = indicators.sma_20
        elif pp_sma_fast == 50:
            sma_fast = indicators.sma_50
        else:
            import vectorbt as vbt

            sma_fast = vbt.talib("SMA").run(c, timeperiod=pp_sma_fast).real
            sma_fast.columns = symbols

        if pp_sma_slow == 50:
            sma_slow = indicators.sma_50
        elif pp_sma_slow == 150:
            sma_slow = indicators.sma_150
        else:
            import vectorbt as vbt

            sma_slow = vbt.talib("SMA").run(c, timeperiod=pp_sma_slow).real
            sma_slow.columns = symbols

        above_mas = (c > sma_fast) & (c > sma_slow)

        down_day = c < c.shift(1)
        down_vol = v.where(down_day, 0)
        max_down_vol = down_vol.rolling(pp_lookback, min_periods=1).max().shift(1)
        pp_hit = above_mas & (v > max_down_vol)

        signal_count += 1
        strategy_count += 1
        signal_hits += pp_hit.astype(int)
        strategy_hits += pp_hit.astype(int)

    # Volume Breakout — mirrors strategies/volume_breakout.py
    # Scoring-only signal (StrategyEvaluator), NOT a hard gate
    if strat_cfg.get("Volume_Breakout", {}).get("enabled", False):
        vb_threshold = strat_cfg["Volume_Breakout"].get("threshold", 2.0)
        vol_ma20 = v.rolling(20, min_periods=10).mean()
        resistance_20 = h.rolling(20, min_periods=10).max().shift(1)
        daily_range = h - low
        close_position = (c - low) / daily_range.replace(0, pd.NA)
        volume_spike = v >= (vol_ma20 * vb_threshold)
        vb_hit = (c > resistance_20) & (close_position >= 0.75) & volume_spike

        signal_count += 1
        strategy_count += 1
        signal_hits += vb_hit.astype(int)
        strategy_hits += vb_hit.astype(int)

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
            if pat.get("max_distance_from_21ema_pct") is not None:
                max_dist = pat["max_distance_from_21ema_pct"] / 100.0
                dist = (c - indicators.ema_21) / indicators.ema_21
                breakout = breakout & (dist <= max_dist)
            if pat.get("retest_required", False):
                recent_low_5 = low.rolling(5).min()
                retest_ok = (recent_low_5 <= high_20 * 1.025) & (recent_low_5 >= high_20 * 0.97)
                breakout = breakout & retest_ok
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
    # Use strategy-only scoring to match live StrategyEvaluator (pos/total_strategies)
    # Entry patterns contribute to signal_hits (informational) but NOT to the BUY score
    total_strats = strategy_count.replace(0, 1)
    technical_score = strategy_hits / total_strats

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
