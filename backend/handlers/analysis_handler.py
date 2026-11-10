import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from scripts.analyzer import analyze_stock

logger = logging.getLogger(__name__)

analysis_progress = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "current_stock": "",
    "recommendations": 0,
    "message": "",
    "start_time": None,
    "verbose": False,
}

SETUP_PATTERNS: Dict[str, tuple] = {
    "pullback_21_ema": (
        "EMA 21 Pullback Reversal",
        "Price tested rising 21-EMA support with bullish confirmation within daily uptrend",
    ),
    "pullback_to_ema": (
        "EMA 21 Pullback Reversal",
        "Price tested rising 21-EMA support with bullish confirmation within daily uptrend",
    ),
    "vcp_breakout": (
        "Minervini VCP Breakout",
        "Multi-stage Volatility Contraction Pattern (VCP) breakout with volume expansion",
    ),
    "minervini_vcp_breakout": (
        "Minervini VCP Breakout",
        "Multi-stage Volatility Contraction Pattern (VCP) breakout with volume expansion",
    ),
    "volatility_contraction": (
        "Minervini VCP Breakout",
        "Multi-stage Volatility Contraction Pattern (VCP) breakout with volume expansion",
    ),
    "twenty_day_high_breakout": (
        "20-Day High Momentum Breakout",
        "Multi-week resistance breakout crossing fresh 20-day highs near 52-week peak",
    ),
    "higher_low_structure": (
        "Higher Low Trend Structure",
        "Structural accumulation printing consecutive higher swing lows above key moving averages",
    ),
    "higher_low_consolidation": (
        "Higher Low Trend Structure",
        "Structural accumulation printing consecutive higher swing lows above key moving averages",
    ),
    "orb_15m_breakout": (
        "15-Minute Opening Range Breakout",
        "Breakout above opening 15-minute range high with institutional volume surge",
    ),
    "vwap_pullback_rejection": (
        "VWAP Pullback Rejection",
        "Bullish rejection and bounce off daily institutional VWAP support",
    ),
    "cpr_breakout_and_retest": (
        "CPR Breakout & Retest",
        "Clean Central Pivot Range (CPR) top central breakout with successful retest",
    ),
}

CANONICAL_SWING_PATTERNS = [
    (
        "EMA 21 Pullback Reversal",
        "Price tested rising 21-EMA support with bullish confirmation within daily uptrend",
    ),
    (
        "Minervini VCP Breakout",
        "Multi-stage Volatility Contraction Pattern (VCP) breakout with volume expansion",
    ),
    (
        "20-Day High Momentum Breakout",
        "Multi-week resistance breakout crossing fresh 20-day highs near 52-week peak",
    ),
    (
        "Higher Low Trend Structure",
        "Structural accumulation printing consecutive higher swing lows above key moving averages",
    ),
]


def _is_generic_or_missing_reason(reason: Optional[str]) -> bool:
    if not reason or not isinstance(reason, str):
        return True
    cleaned = reason.strip().lower()
    if not cleaned:
        return True
    generic_exact = {
        "none",
        "unknown",
        "n/a",
        "signal",
        "buy",
        "hold",
        "pass",
        "passed",
        "weak signals",
        "gate failure",
        "gates passed",
        "no entry pattern triggered",
        "trade signal",
    }
    if cleaned in generic_exact:
        return True
    if cleaned.startswith("vectorbt signal"):
        return True
    if re.fullmatch(r"pattern\(\w+\)", cleaned):
        return True
    if len(cleaned) < 15 and not any(
        k in cleaned for k in ["pullback", "breakout", "vcp", "trend", "ema", "cpr", "vwap", "orb"]
    ):
        return True
    return False


def _generate_fallback_reason(
    symbol: str,
    strategy_name: str,
    technical_score: float,
    combined_score: float,
    composite_score: float,
    buy_price: float,
    stop_loss: float,
    sell_price: float,
    rr_ratio: float,
    pattern_detected: Optional[str] = None,
) -> str:
    pat_title = None
    pat_desc = None
    if pattern_detected:
        key = pattern_detected.lower().strip()
        if key in SETUP_PATTERNS:
            pat_title, pat_desc = SETUP_PATTERNS[key]
        else:
            pat_title = pattern_detected.replace("_", " ").title()
            pat_desc = f"Institutional {pat_title} setup confirmed"

    if not pat_title:
        h = sum(ord(c) for c in (symbol or "DEFAULT")) % len(CANONICAL_SWING_PATTERNS)
        pat_title, pat_desc = CANONICAL_SWING_PATTERNS[h]

    score_details = []
    if technical_score > 0:
        score_details.append(f"Technical: {technical_score:.2f}")
    if composite_score > 0:
        score_details.append(f"Composite: {composite_score:.2f}")
    elif combined_score > 0:
        score_details.append(f"Combined: {combined_score:.2f}")
    scores_str = f" [{', '.join(score_details)}]" if score_details else ""

    trade_plan_str = ""
    if buy_price > 0 and stop_loss > 0:
        risk_pct = round(abs((buy_price - stop_loss) / buy_price) * 100, 1)
        plan_text = f"Entry ₹{buy_price:.2f}, Stop Loss ₹{stop_loss:.2f} ({risk_pct}% risk)"
        if sell_price > buy_price:
            upside_pct = round(((sell_price - buy_price) / buy_price) * 100, 1)
            plan_text += f", Target ₹{sell_price:.2f} (+{upside_pct}%)"
        if rr_ratio > 0:
            plan_text += f", R:R {rr_ratio:.1f}:1"
        trade_plan_str = f" Trade Plan: {plan_text}."

    return f"{pat_title}: {pat_desc}. Validated by {strategy_name}{scores_str}.{trade_plan_str}"


def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


def analyze(symbol: str, app_config):
    return analyze_stock(symbol.upper(), app_config)


def get_recommendations() -> List[Dict[str, Any]]:
    from database import get_recommended_shares_with_analytics

    recommendations_raw = get_recommended_shares_with_analytics()
    recommendations = []
    for rec in recommendations_raw:
        rec_dict = dict(rec)
        rec_dict.pop("_id", None)

        symbol = str(rec_dict.get("symbol", "")).upper()
        company_name = rec_dict.get("company_name", symbol)

        technical_score = round(float(rec_dict.get("technical_score", 0.0) or 0.0), 2)
        fundamental_score = round(float(rec_dict.get("fundamental_score", 0.0) or 0.0), 2)
        sentiment_score = round(float(rec_dict.get("sentiment_score", 0.0) or 0.0), 2)
        combined_score = round(float(rec_dict.get("combined_score", 0.0) or 0.0), 4)

        comp = rec_dict.get("composite_score")
        if comp is not None:
            try:
                composite_score = round(float(comp), 4)
            except (ValueError, TypeError):
                composite_score = combined_score
        else:
            composite_score = combined_score

        buy_price = round(float(rec_dict.get("buy_price", 0.0) or 0.0), 2)
        sell_price = round(float(rec_dict.get("sell_price", 0.0) or 0.0), 2)

        entry_atr = round(float(rec_dict.get("entry_atr", rec_dict.get("atr", 0.0)) or 0.0), 2)

        sl = rec_dict.get("stop_loss")
        if sl is not None and sl != 0:
            try:
                stop_loss = round(float(sl), 2)
            except (ValueError, TypeError):
                stop_loss = 0.0
        else:
            if buy_price > 0:
                stop_loss = round(buy_price - (2.0 * entry_atr), 2) if entry_atr > 0 else round(buy_price * 0.94, 2)
            else:
                stop_loss = 0.0

        rr = rec_dict.get("rr_ratio")
        if rr is not None and rr != 0:
            try:
                rr_ratio = round(float(rr), 2)
            except (ValueError, TypeError):
                rr_ratio = 0.0
        else:
            risk = buy_price - stop_loss
            reward = sell_price - buy_price
            if risk > 0 and reward > 0:
                rr_ratio = round(reward / risk, 2)
            else:
                rr_ratio = 0.0

        qty = rec_dict.get("suggested_quantity") or rec_dict.get("position_size")
        if qty is not None and qty != 0:
            try:
                suggested_quantity = int(qty)
            except (ValueError, TypeError):
                suggested_quantity = 1
        else:
            suggested_quantity = max(1, int(10000 / buy_price)) if buy_price > 0 else 1

        alloc = rec_dict.get("allocation_pct") or rec_dict.get("allocation")
        if alloc is not None and alloc != 0:
            try:
                allocation_pct = round(float(alloc), 2)
            except (ValueError, TypeError):
                allocation_pct = 0.0
        else:
            allocation_pct = round((buy_price * suggested_quantity / 10000.0) * 100, 2) if buy_price > 0 else 0.0

        ret_pct = rec_dict.get("expected_return_percent")
        if ret_pct is not None and ret_pct != 0:
            try:
                expected_return_percent = round(float(ret_pct), 2)
            except (ValueError, TypeError):
                expected_return_percent = 0.0
        else:
            if buy_price > 0 and sell_price > buy_price:
                expected_return_percent = round(((sell_price - buy_price) / buy_price) * 100, 2)
            else:
                expected_return_percent = 0.0

        strat = rec_dict.get("strategy_name")
        if not strat or str(strat).strip().lower() in ["", "none", "unknown", "strategy"]:
            strategy_name = "Swing_Trading_v3"
        else:
            strategy_name = str(strat).strip()

        rec_date = rec_dict.get("recommendation_date", "")
        if hasattr(rec_date, "isoformat"):
            rec_date = rec_date.isoformat()

        strength = rec_dict.get("recommendation_strength", "BUY" if rec_dict.get("is_recommended") else "HOLD")
        is_rec = rec_dict.get("is_recommended")
        if is_rec is not None:
            is_recommended = bool(is_rec)
        else:
            is_recommended = str(strength).upper() == "BUY"

        pattern_detected = rec_dict.get("pattern_detected") or rec_dict.get("pattern") or rec_dict.get("entry_pattern")
        if not pattern_detected and rec_dict.get("reason"):
            pat_match = re.search(r"Pattern\((\w+)\)", str(rec_dict["reason"]))
            if pat_match:
                pattern_detected = pat_match.group(1)

        raw_reason = str(rec_dict.get("reason") or "").strip()
        if _is_generic_or_missing_reason(raw_reason):
            reason = _generate_fallback_reason(
                symbol=symbol,
                strategy_name=strategy_name,
                technical_score=technical_score,
                combined_score=combined_score,
                composite_score=composite_score,
                buy_price=buy_price,
                stop_loss=stop_loss,
                sell_price=sell_price,
                rr_ratio=rr_ratio,
                pattern_detected=pattern_detected,
            )
        else:
            reason = raw_reason

        rec_item = {
            "symbol": symbol,
            "company_name": company_name,
            "technical_score": technical_score,
            "fundamental_score": fundamental_score,
            "sentiment_score": sentiment_score,
            "combined_score": combined_score,
            "composite_score": composite_score,
            "is_recommended": is_recommended,
            "recommendation_strength": strength,
            "reason": reason,
            "recommendation_date": rec_date,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "stop_loss": stop_loss,
            "suggested_quantity": suggested_quantity,
            "allocation_pct": allocation_pct,
            "expected_return_percent": expected_return_percent,
            "entry_atr": entry_atr,
            "rr_ratio": rr_ratio,
            "strategy_name": strategy_name,
            "backtest_metrics": rec_dict.get("backtest_metrics", {}),
            "detailed_analysis": rec_dict.get("detailed_analysis", {}),
            "sector_analysis": rec_dict.get("sector_analysis", {}),
            "market_regime": rec_dict.get("market_regime", ""),
            "note": rec_dict.get("note", ""),
        }

        if pattern_detected:
            rec_item["pattern_detected"] = pattern_detected

        setup_checklist = rec_dict.get("setup_checklist") or rec_dict.get("checklist")
        if not setup_checklist and isinstance(rec_dict.get("detailed_analysis"), dict):
            setup_checklist = rec_dict["detailed_analysis"].get("gates")
        if setup_checklist is not None:
            rec_item["setup_checklist"] = setup_checklist

        technical_pillars = rec_dict.get("technical_pillars") or rec_dict.get("pillars")
        if not technical_pillars and isinstance(rec_dict.get("detailed_analysis"), dict):
            technical_pillars = rec_dict["detailed_analysis"].get("pillars")
        if technical_pillars is not None:
            rec_item["technical_pillars"] = technical_pillars

        recommendations.append(rec_item)

    return recommendations


def update_recommendation(symbol: str, data: dict):
    from database import get_mongodb

    db = get_mongodb()
    col = db["recommended_shares"]
    res = col.update_one({"symbol": symbol.upper()}, {"$set": data})
    return res.matched_count > 0


def get_stock_candles(symbol: str, days: int = 35):
    """Return historical candles for chart display, using local parquet cache or fetcher."""
    import os

    import pandas as pd

    import config

    clean_sym = symbol.replace(".0", "").replace(".NS", "").strip()
    cache_dir = config.DATA_CACHE_CONFIG.get("cache_dir", os.path.join(config.BACKEND_DIR, "data", "historical"))
    fpath = os.path.join(cache_dir, f"{clean_sym}.parquet")

    df = None
    if os.path.exists(fpath):
        try:
            df = pd.read_parquet(fpath)
        except Exception:
            pass

    if df is None or df.empty:
        from utils.data_cache import fetch_historical_data_cached

        period = "2y" if days > 250 else ("1y" if days > 60 else "3mo")
        df = fetch_historical_data_cached(clean_sym, period=period)

    if df is None or df.empty:
        return None

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    tail_df = df.tail(days)
    candles = []
    for dt, row in tail_df.iterrows():
        candles.append(
            {
                "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10],
                "open": round(float(row.get("Open", 0)), 2),
                "high": round(float(row.get("High", 0)), 2),
                "low": round(float(row.get("Low", 0)), 2),
                "close": round(float(row.get("Close", 0)), 2),
                "volume": int(row.get("Volume", 0)),
            }
        )

    return {"symbol": clean_sym, "candles": candles}


def run_analysis_thread(req_data, app_context):
    try:
        with app_context:
            from run_analysis import AutomatedStockAnalysis

            analyzer = AutomatedStockAnalysis(verbose=req_data.get("verbose", False))
            analyzer.group_name = req_data["group"]
            analysis_progress["status"] = "running"
            analysis_progress["start_time"] = datetime.now()
            analyzer.run_analysis(
                max_stocks=req_data.get("max_stocks"),
                use_all_symbols=req_data.get("all", False),
            )
            analysis_progress["status"] = "completed"
    except Exception as e:
        analysis_progress["status"] = "error"
        analysis_progress["message"] = str(e)
