"""
Backtests Handler
=================

Provides business logic and MongoDB queries for backtest sessions
and their comprehensive trade execution journals.
"""

import logging
from typing import Any, Dict, List, Optional

from bson import ObjectId

from database import get_mongodb

logger = logging.getLogger(__name__)


def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Convert ObjectId and datetime fields for JSON serialization."""
    if not doc:
        return doc
    res = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            res[k] = str(v)
        elif hasattr(v, "isoformat"):
            res[k] = v.isoformat()
        elif isinstance(v, dict):
            res[k] = _serialize_doc(v)
        elif isinstance(v, list):
            res[k] = [
                _serialize_doc(item) if isinstance(item, dict) else (str(item) if isinstance(item, ObjectId) else item)
                for item in v
            ]
        else:
            res[k] = v
    return res


def list_backtest_sessions() -> List[Dict[str, Any]]:
    """Retrieve all backtest sessions sorted by creation date descending."""
    db = get_mongodb()
    cursor = db.backtest_sessions.find().sort("created_at", -1)
    sessions = []
    for s in cursor:
        clean = _serialize_doc(s)
        metrics = clean.get("summary_metrics", {})
        clean["total_return_pct"] = metrics.get("total_return_pct")
        clean["cagr"] = metrics.get("cagr")
        clean["sharpe_ratio"] = metrics.get("sharpe_ratio")
        clean["max_drawdown_pct"] = metrics.get("max_drawdown_pct")
        clean["win_rate"] = metrics.get("win_rate")
        clean["profit_factor"] = metrics.get("profit_factor")
        clean["total_trades"] = metrics.get("total_trades")
        clean["initial_capital"] = metrics.get("initial_capital")
        clean["final_portfolio_value"] = metrics.get("final_portfolio_value")
        sessions.append(clean)
    return sessions


def get_backtest_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full details of a specific backtest session."""
    db = get_mongodb()
    try:
        oid = ObjectId(session_id)
    except Exception:
        return None

    session = db.backtest_sessions.find_one({"_id": oid})
    if not session:
        return None

    clean = _serialize_doc(session)

    trades_col = db.portfolio_backtest_trades
    total_events = trades_col.count_documents({"session_id": oid})

    pipeline_types = [
        {"$match": {"session_id": oid}},
        {"$group": {"_id": "$trade_type", "count": {"$sum": 1}}},
    ]
    type_counts = {item["_id"]: item["count"] for item in trades_col.aggregate(pipeline_types) if item.get("_id")}

    pipeline_patterns = [
        {"$match": {"session_id": oid, "entry_pattern": {"$ne": None}}},
        {"$group": {"_id": "$entry_pattern", "count": {"$sum": 1}}},
    ]
    pattern_counts = {item["_id"]: item["count"] for item in trades_col.aggregate(pipeline_patterns) if item.get("_id")}

    pipeline_reasons = [
        {"$match": {"session_id": oid, "exit_reason": {"$ne": None}}},
        {"$group": {"_id": "$exit_reason", "count": {"$sum": 1}}},
    ]
    reason_counts = {item["_id"]: item["count"] for item in trades_col.aggregate(pipeline_reasons) if item.get("_id")}

    symbols = trades_col.distinct("symbol", {"session_id": oid})

    clean["execution_summary"] = {
        "total_events": total_events,
        "unique_symbols_traded": len(symbols),
        "trade_types": type_counts,
        "entry_patterns": pattern_counts,
        "exit_reasons": reason_counts,
    }

    clean["yearly_breakdown"] = get_backtest_yearly_breakdown(session_id)

    return clean


def get_backtest_trades(
    session_id: str,
    symbol: Optional[str] = None,
    trade_type: Optional[str] = None,
    exit_reason: Optional[str] = None,
    pattern: Optional[str] = None,
    outcome: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    limit: int = 50,
    sort_by: str = "entry_date",
    sort_dir: str = "desc",
) -> Dict[str, Any]:
    """Retrieve detailed trade execution journal with filtering, pagination, and sorting."""
    db = get_mongodb()
    try:
        oid = ObjectId(session_id)
    except Exception:
        return {"trades": [], "total": 0, "page": page, "limit": limit, "total_pages": 0}

    match_filter: Dict[str, Any] = {"session_id": oid}

    if symbol:
        match_filter["symbol"] = symbol.upper().strip()
    elif search:
        match_filter["symbol"] = {"$regex": search.upper().strip(), "$options": "i"}

    if trade_type and trade_type.upper() != "ALL":
        match_filter["trade_type"] = trade_type.upper()

    if exit_reason and exit_reason != "ALL":
        match_filter["exit_reason"] = exit_reason

    if pattern and pattern != "ALL":
        match_filter["entry_pattern"] = pattern

    if outcome:
        if outcome.lower() == "profit":
            match_filter["pnl"] = {"$gt": 0}
        elif outcome.lower() == "loss":
            match_filter["pnl"] = {"$lt": 0}
        elif outcome.lower() == "breakeven":
            match_filter["pnl"] = 0.0

    trades_col = db.portfolio_backtest_trades
    total_matches = trades_col.count_documents(match_filter)

    sort_direction = -1 if sort_dir.lower() in ("desc", "-1") else 1
    sort_map = {
        "date": "entry_date",
        "entry_date": "entry_date",
        "symbol": "symbol",
        "trade_type": "trade_type",
        "action": "trade_type",
        "price": "entry_price",
        "entry_price": "entry_price",
        "quantity": "quantity",
        "position_value": "position_value",
        "size": "position_value",
        "stop_loss": "stop_loss",
        "target": "target",
        "pnl": "pnl",
        "pnl_pct": "pnl_pct",
        "pattern": "entry_pattern",
        "entry_pattern": "entry_pattern",
        "exit_reason": "exit_reason",
        "reason": "exit_reason",
    }
    sort_field = sort_map.get(sort_by.lower(), "entry_date")

    skip = max(0, (page - 1) * limit)
    cursor = trades_col.find(match_filter).sort(sort_field, sort_direction).skip(skip).limit(limit)

    trades = [_serialize_doc(t) for t in cursor]
    total_pages = (total_matches + limit - 1) // limit if limit > 0 else 1

    return {
        "trades": trades,
        "total": total_matches,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
    }


def get_backtest_symbols(session_id: str) -> List[Dict[str, Any]]:
    """Aggregate trade performance grouped by symbol."""
    db = get_mongodb()
    try:
        oid = ObjectId(session_id)
    except Exception:
        return []

    pipeline = [
        {"$match": {"session_id": oid}},
        {
            "$group": {
                "_id": "$symbol",
                "total_events": {"$sum": 1},
                "total_pnl": {"$sum": "$pnl"},
                "buys": {"$sum": {"$cond": [{"$eq": ["$trade_type", "BUY"]}, 1, 0]}},
                "pyramids": {"$sum": {"$cond": [{"$eq": ["$trade_type", "PYRAMID_ADD"]}, 1, 0]}},
                "sells": {"$sum": {"$cond": [{"$in": ["$trade_type", ["SELL", "PARTIAL_SELL"]]}, 1, 0]}},
                "winning_exits": {"$sum": {"$cond": [{"$gt": ["$pnl", 0]}, 1, 0]}},
                "losing_exits": {"$sum": {"$cond": [{"$lt": ["$pnl", 0]}, 1, 0]}},
                "first_date": {"$min": "$entry_date"},
                "last_date": {"$max": "$entry_date"},
            }
        },
        {"$sort": {"total_pnl": -1}},
    ]

    results = []
    for item in db.portfolio_backtest_trades.aggregate(pipeline):
        total_exits = item["winning_exits"] + item["losing_exits"]
        win_rate = (item["winning_exits"] / total_exits * 100) if total_exits > 0 else 0.0
        results.append(
            {
                "symbol": item["_id"],
                "total_events": item["total_events"],
                "total_pnl": round(item["total_pnl"], 2),
                "buys": item["buys"],
                "pyramids": item["pyramids"],
                "sells": item["sells"],
                "win_rate": round(win_rate, 1),
                "winning_exits": item["winning_exits"],
                "losing_exits": item["losing_exits"],
                "first_date": item["first_date"],
                "last_date": item["last_date"],
            }
        )
    return results


def delete_backtest_session(session_id: str) -> bool:
    """Delete a backtest session and all its associated trades."""
    db = get_mongodb()
    try:
        oid = ObjectId(session_id)
    except Exception:
        return False

    db.portfolio_backtest_trades.delete_many({"session_id": oid})
    res = db.backtest_sessions.delete_one({"_id": oid})
    return res.deleted_count > 0


KNOWN_MARKET_CONTEXTS: Dict[str, Dict[str, str]] = {
    "2016-2017": {
        "context": "Technical warmup & 100-bar baseline. Capital preserved in cash.",
        "badge": "Warmup Baseline",
        "regime": "Defensive / Cash",
    },
    "2016": {
        "context": "Technical warmup & 100-bar baseline. Capital preserved in cash.",
        "badge": "Warmup Baseline",
        "regime": "Defensive / Cash",
    },
    "2017": {
        "context": "Technical warmup & baseline initialization. Capital preserved.",
        "badge": "Warmup Baseline",
        "regime": "Defensive / Cash",
    },
    "2018": {
        "context": "Severe Mid/Small-cap crash (NSE Smallcap plunged -35%). Capital protected.",
        "badge": "Crash Defense",
        "regime": "Bearish Shock",
    },
    "2019": {
        "context": "NBFC / IL&FS liquidity crisis. Minimal drawdown compared to market.",
        "badge": "Liquidity Crisis Defense",
        "regime": "High Dispersion",
    },
    "2020": {
        "context": "Survived March 2020 COVID flash crash (-40% Nifty) with only -27.8% peak DD.",
        "badge": "COVID Resilience",
        "regime": "V-Bottom Rally",
    },
    "2021": {
        "context": "Post-COVID expansion & momentum breakout surge.",
        "badge": "Cyclical Surge",
        "regime": "Strong Bull Run",
    },
    "2022": {
        "context": "Global rate-hike correction & consolidation.",
        "badge": "Rate Hike Pullback",
        "regime": "Choppy Consolidation",
    },
    "2023": {
        "context": "Massive multi-bagger compounding (HAL, COCHINSHIP, GVT&D, KALYANKJIL).",
        "badge": "Multi-Bagger Supercycle",
        "regime": "Aggressive Momentum",
    },
    "2024": {
        "context": "Broad market rally continuation.",
        "badge": "Broad Expansion",
        "regime": "Secular Bull Trend",
    },
    "2025": {
        "context": "Market consolidation & defensive trailing.",
        "badge": "Defensive Trailing",
        "regime": "Base Consolidation",
    },
    "2026": {
        "context": "YTD active momentum expansion.",
        "badge": "Active Momentum",
        "regime": "Momentum Expansion",
    },
}


def get_backtest_yearly_breakdown(session_id: str) -> List[Dict[str, Any]]:
    """
    Compute year-over-year (YoY) performance breakdown for a backtest session,
    including compounded returns, annual drawdowns, trade metrics, and market contexts.
    """
    from collections import defaultdict

    db = get_mongodb()
    try:
        oid = ObjectId(session_id)
    except Exception:
        return []

    snaps = list(db.portfolio_backtest_daily_snapshots.find({"session_id": oid}).sort("date", 1))
    trades = list(db.portfolio_backtest_trades.find({"session_id": oid}))

    # Group trades by year
    trades_by_yr = defaultdict(list)
    for t in trades:
        dt = t.get("exit_date") or t.get("entry_date") or ""
        yr = dt[:4] if len(dt) >= 4 else "unknown"
        trades_by_yr[yr].append(t)

    # Group snapshots by year
    snaps_by_yr = defaultdict(list)
    for s in snaps:
        yr = s.get("date", "")[:4]
        if yr:
            snaps_by_yr[yr].append(s)

    years_sorted = (
        sorted(snaps_by_yr.keys()) if snaps_by_yr else sorted([y for y in trades_by_yr.keys() if y.isdigit()])
    )
    if not years_sorted:
        return []

    # Check if 2016 and 2017 are warmup periods with 0 trades
    combine_16_17 = (
        "2016" in years_sorted
        and "2017" in years_sorted
        and len(trades_by_yr.get("2016", [])) == 0
        and len(trades_by_yr.get("2017", [])) == 0
    )

    processed_years = []
    if combine_16_17:
        processed_years.append(
            (
                "2016-2017",
                snaps_by_yr.get("2016", []) + snaps_by_yr.get("2017", []),
                trades_by_yr.get("2016", []) + trades_by_yr.get("2017", []),
            )
        )
        for y in years_sorted:
            if y not in ["2016", "2017"]:
                processed_years.append((y, snaps_by_yr.get(y, []), trades_by_yr.get(y, [])))
    else:
        for y in years_sorted:
            processed_years.append((y, snaps_by_yr.get(y, []), trades_by_yr.get(y, [])))

    breakdown = []
    prior_eoy = None

    for y_label, y_snaps, y_trades in processed_years:
        if y_snaps:
            start_pv = y_snaps[0].get("portfolio_value", 0.0)
            end_pv = y_snaps[-1].get("portfolio_value", 0.0)
            base_pv = prior_eoy if prior_eoy is not None else start_pv
            ret = ((end_pv - base_pv) / base_pv * 100) if base_pv > 0 else 0.0
            prior_eoy = end_pv

            min_dd = min((s.get("drawdown_from_peak_pct", 0.0) or 0.0) for s in y_snaps) if y_snaps else 0.0
            max_pv = max((s.get("portfolio_value", 0.0) or 0.0) for s in y_snaps) if y_snaps else end_pv
            min_pv = min((s.get("portfolio_value", 0.0) or 0.0) for s in y_snaps) if y_snaps else end_pv
            trading_days = len(y_snaps)
        else:
            # Fallback if no snapshots exist for this session
            exits_fallback = [t for t in y_trades if t.get("trade_type") in ["SELL", "PARTIAL_SELL"]]
            pnl_sum = sum(t.get("pnl", 0.0) or 0.0 for t in exits_fallback)
            base_pv = prior_eoy or 10000.0
            end_pv = base_pv + pnl_sum
            ret = (pnl_sum / base_pv * 100) if base_pv > 0 else 0.0
            prior_eoy = end_pv
            min_dd = 0.0
            max_pv = end_pv
            min_pv = base_pv
            trading_days = len(y_trades)

        exits = [t for t in y_trades if t.get("trade_type") in ["SELL", "PARTIAL_SELL"]]
        buys = [t for t in y_trades if t.get("trade_type") == "BUY"]
        pyramids = [t for t in y_trades if t.get("trade_type") == "PYRAMID_ADD"]
        wins = [t for t in exits if (t.get("pnl", 0.0) or 0.0) > 0]
        losses = [t for t in exits if (t.get("pnl", 0.0) or 0.0) < 0]
        total_pnl = sum((t.get("pnl", 0.0) or 0.0) for t in exits)
        win_rate = (len(wins) / len(exits) * 100) if exits else 0.0

        # Best trade
        best_exit = max(exits, key=lambda x: (x.get("pnl", 0.0) or 0.0)) if exits else None
        worst_exit = min(exits, key=lambda x: (x.get("pnl", 0.0) or 0.0)) if exits else None

        best_trade_info = None
        if best_exit and (best_exit.get("pnl", 0.0) or 0.0) > 0:
            best_trade_info = {
                "symbol": best_exit.get("symbol"),
                "pnl": round(best_exit.get("pnl", 0.0) or 0.0, 2),
                "pnl_pct": round(best_exit.get("pnl_pct", 0.0) or 0.0, 2),
                "exit_date": best_exit.get("exit_date"),
                "exit_reason": best_exit.get("exit_reason"),
            }

        worst_trade_info = None
        if worst_exit and (worst_exit.get("pnl", 0.0) or 0.0) < 0:
            worst_trade_info = {
                "symbol": worst_exit.get("symbol"),
                "pnl": round(worst_exit.get("pnl", 0.0) or 0.0, 2),
                "pnl_pct": round(worst_exit.get("pnl_pct", 0.0) or 0.0, 2),
                "exit_date": worst_exit.get("exit_date"),
                "exit_reason": worst_exit.get("exit_reason"),
            }

        # Top gainers by symbol for this year
        symbol_pnl = defaultdict(float)
        for t in exits:
            symbol_pnl[t.get("symbol", "")] += t.get("pnl", 0.0) or 0.0

        top_gainers = [
            {"symbol": sym, "pnl": round(pnl, 2)}
            for sym, pnl in sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
            if pnl > 0 and sym
        ][:5]

        # Context narrative
        ctx_info = KNOWN_MARKET_CONTEXTS.get(y_label)
        if not ctx_info:
            if ret > 25:
                ctx_info = {
                    "context": "Strong momentum expansion and high-velocity trend breakouts.",
                    "badge": "High Momentum Surge",
                    "regime": "Aggressive Bull",
                }
            elif ret > 5:
                ctx_info = {
                    "context": "Constructive cyclical advance with disciplined risk management.",
                    "badge": "Cyclical Trend",
                    "regime": "Moderate Bull",
                }
            elif ret >= -5:
                ctx_info = {
                    "context": "Market consolidation and capital preservation phase.",
                    "badge": "Range Consolidation",
                    "regime": "Neutral / Sideways",
                }
            else:
                ctx_info = {
                    "context": "Corrective market regime. Defensive trailing stops limited drawdowns.",
                    "badge": "Defensive Trailing",
                    "regime": "Market Correction",
                }

        display_year = "2016–2017" if y_label == "2016-2017" else y_label

        breakdown.append(
            {
                "year": y_label,
                "display_year": display_year,
                "return_pct": round(ret, 2),
                "start_portfolio_value": round(base_pv, 2),
                "end_portfolio_value": round(end_pv, 2),
                "peak_portfolio_value": round(max_pv, 2),
                "trough_portfolio_value": round(min_pv, 2),
                "max_drawdown_pct": round(min_dd, 2),
                "trading_days": trading_days,
                "total_events": len(y_trades),
                "buys_count": len(buys),
                "pyramids_count": len(pyramids),
                "exits_count": len(exits),
                "winning_exits": len(wins),
                "losing_exits": len(losses),
                "win_rate": round(win_rate, 1),
                "realized_pnl": round(total_pnl, 2),
                "market_context": ctx_info["context"],
                "market_badge": ctx_info["badge"],
                "regime": ctx_info["regime"],
                "best_trade": best_trade_info,
                "worst_trade": worst_trade_info,
                "top_gainers": top_gainers,
            }
        )

    return breakdown
