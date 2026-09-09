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
