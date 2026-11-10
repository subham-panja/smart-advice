import logging
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


def get_strategies():
    from utils.strategy_loader import StrategyLoader

    all_strats = StrategyLoader.load_all_strategies_including_disabled()
    result = []
    for s in all_strats:
        file_name = s.pop("_file_name", "")
        category = s.pop("_category", "swing")
        relative_path = s.pop("_relative_path", file_name)
        result.append({**s, "file_name": file_name, "category": category, "relative_path": relative_path})
    return result


def get_trading_config():
    return config.TRADING_OPTIONS


def get_cycle_stats():
    from database import get_mongodb, get_open_positions
    from utils.settlement import get_available_cash

    db = get_mongodb()
    positions = get_open_positions()
    closed_positions = list(db[config.MONGODB_COLLECTIONS["positions"]].find({"status": "CLOSED"}))
    initial_cap = config.TRADING_OPTIONS.get("initial_capital", 100000.0)
    brokerage_pct = config.TRADING_OPTIONS.get("brokerage_charges", 0.0020)

    def pos_cost(p):
        return p.get("quantity", 0) * p.get("entry_price", 0) * (1 + brokerage_pct)

    total_invested = sum(pos_cost(p) for p in positions)
    total_mkt_val = sum(p.get("current_price", p["entry_price"]) * p["quantity"] for p in positions)

    cash_info = get_available_cash(initial_cap, positions, closed_positions)
    cash_remaining = cash_info["available_cash"]
    unsettled_funds = cash_info["unsettled_funds"]
    total_equity = total_mkt_val + cash_info["total_cash"]
    pnl_pct = ((total_equity - initial_cap) / initial_cap) * 100 if initial_cap > 0 else 0

    return {
        "open_positions": len(positions),
        "total_invested": round(total_invested, 2),
        "cash_remaining": round(cash_remaining, 2),
        "unsettled_funds": round(unsettled_funds, 2),
        "total_equity": round(total_equity, 2),
        "pnl_pct": round(pnl_pct, 2),
        "initial_capital": initial_cap,
    }


def get_dashboard_stats():
    from database import get_mongodb, get_open_positions

    db = get_mongodb()
    col = config.MONGODB_COLLECTIONS["positions"]
    initial_cap = config.TRADING_OPTIONS.get("initial_capital", 100000.0)
    brokerage_pct = config.TRADING_OPTIONS.get("brokerage_charges", 0.0020)

    open_positions = get_open_positions()
    closed_positions = list(db[col].find({"status": "CLOSED"}))

    def pos_cost(p):
        return p.get("quantity", 0) * p.get("entry_price", 0) * (1 + brokerage_pct)

    total_invested = sum(pos_cost(p) for p in open_positions)
    total_mkt_val = sum(p.get("current_price", p["entry_price"]) * p["quantity"] for p in open_positions)

    # Settlement-aware cash calculation
    from utils.settlement import get_available_cash

    cash_info = get_available_cash(initial_cap, open_positions, closed_positions)
    cash_remaining = cash_info["available_cash"]
    unsettled_funds = cash_info["unsettled_funds"]
    total_equity = total_mkt_val + cash_info["total_cash"]

    closed_realized = sum(
        (p.get("exit_price", 0) - p.get("entry_price", 0)) * p.get("quantity", 0) for p in closed_positions
    )
    partial_realized = sum(
        sum(
            (pe.get("price", p.get("entry_price", 0)) - p.get("entry_price", 0)) * pe.get("quantity", 0)
            for pe in p.get("partial_exits", [])
        )
        for p in open_positions
    )
    realized_pnl = closed_realized + partial_realized

    unrealized_pnl = sum(
        (p.get("current_price", p["entry_price"]) - p["entry_price"]) * p["quantity"] for p in open_positions
    )
    total_pnl = realized_pnl + unrealized_pnl
    pnl_pct = (total_pnl / initial_cap) * 100 if initial_cap > 0 else 0
    deployed_pct = (total_invested / initial_cap) * 100 if initial_cap > 0 else 0

    wins = [p for p in closed_positions if (p.get("pnl_pct", 0) or 0) > 0]
    losses = [p for p in closed_positions if (p.get("pnl_pct", 0) or 0) <= 0]
    total_trades = len(closed_positions)
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    avg_win = sum(p.get("pnl_pct", 0) for p in wins) / len(wins) if wins else 0
    avg_loss = sum(p.get("pnl_pct", 0) for p in losses) / len(losses) if losses else 0
    gross_profit = sum((p.get("exit_price", 0) - p.get("entry_price", 0)) * p.get("quantity", 0) for p in wins)
    gross_loss = abs(sum((p.get("exit_price", 0) - p.get("entry_price", 0)) * p.get("quantity", 0) for p in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

    now = datetime.now(timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_logs = list(db.activity_logs.find({"timestamp": {"$gte": today_start}}).sort("timestamp", -1))

    today_actions = {
        "POSITION_OPENED": 0,
        "PYRAMID": 0,
        "TRAIL_SL": 0,
        "TARGET_HIT": 0,
        "CLOSED": 0,
        "PARTIAL_SELL": 0,
    }
    for log in today_logs:
        action = log.get("action", "")
        if action in today_actions:
            today_actions[action] += 1

    sl_hits_today = today_actions["TRAIL_SL"]
    pyramids_today = today_actions["PYRAMID"]
    targets_hit_today = today_actions["TARGET_HIT"]
    trades_today = today_actions["POSITION_OPENED"]
    exits_today = today_actions["CLOSED"]

    recent_logs = list(db.activity_logs.find().sort("timestamp", -1).limit(15))
    activity_feed = []
    for log in recent_logs:
        activity_feed.append(
            {
                "symbol": log.get("symbol", ""),
                "action": log.get("action", ""),
                "timestamp": log.get("timestamp", "").isoformat()
                if hasattr(log.get("timestamp", ""), "isoformat")
                else str(log.get("timestamp", "")),
                "details": log.get("details", {}),
            }
        )

    position_cards = []
    for p in open_positions:
        entry = p.get("entry_price", 0)
        current = p.get("current_price", entry)
        qty = p.get("quantity", 0)
        pnl = (current - entry) * qty
        pnl_pct_pos = ((current - entry) / entry * 100) if entry > 0 else 0
        position_cards.append(
            {
                "symbol": p.get("symbol", ""),
                "quantity": qty,
                "entry_price": round(entry, 2),
                "current_price": round(current, 2),
                "total_investment": round(pos_cost(p), 2),
                "unrealized_pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct_pos, 2),
                "stop_loss": round(p.get("current_stop_loss", p.get("stop_loss", 0)), 2),
                "target": round(p.get("current_target", p.get("target", 0)), 2),
                "adds_count": p.get("adds_count", 0),
                "strategy": p.get("strategy_name", ""),
                "note": p.get("note", ""),
                "reason": p.get("reason", ""),
                "entry_atr": round(p.get("entry_atr", 0), 2),
            }
        )

    return {
        "portfolio": {
            "total_equity": round(total_equity, 2),
            "total_invested": round(total_invested, 2),
            "cash_remaining": round(cash_remaining, 2),
            "unsettled_funds": round(unsettled_funds, 2),
            "deployed_pct": round(deployed_pct, 1),
            "initial_capital": initial_cap,
            "total_pnl": round(total_pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "open_positions": len(open_positions),
            "total_trades": total_trades,
        },
        "performance": {
            "win_rate": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
            "wins": len(wins),
            "losses": len(losses),
            "total_closed": total_trades,
        },
        "today": {
            "trades_opened": trades_today,
            "pyramids_added": pyramids_today,
            "sl_trails": sl_hits_today,
            "targets_hit": targets_hit_today,
            "positions_closed": exits_today,
        },
        "positions": position_cards,
        "activity_feed": activity_feed,
    }


def get_daily_earnings(session_id: str = None, days: int = None, start_date: str = None, end_date: str = None):
    """Retrieve day-by-day PnL earnings report for the dashboard with optional date filtering."""
    from bson import ObjectId

    from database import get_mongodb

    db = get_mongodb()
    query = {}
    if session_id:
        try:
            query["session_id"] = ObjectId(session_id)
        except Exception:
            pass
    else:
        # Default to the most recent backtest session
        latest_sess = db.backtest_sessions.find_one(sort=[("created_at", -1)])
        if latest_sess:
            query["session_id"] = latest_sess["_id"]

    date_filter = {}
    if start_date:
        date_filter["$gte"] = start_date
    if end_date:
        date_filter["$lte"] = end_date
    if date_filter:
        query["date"] = date_filter

    cursor = db.portfolio_backtest_daily_snapshots.find(query).sort("date", 1)
    if days and not (start_date or end_date):
        total_count = db.portfolio_backtest_daily_snapshots.count_documents(query)
        skip = max(0, total_count - days)
        cursor = cursor.skip(skip)

    snaps = list(cursor)
    daily_records = []
    prev_val = None

    for s in snaps:
        curr_val = s.get("portfolio_value", 0.0)
        daily_pnl = (curr_val - prev_val) if prev_val is not None else 0.0
        pnl_pct = ((curr_val - prev_val) / prev_val * 100.0) if prev_val and prev_val > 0 else 0.0
        daily_records.append(
            {
                "date": s.get("date"),
                "portfolio_value": round(curr_val, 2),
                "daily_pnl": round(daily_pnl, 2),
                "daily_pnl_pct": round(pnl_pct, 2),
                "cash_balance": round(s.get("cash_balance", 0.0), 2),
                "open_positions": s.get("open_positions_count", 0),
                "drawdown_pct": round(s.get("drawdown_from_peak_pct", 0.0), 2),
            }
        )
        prev_val = curr_val

    # Summary statistics across the selected range
    total_net_pnl = sum(r["daily_pnl"] for r in daily_records)
    up_days = sum(1 for r in daily_records if r["daily_pnl"] > 0)
    down_days = sum(1 for r in daily_records if r["daily_pnl"] < 0)
    win_rate = (up_days / (up_days + down_days) * 100) if (up_days + down_days) > 0 else 0

    return {
        "daily_records": daily_records,
        "summary": {
            "total_days": len(daily_records),
            "total_net_pnl": round(total_net_pnl, 2),
            "up_days": up_days,
            "down_days": down_days,
            "win_rate": round(win_rate, 1),
            "best_day_pnl": round(max((r["daily_pnl"] for r in daily_records), default=0.0), 2),
            "worst_day_pnl": round(min((r["daily_pnl"] for r in daily_records), default=0.0), 2),
        },
    }
