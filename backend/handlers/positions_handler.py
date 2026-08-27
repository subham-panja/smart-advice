import logging
from datetime import datetime

from bson import ObjectId

import config

logger = logging.getLogger(__name__)


def _serialize_value(v):
    if isinstance(v, ObjectId):
        return str(v)
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, dict):
        return {k: _serialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    return v


def serialize_position(pos):
    if not pos:
        return None
    return {k: _serialize_value(v) for k, v in pos.items()}


def list_positions(status_filter=None):
    from database import get_all_positions

    positions = get_all_positions(status_filter)
    return [serialize_position(p) for p in positions]


def create_position(data: dict):
    from database import insert_position

    if "total_investment" not in data:
        data["total_investment"] = round(data["quantity"] * data["entry_price"], 2)

    if "stop_loss" in data:
        data.setdefault("current_stop_loss", data["stop_loss"])
    if "target" in data:
        data.setdefault("current_target", data["target"])
    data.setdefault("strategy_name", "Manual")
    data.setdefault("adds_count", 0)
    data.setdefault("trade_type", "LONG_BUY")
    data.setdefault("is_paper", config.TRADING_OPTIONS.get("is_paper_trading", True))

    return insert_position(data)


def _get_original_position_metrics(symbol: str, existing: dict):
    try:
        from database import get_mongodb

        db = get_mongodb()
        log = db.activity_logs.find_one({"symbol": symbol, "action": "POSITION_OPENED"})
        if log and "details" in log:
            d = log["details"]
            orig_entry = d.get("entry_price", existing.get("entry_price"))
            orig_sl = d.get("stop_loss", existing.get("stop_loss"))
            orig_target = d.get("target", existing.get("target"))
            return orig_entry, orig_sl, orig_target
    except Exception as e:
        logger.warning(f"Failed to fetch POSITION_OPENED log for {symbol}: {e}")

    return existing.get("entry_price"), existing.get("stop_loss"), existing.get("target")


def update_position(symbol: str, data: dict):
    import config
    from database import get_position_by_symbol
    from database import update_position as db_update

    existing = get_position_by_symbol(symbol)
    if not existing:
        return None

    if "entry_price" in data:
        new_entry = float(data["entry_price"])
        data["entry_price"] = new_entry
        qty = data.get("quantity", existing.get("quantity", 1))

        # Recalculate total investment with brokerage (0.20%)
        if "total_investment" not in data:
            brokerage_pct = config.TRADING_OPTIONS.get("brokerage_charges", 0.0020)
            data["total_investment"] = round(qty * new_entry * (1 + brokerage_pct), 2)

        # Get original position metrics to accurately derive risk & target distances
        orig_entry, orig_sl, orig_target = _get_original_position_metrics(symbol, existing)

        if orig_entry and orig_sl and orig_target:
            sl_dist = float(orig_entry) - float(orig_sl)
            target_dist = float(orig_target) - float(orig_entry)

            if "stop_loss" not in data:
                new_sl = round(new_entry - sl_dist, 2)
                data["stop_loss"] = new_sl
                data["current_stop_loss"] = new_sl

            if "target" not in data:
                new_target = round(new_entry + target_dist, 2)
                data["target"] = new_target
                data["current_target"] = new_target

        logger.info(
            f"Entry correction for {symbol}: "
            f"₹{existing.get('entry_price', 0):.2f} → ₹{new_entry:.2f}, "
            f"new investment: ₹{data.get('total_investment', 0):.2f}, "
            f"new SL: ₹{data.get('current_stop_loss', 0):.2f}, "
            f"new Target: ₹{data.get('current_target', 0):.2f}"
        )

    db_update(symbol, data)
    return True


def close_position(symbol: str):
    from database import close_position as db_close
    from database import get_position_by_symbol

    existing = get_position_by_symbol(symbol)
    if not existing:
        return None
    current_price = existing.get("current_price", existing["entry_price"])
    db_close(symbol, current_price, "MANUAL_CLOSE")
    return True
