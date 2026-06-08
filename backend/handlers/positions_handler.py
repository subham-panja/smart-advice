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


def update_position(symbol: str, data: dict):
    from database import get_position_by_symbol
    from database import update_position as db_update

    existing = get_position_by_symbol(symbol)
    if not existing:
        return None
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
