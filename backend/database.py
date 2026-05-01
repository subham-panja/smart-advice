import logging
from datetime import datetime, timezone

from flask import current_app, g
from pymongo import MongoClient

import config

logger = logging.getLogger(__name__)


def get_db():
    """Flask-specific DB connection."""
    if "db" not in g:
        g.client = MongoClient(current_app.config["MONGODB_HOST"], current_app.config["MONGODB_PORT"])
        g.db = g.client[current_app.config["MONGODB_DATABASE"]]
    return g.db


def get_mongodb():
    """Standalone DB connection."""
    client = MongoClient(config.MONGODB_HOST, config.MONGODB_PORT)
    return client[config.MONGODB_DATABASE]


def _get_db_internal():
    try:
        return get_db()
    except Exception:
        return get_mongodb()


def insert_recommended_share(doc: dict):
    db = _get_db_internal()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    doc["created_at"] = doc.get("created_at", now)
    doc["updated_at"] = now
    col_name = config.MONGODB_COLLECTIONS["recommended_shares"]
    return db[col_name].insert_one(doc)


def get_open_positions():
    col_name = config.MONGODB_COLLECTIONS["positions"]
    return list(_get_db_internal()[col_name].find({"status": "OPEN"}))


def insert_position(doc: dict):
    db = _get_db_internal()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    doc["created_at"] = now
    doc["updated_at"] = now
    doc["status"] = "OPEN"
    col_name = config.MONGODB_COLLECTIONS["positions"]
    return db[col_name].insert_one(doc)


def update_position(symbol: str, update_data: dict):
    db = _get_db_internal()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    update_data["updated_at"] = now
    col_name = config.MONGODB_COLLECTIONS["positions"]
    return db[col_name].update_one({"symbol": symbol, "status": "OPEN"}, {"$set": update_data})


def close_position(symbol: str, exit_price: float, reason: str):
    db = _get_db_internal()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    col_name = config.MONGODB_COLLECTIONS["positions"]
    pos = db[col_name].find_one({"symbol": symbol, "status": "OPEN"})
    if not pos:
        return None
    pnl = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
    return db[col_name].update_one(
        {"_id": pos["_id"]},
        {
            "$set": {
                "status": "CLOSED",
                "exit_price": exit_price,
                "exit_reason": reason,
                "pnl_pct": pnl,
                "exit_date": now,
                "updated_at": now,
            }
        },
    )


def insert_backtest_result(doc: dict):
    db = _get_db_internal()
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    doc["created_at"] = now
    doc["updated_at"] = now
    col_name = config.MONGODB_COLLECTIONS["backtest_results"]
    return db[col_name].insert_one(doc)


def init_app(app):
    """Placeholder for Flask app initialization."""
    pass


def screen_stocks(filters=None):
    """Simplified stock screening."""
    db = _get_db_internal()
    col_name = config.MONGODB_COLLECTIONS["recommended_shares"]
    return list(db[col_name].find(filters or {}).sort("created_at", -1))


def get_recommended_shares_with_analytics():
    col_name = config.MONGODB_COLLECTIONS["recommended_shares"]
    return list(_get_db_internal()[col_name].find().sort("created_at", -1))


def get_backtest_results(symbol=None, period=None):
    db = _get_db_internal()
    q = {}
    if symbol:
        q["symbol"] = symbol
    if period:
        q["period"] = period
    col_name = config.MONGODB_COLLECTIONS["backtest_results"]
    return list(db[col_name].find(q).sort("created_at", -1))


def query_mongodb(collection_name, query_filter=None, projection=None, sort=None, limit=None, one=False):
    db = _get_db_internal()
    col = db[collection_name]
    if one:
        return col.find_one(query_filter or {}, projection)
    cursor = col.find(query_filter or {}, projection)
    if sort:
        cursor = cursor.sort(sort)
    if limit:
        cursor = cursor.limit(limit)
    return list(cursor)
