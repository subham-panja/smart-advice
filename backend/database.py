from pymongo import MongoClient
from datetime import datetime
from flask import g, current_app
import config

def get_db():
    """Flask-specific DB connection."""
    if 'db' not in g:
        g.client = MongoClient(current_app.config['MONGODB_HOST'], current_app.config['MONGODB_PORT'])
        g.db = g.client[current_app.config['MONGODB_DATABASE']]
    return g.db

def get_mongodb():
    """Standalone DB connection."""
    client = MongoClient(config.MONGODB_HOST, config.MONGODB_PORT)
    return client[config.MONGODB_DATABASE]

def _get_db_internal():
    try: return get_db()
    except: return get_mongodb()

def insert_recommended_share(doc: dict):
    db = _get_db_internal()
    doc['recommendation_date'] = datetime.now()
    return db['recommended_shares'].insert_one(doc)

def get_open_positions():
    return list(_get_db_internal()['positions'].find({'status': 'OPEN'}))

def insert_position(doc: dict):
    db = _get_db_internal()
    doc['entry_date'] = doc.get('entry_date', datetime.now())
    doc['status'] = 'OPEN'
    return db['positions'].insert_one(doc)

def update_position(symbol: str, update_data: dict):
    return _get_db_internal()['positions'].update_one({'symbol': symbol, 'status': 'OPEN'}, {'$set': update_data})

def close_position(symbol: str, exit_price: float, reason: str):
    db = _get_db_internal()
    pos = db['positions'].find_one({'symbol': symbol, 'status': 'OPEN'})
    if not pos: return None
    pnl = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100
    return db['positions'].update_one({'_id': pos['_id']}, {'$set': {'status': 'CLOSED', 'exit_price': exit_price, 'exit_reason': reason, 'pnl_pct': pnl, 'exit_date': datetime.now()}})

def insert_backtest_result(doc: dict):
    doc['created_at'] = datetime.now()
    return _get_db_internal()['backtest_results'].insert_one(doc)

def init_app(app):
    """Placeholder for Flask app initialization."""
    pass

def screen_stocks(filters=None):
    """Simplified stock screening."""
    db = _get_db_internal()
    return list(db['recommended_shares'].find(filters or {}).sort('recommendation_date', -1))

def get_recommended_shares_with_analytics():
    return list(_get_db_internal()['recommended_shares'].find().sort('recommendation_date', -1))

def get_backtest_results(symbol=None, period=None):
    db = _get_db_internal()
    q = {}
    if symbol: q['symbol'] = symbol
    if period: q['period'] = period
    return list(db['backtest_results'].find(q).sort('created_at', -1))

def query_mongodb(collection_name, query_filter=None, projection=None, sort=None, limit=None, one=False):
    db = _get_db_internal()
    col = db[collection_name]
    if one: return col.find_one(query_filter or {}, projection)
    cursor = col.find(query_filter or {}, projection)
    if sort: cursor = cursor.sort(sort)
    if limit: cursor = cursor.limit(limit)
    return list(cursor)
