import click
from flask import current_app, g
from flask.cli import with_appcontext
from pymongo import MongoClient
from datetime import datetime
import os

def get_db():
    """Get MongoDB database connection from Flask application context."""
    if 'db' not in g:
        client = MongoClient(
            current_app.config['MONGODB_HOST'], 
            current_app.config['MONGODB_PORT']
        )
        g.db = client[current_app.config['MONGODB_DATABASE']]
        g.client = client
    return g.db

def get_mongodb():
    """Get MongoDB database connection for non-Flask contexts (like the analysis script)."""
    import config
    client = MongoClient(config.MONGODB_HOST, config.MONGODB_PORT)
    return client[config.MONGODB_DATABASE]

def close_db(e=None):
    """Close MongoDB database connection."""
    client = g.pop('client', None)
    if client is not None:
        client.close()
    g.pop('db', None)

def init_db():
    """Initialize MongoDB collections with indexes."""
    db = get_db()
    
    # Create collections if they don't exist
    recommended_collection = current_app.config['MONGODB_COLLECTIONS']['recommended_shares']
    backtest_collection = current_app.config['MONGODB_COLLECTIONS']['backtest_results']
    
    # Create indexes for better performance
    db[recommended_collection].create_index("symbol")
    db[recommended_collection].create_index("recommendation_date")
    db[backtest_collection].create_index("symbol")
    db[backtest_collection].create_index("created_at")
    
    current_app.logger.info("MongoDB collections initialized with indexes.")

def query_mongodb(collection_name, query_filter=None, projection=None, sort=None, limit=None, one=False):
    """Query MongoDB collection and return results."""
    db = get_db()
    collection = db[collection_name]
    
    query_filter = query_filter or {}
    
    if one:
        result = collection.find_one(query_filter, projection)
        return result
    else:
        cursor = collection.find(query_filter, projection)
        
        if sort:
            cursor = cursor.sort(sort)
        if limit:
            cursor = cursor.limit(limit)
            
        return list(cursor)

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

import json

def insert_recommended_share(symbol, company_name, technical_score, fundamental_score, 
                             sentiment_score, reason, buy_price=None, sell_price=None, 
                             est_time_to_target=None, backtest_metrics=None):
    """Insert a new recommended share with analytics fields and backtesting metrics as a JSON object."""
    db = get_db()
    collection = db[current_app.config['MONGODB_COLLECTIONS']['recommended_shares']]
    
    document = {
        'symbol': symbol,
        'company_name': company_name,
        'technical_score': technical_score,
        'fundamental_score': fundamental_score,
        'sentiment_score': sentiment_score,
        'reason': reason,
        'buy_price': buy_price,
        'sell_price': sell_price,
        'est_time_to_target': est_time_to_target,
        'backtest_metrics': backtest_metrics,
        'recommendation_date': datetime.now()
    }
    
    collection.insert_one(document)

def update_share_analytics(symbol, buy_price=None, sell_price=None, est_time_to_target=None):
    """Update analytics fields for an existing recommended share."""
    db = get_db()
    collection = db[current_app.config['MONGODB_COLLECTIONS']['recommended_shares']]
    
    update_doc = {}
    
    if buy_price is not None:
        update_doc['buy_price'] = buy_price
    if sell_price is not None:
        update_doc['sell_price'] = sell_price
    if est_time_to_target is not None:
        update_doc['est_time_to_target'] = est_time_to_target
    
    if update_doc:
        collection.update_one(
            {'symbol': symbol},
            {'$set': update_doc}
        )

def insert_backtest_result(symbol, period, cagr, win_rate, max_drawdown, **kwargs):
    """Insert a new backtest result with enhanced fields."""
    db = get_db()
    collection = db[current_app.config['MONGODB_COLLECTIONS']['backtest_results']]
    
    # Create document for MongoDB
    document = {
        'symbol': symbol,
        'period': period,
        'CAGR': cagr,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'total_trades': kwargs.get('total_trades'),
        'winning_trades': kwargs.get('winning_trades'),
        'losing_trades': kwargs.get('losing_trades'),
        'avg_trade_duration': kwargs.get('avg_trade_duration'),
        'avg_profit_per_trade': kwargs.get('avg_profit_per_trade'),
        'avg_loss_per_trade': kwargs.get('avg_loss_per_trade'),
        'largest_win': kwargs.get('largest_win'),
        'largest_loss': kwargs.get('largest_loss'),
        'sharpe_ratio': kwargs.get('sharpe_ratio'),
        'sortino_ratio': kwargs.get('sortino_ratio'),
        'calmar_ratio': kwargs.get('calmar_ratio'),
        'volatility': kwargs.get('volatility'),
        'start_date': kwargs.get('start_date'),
        'end_date': kwargs.get('end_date'),
        'initial_capital': kwargs.get('initial_capital'),
        'final_capital': kwargs.get('final_capital'),
        'total_return': kwargs.get('total_return'),
        'created_at': datetime.now()
    }
    
    collection.insert_one(document)

def get_backtest_results(symbol=None, period=None):
    """Get backtest results with optional filtering."""
    db = get_db()
    collection = db[current_app.config['MONGODB_COLLECTIONS']['backtest_results']]
    
    query_filter = {}
    
    if symbol:
        query_filter['symbol'] = symbol
    if period:
        query_filter['period'] = period
    
    # Return cursor as list, sorted by created_at descending
    results = collection.find(query_filter).sort('created_at', -1)
    return list(results)

def get_recommended_shares_with_analytics():
    """Get all recommended shares including analytics fields."""
    db = get_db()
    collection = db[current_app.config['MONGODB_COLLECTIONS']['recommended_shares']]
    
    # Return cursor as list, sorted by recommendation_date descending
    results = collection.find().sort('recommendation_date', -1)
    return list(results)

def screen_stocks(filters=None):
    """
    Screen stocks based on provided filters.
    
    Args:
        filters (dict): Dictionary of filters like min_price, max_price, min_volume, etc.
        
    Returns:
        list: List of matching stock documents
    """
    db = get_db()
    collection = db[current_app.config['MONGODB_COLLECTIONS']['recommended_shares']]
    
    query = {}
    
    if not filters:
        return list(collection.find().sort('recommendation_date', -1))
        
    # Price filters
    if 'min_price' in filters or 'max_price' in filters:
        price_query = {}
        if 'min_price' in filters:
            price_query['$gte'] = float(filters['min_price'])
        if 'max_price' in filters:
            price_query['$lte'] = float(filters['max_price'])
        # Check against buy_price (current recommended entry)
        query['buy_price'] = price_query

    # Score filters
    if 'min_technical_score' in filters:
        query['technical_score'] = {'$gte': float(filters['min_technical_score'])}
        
    if 'min_fundamental_score' in filters:
        query['fundamental_score'] = {'$gte': float(filters['min_fundamental_score'])}
        
    if 'min_sentiment_score' in filters:
        query['sentiment_score'] = {'$gte': float(filters['min_sentiment_score'])}
        
    if 'min_combined_score' in filters:
        query['combined_score'] = {'$gte': float(filters['min_combined_score'])}

    # Sector filter
    if 'sector' in filters:
        query['sector_analysis.sector'] = filters['sector']

    # RSI filter (nested in detailed_analysis)
    if 'min_rsi' in filters or 'max_rsi' in filters:
        rsi_query = {}
        if 'min_rsi' in filters:
            rsi_query['$gte'] = float(filters['min_rsi'])
        if 'max_rsi' in filters:
            rsi_query['$lte'] = float(filters['max_rsi'])
        query['detailed_analysis.technical.rsi'] = rsi_query
        
    # Volume filter (nested)
    if 'min_volume' in filters:
        # Note: Volume might be in 'detailed_analysis.technical.volume' or similar
        # Depending on how it's stored. Assuming detailed_analysis structure.
        query['detailed_analysis.technical.volume'] = {'$gte': float(filters['min_volume'])}

    # Moving Average Logic (e.g., Price > SMA20)
    # This is harder to query directly if not pre-calculated as a boolean.
    # We'll assume the client filters this or we check against stored SMA values.
    if 'above_sma_20' in filters and filters['above_sma_20'].lower() == 'true':
        # Requires $expr to compare two fields
        query['$expr'] = {'$gt': ['$buy_price', '$detailed_analysis.technical.sma_20']}

    return list(collection.find(query).sort('recommendation_date', -1))

def init_app(app):
    """Register database functions with the Flask app."""
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)
    
    # Register migration command
    from scripts.db_migrate import migrate_db_command
    app.cli.add_command(migrate_db_command)


class DatabaseManager:
    """Database manager class for handling database operations outside Flask context.
    Provides a class-based interface to database functions for use in ML pipeline."""
    
    def __init__(self):
        """Initialize the database manager"""
        # This will be lazy-loaded when needed
        self._db = None
    
    def _get_db(self):
        """Get database connection"""
        if self._db is None:
            self._db = get_mongodb()
        return self._db
    
    def fetch_historical_data(self, symbol=None, start_date=None, end_date=None):
        """Fetch historical price data for a symbol or multiple symbols
        
        Args:
            symbol: Stock symbol to fetch or None for all
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            Dictionary of DataFrames keyed by symbol
        """
        # This is a placeholder implementation
        # In a real application, this would fetch from the price history collection
        return {}
    
    def get_training_symbols(self, min_samples=100, max_symbols=50):
        """Get list of symbols suitable for model training
        
        Args:
            min_samples: Minimum number of samples required per symbol
            max_symbols: Maximum number of symbols to return
            
        Returns:
            List of symbol strings
        """
        # This is a placeholder implementation
        # In a real application, this would query the database for symbols
        # with sufficient historical data
        return []
