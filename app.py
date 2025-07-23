from flask import Flask, jsonify, request
from database import init_app, get_db, query_mongodb
from scripts.analyzer import analyze_stock
from models.recommendation import RecommendedShare
from utils.logger import setup_logging
import config
import sqlite3
from datetime import datetime
from typing import Optional

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(config)
    
    # Set up logging
    app.logger = setup_logging()
    
    # Initialize database
    init_app(app)
    
    return app

def get_backtest_cagr_for_symbol(symbol: str) -> Optional[float]:
    """Get the latest backtest CAGR for a given symbol."""
    try:
        from database import get_backtest_results
        # Query the most recent overall backtest result for the symbol
        backtest_results = get_backtest_results(symbol=symbol, period='Overall')
        
        if backtest_results:
            cagr = backtest_results[0]['CAGR']
            return round(float(cagr), 2) if cagr is not None else None
        
        return None
        
    except Exception as e:
        app.logger.error(f"Error fetching backtest CAGR for {symbol}: {e}")
        return None

app = create_app()

@app.route('/')
def index():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "message": "Share Market Analyzer API is running",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/analyze_stock/<symbol>', methods=['GET'])
def analyze_stock_endpoint(symbol):
    """API endpoint to analyze a stock symbol."""
    try:
        analysis_result = analyze_stock(symbol.upper(), app.config)
        return jsonify(analysis_result)
    except Exception as e:
        app.logger.error(f"Error analyzing stock {symbol}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get all stock recommendations with enhanced fields including backtest CAGR."""
    try:
        from database import get_recommended_shares_with_analytics
        recommendations_raw = get_recommended_shares_with_analytics()
        
        recommendations = []
        for rec in recommendations_raw:
            rec_dict = dict(rec)
            # Remove MongoDB _id field if present
            rec_dict.pop('_id', None)
            
            # Convert to RecommendedShare for validation
            rec_obj = RecommendedShare(**rec_dict)
            rec_json = rec_obj.model_dump()
            
            # Add backtest CAGR to the response
            symbol = rec_dict['symbol']
            backtest_cagr = get_backtest_cagr_for_symbol(symbol)
            rec_json['backtest_cagr'] = backtest_cagr
            
            recommendations.append(rec_json)
        
        return jsonify({
            "status": "success",
            "count": len(recommendations),
            "recommendations": recommendations
        })
        
    except Exception as e:
        app.logger.error(f"Error fetching recommendations: {e}")
        return jsonify({
            "status": "error",
            "error": "Failed to fetch recommendations"
        }), 500

@app.route('/test_db', methods=['GET'])
def test_db():
    """Test database connection."""
    try:
        # Test database connection by getting database info
        db = get_db()
        collections = db.list_collection_names()
        return jsonify({
            "status": "success",
            "message": "Database connection successful",
            "database": db.name,
            "collections": collections
        })
    except Exception as e:
        app.logger.error(f"Database test failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/symbols', methods=['GET'])
def get_symbols():
    """Get available NSE symbols."""
    try:
        from scripts.data_fetcher import get_all_nse_symbols
        symbols = get_all_nse_symbols()
        return jsonify({
            "status": "success",
            "count": len(symbols),
            "symbols": symbols
        })
    except Exception as e:
        app.logger.error(f"Error fetching symbols: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/test_data/<symbol>', methods=['GET'])
def test_data(symbol):
    """Test data fetching for a specific symbol."""
    try:
        from scripts.data_fetcher import get_historical_data
        data = get_historical_data(symbol.upper())
        
        if data.empty:
            return jsonify({
                "status": "error",
                "error": f"No data found for symbol {symbol}"
            }), 404
        
        return jsonify({
            "status": "success",
            "symbol": symbol.upper(),
            "data_points": len(data),
            "date_range": {
                "start": data.index[0].strftime('%Y-%m-%d'),
                "end": data.index[-1].strftime('%Y-%m-%d')
            },
            "latest_close": float(data['Close'].iloc[-1])
        })
        
    except Exception as e:
        app.logger.error(f"Error testing data for {symbol}: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
