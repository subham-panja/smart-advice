from flask import Flask, jsonify, request
from flask_cors import CORS
from database import init_app, get_db, query_mongodb
from scripts.analyzer import analyze_stock
from models.recommendation import RecommendedShare
from utils.logger import setup_logging
import config
import sqlite3
from datetime import datetime, timedelta
from typing import Optional
import threading
import time

# Global progress tracking
analysis_progress = {
    'status': 'idle',  # idle, running, completed, error
    'progress': 0,
    'total': 0,
    'current_stock': '',
    'recommendations': 0,
    'message': '',
    'start_time': None,
    'verbose': False
}

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

# Enable CORS for all routes (allow all origins for development)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

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
    """Get all stock recommendations with enhanced fields including detailed analysis data."""
    try:
        from database import get_recommended_shares_with_analytics
        recommendations_raw = get_recommended_shares_with_analytics()
        
        recommendations = []
        for rec in recommendations_raw:
            rec_dict = dict(rec)
            # Remove MongoDB _id field if present
            rec_dict.pop('_id', None)
            
            symbol = rec_dict['symbol']
            
            # Build enhanced recommendation with all available data
            enhanced_rec = {
                # Basic recommendation fields
                'symbol': symbol,
                'company_name': rec_dict.get('company_name', symbol),
                'technical_score': rec_dict.get('technical_score', 0),
                'fundamental_score': rec_dict.get('fundamental_score', 0),
                'sentiment_score': rec_dict.get('sentiment_score', 0),
                'combined_score': rec_dict.get('combined_score', 0),
                'is_recommended': rec_dict.get('is_recommended', False),
                'recommendation_strength': rec_dict.get('recommendation_strength', 'HOLD'),
                'reason': rec_dict.get('reason', ''),
                'recommendation_date': rec_dict.get('recommendation_date'),
                
                # Trade-level fields
                'buy_price': rec_dict.get('buy_price', 0),
                'sell_price': rec_dict.get('sell_price', 0),
                'est_time_to_target': rec_dict.get('est_time_to_target', 'Unknown'),
                'expected_return_percent': rec_dict.get('expected_return_percent', 0),
                
                # Detailed backtest metrics (already structured)
                'backtest_metrics': rec_dict.get('backtest_metrics', {}),
                
                # Add legacy backtest CAGR for compatibility
                'backtest_cagr': None,
                
                # Detailed analysis data
                'detailed_analysis': rec_dict.get('detailed_analysis', {}),
                'sector_analysis': rec_dict.get('sector_analysis', {}), 
                'market_regime': rec_dict.get('market_regime', {}),
                'market_microstructure': rec_dict.get('market_microstructure', {}),
                'alternative_data': rec_dict.get('alternative_data', {}),
                'prediction': rec_dict.get('prediction', {}),
                'rl_action': rec_dict.get('rl_action', {}),
                'tca_analysis': rec_dict.get('tca_analysis', {})
            }
            
            # Extract legacy backtest CAGR for compatibility
            backtest_metrics = enhanced_rec['backtest_metrics']
            if backtest_metrics and isinstance(backtest_metrics, dict):
                enhanced_rec['backtest_cagr'] = backtest_metrics.get('cagr', 0)
            else:
                # Fallback to legacy method
                enhanced_rec['backtest_cagr'] = get_backtest_cagr_for_symbol(symbol)
            
            recommendations.append(enhanced_rec)
        
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

@app.route('/screener', methods=['GET'])
def screen_stocks_endpoint():
    """
    Filter stocks based on criteria.
    Query Params:
    - min_price, max_price
    - min_volume
    - min_rsi, max_rsi
    - min_technical_score, min_fundamental_score, min_sentiment_score, min_combined_score
    - sector
    - above_sma_20 (true/false)
    """
    try:
        from database import screen_stocks
        
        # Convert query params to dict
        filters = request.args.to_dict()
        
        results = screen_stocks(filters)
        
        # Format results similar to recommendations
        formatted_results = []
        for rec in results:
            rec_dict = dict(rec)
            rec_dict.pop('_id', None)
            formatted_results.append(rec_dict)
            
        return jsonify({
            "status": "success",
            "count": len(formatted_results),
            "filters": filters,
            "results": formatted_results
        })
        
    except Exception as e:
        app.logger.error(f"Error screening stocks: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
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

@app.route('/analysis-progress', methods=['GET'])
def get_analysis_progress():
    """Get current analysis progress status."""
    try:
        progress_data = analysis_progress.copy()
        
        # Calculate elapsed time if analysis is running
        if progress_data['start_time']:
            elapsed = datetime.now() - progress_data['start_time']
            progress_data['elapsed_seconds'] = elapsed.total_seconds()
            progress_data['elapsed_time'] = str(elapsed).split('.')[0]  # Remove microseconds
            
            # Estimate remaining time if we have progress
            if progress_data['progress'] > 0 and progress_data['total'] > 0:
                avg_time_per_stock = elapsed.total_seconds() / progress_data['progress']
                remaining_stocks = progress_data['total'] - progress_data['progress']
                estimated_remaining = remaining_stocks * avg_time_per_stock
                progress_data['estimated_remaining_seconds'] = estimated_remaining
                progress_data['estimated_remaining'] = str(timedelta(seconds=int(estimated_remaining)))
        
        # Remove start_time from response as it's not JSON serializable
        progress_data.pop('start_time', None)
        
        return jsonify({
            "status": "success",
            "progress": progress_data
        })
        
    except Exception as e:
        app.logger.error(f"Error getting analysis progress: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/trigger-analysis', methods=['POST'])
def trigger_analysis():
    """Trigger stock analysis with configurable parameters."""
    try:
        # Get parameters from request body
        config = request.get_json() or {}
        
        # Extract analysis configuration
        max_stocks = config.get('max_stocks')
        test_mode = config.get('test', False)
        use_all_symbols = config.get('all', False)
        offline_mode = config.get('offline', False)
        verbose = config.get('verbose', False)
        purge_days = config.get('purge_days')
        disable_volume_filter = config.get('disable_volume_filter', False)
        
        # Import the AutomatedStockAnalysis class
        from run_analysis import AutomatedStockAnalysis
        
        # Create analyzer instance
        analyzer = AutomatedStockAnalysis(verbose=verbose)
        
        # Override config if purge_days is provided
        if purge_days is not None:
            analyzer.app.config['DATA_PURGE_DAYS'] = purge_days
        
        # Set max_stocks for test mode
        if test_mode and max_stocks is None:
            max_stocks = 2
        
        app.logger.info(f"Starting analysis with config: max_stocks={max_stocks}, test={test_mode}, all={use_all_symbols}, offline={offline_mode}, verbose={verbose}")
        
        # Run analysis in a separate thread to avoid blocking
        
        def update_progress_callback(processed, total, recommended, current_stock):
            analysis_progress['status'] = 'running'
            analysis_progress['progress'] = processed
            analysis_progress['total'] = total
            analysis_progress['current_stock'] = current_stock
            analysis_progress['recommendations'] = recommended
            analysis_progress['message'] = f"{processed}/{total} stocks processed"
            if processed == total:
                analysis_progress['status'] = 'completed'

        def run_analysis_thread():
            try:
                with app.app_context():
                    analysis_progress['status'] = 'running'
                    analysis_progress['start_time'] = datetime.now()
                    analysis_progress['verbose'] = verbose

                    # Set the progress callback
                    analyzer.progress_callback = update_progress_callback if not verbose else None

                    analyzer.run_analysis(
                        max_stocks=max_stocks,
                        use_all_symbols=use_all_symbols,
                        offline_mode=offline_mode
                    )
                    app.logger.info("Analysis completed successfully")
                    analysis_progress['status'] = 'completed'
            except Exception as e:
                analysis_progress['status'] = 'error'
                app.logger.error(f"Analysis failed: {e}")
                analysis_progress['message'] = str(e)
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(target=run_analysis_thread)
        analysis_thread.daemon = True
        analysis_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Stock analysis started successfully",
            "config": {
                "max_stocks": max_stocks,
                "test_mode": test_mode,
                "use_all_symbols": use_all_symbols,
                "offline_mode": offline_mode,
                "verbose": verbose,
                "purge_days": purge_days,
                "disable_volume_filter": disable_volume_filter
            }
        })
        
    except Exception as e:
        app.logger.error(f"Error triggering analysis: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
