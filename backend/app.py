import threading
from datetime import datetime

from flask import Flask, jsonify, request
from flask_cors import CORS

import config
from database import init_app
from scripts.analyzer import analyze_stock
from utils.logger import setup_logging

# Global progress tracking
analysis_progress = {
    "status": "idle",  # idle, running, completed, error
    "progress": 0,
    "total": 0,
    "current_stock": "",
    "recommendations": 0,
    "message": "",
    "start_time": None,
    "verbose": False,
}


def create_app():
    """Create and configure the Flask application strictly."""
    app = Flask(__name__)
    app.config.from_object(config)
    app.secret_key = config.SECRET_KEY

    app.logger = setup_logging()
    init_app(app)
    return app


def get_backtest_cagr_for_symbol(symbol: str) -> float:
    """Get backtest CAGR strictly."""
    from database import get_backtest_results

    results = get_backtest_results(symbol=symbol, period="Overall")
    if not results:
        raise ValueError(f"No backtest results found for {symbol}")
    return round(float(results[0]["CAGR"]), 2)


app = create_app()

CORS(
    app,
    resources={
        r"/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
        }
    },
)


@app.route("/")
def index():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/analyze_stock/<symbol>", methods=["GET"])
def analyze_stock_endpoint(symbol):
    try:
        analysis_result = analyze_stock(symbol.upper(), app.config)
        return jsonify(analysis_result)
    except Exception as e:
        app.logger.error(f"Analysis endpoint failure for {symbol}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    try:
        from database import get_recommended_shares_with_analytics

        recommendations_raw = get_recommended_shares_with_analytics()

        recommendations = []
        for rec in recommendations_raw:
            rec_dict = dict(rec)
            rec_dict.pop("_id", None)

            # Strict mapping - no .get() defaults for core scores
            enhanced_rec = {
                "symbol": rec_dict["symbol"],
                "company_name": rec_dict["company_name"],
                "technical_score": rec_dict["technical_score"],
                "fundamental_score": rec_dict["fundamental_score"],
                "sentiment_score": rec_dict["sentiment_score"],
                "combined_score": rec_dict["combined_score"],
                "is_recommended": rec_dict["is_recommended"],
                "recommendation_strength": rec_dict["recommendation_strength"],
                "reason": rec_dict["reason"],
                "recommendation_date": rec_dict["recommendation_date"],
                "buy_price": rec_dict["buy_price"],
                "sell_price": rec_dict["sell_price"],
                "expected_return_percent": rec_dict["expected_return_percent"],
                "backtest_metrics": rec_dict["backtest_metrics"],
                "detailed_analysis": rec_dict["detailed_analysis"],
                "sector_analysis": rec_dict["sector_analysis"],
                "market_regime": rec_dict["market_regime"],
            }
            recommendations.append(enhanced_rec)

        return jsonify({"status": "success", "count": len(recommendations), "recommendations": recommendations})
    except Exception as e:
        app.logger.error(f"Recommendations fetch failure: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/trigger-analysis", methods=["POST"])
def trigger_analysis():
    try:
        req_data = request.get_json()
        if not req_data:
            raise ValueError("Missing analysis configuration payload.")

        # Mandatory fields
        group_name = req_data["group"]

        from run_analysis import AutomatedStockAnalysis

        analyzer = AutomatedStockAnalysis(verbose=req_data.get("verbose", False))
        analyzer.group_name = group_name

        def run_analysis_thread():
            try:
                with app.app_context():
                    analysis_progress["status"] = "running"
                    analysis_progress["start_time"] = datetime.now()
                    analyzer.run_analysis(
                        max_stocks=req_data.get("max_stocks"), use_all_symbols=req_data.get("all", False)
                    )
                    analysis_progress["status"] = "completed"
            except Exception as e:
                analysis_progress["status"] = "error"
                analysis_progress["message"] = str(e)

        thread = threading.Thread(target=run_analysis_thread)
        thread.daemon = True
        thread.start()

        return jsonify({"status": "success", "message": "Analysis started."})
    except Exception as e:
        app.logger.error(f"Analysis trigger failure: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5001)
