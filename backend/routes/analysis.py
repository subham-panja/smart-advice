from flask import Blueprint, current_app, jsonify, request

from handlers import analysis_handler

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/")
def index():
    return jsonify(analysis_handler.health_check())


@analysis_bp.route("/analyze_stock/<symbol>", methods=["GET"])
def analyze_stock_endpoint(symbol):
    try:
        result = analysis_handler.analyze(symbol, current_app.config)
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Analysis endpoint failure for {symbol}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@analysis_bp.route("/recommendations", methods=["GET"])
def get_recommendations():
    try:
        recs = analysis_handler.get_recommendations()
        return jsonify({"status": "success", "count": len(recs), "recommendations": recs})
    except Exception as e:
        current_app.logger.error(f"Recommendations fetch failure: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@analysis_bp.route("/trigger-analysis", methods=["POST"])
def trigger_analysis():
    import threading

    try:
        req_data = request.get_json()
        if not req_data:
            raise ValueError("Missing analysis configuration payload.")

        req_data["group"]  # mandatory field check

        thread = threading.Thread(
            target=analysis_handler.run_analysis_thread,
            args=(req_data, current_app.app_context()),
        )
        thread.daemon = True
        thread.start()

        return jsonify({"status": "success", "message": "Analysis started."})
    except Exception as e:
        current_app.logger.error(f"Analysis trigger failure: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
