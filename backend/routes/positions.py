from flask import Blueprint, current_app, jsonify, request

from handlers import positions_handler

positions_bp = Blueprint("positions", __name__)


@positions_bp.route("/positions", methods=["GET"])
def list_positions():
    try:
        status_filter = request.args.get("status")
        positions = positions_handler.list_positions(status_filter)
        return jsonify({"status": "success", "count": len(positions), "positions": positions})
    except Exception as e:
        current_app.logger.error(f"Positions fetch error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@positions_bp.route("/positions", methods=["POST"])
def create_position():
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "error": "Request body required"}), 400

        for field in ("symbol", "quantity", "entry_price"):
            if field not in data:
                return jsonify({"status": "error", "error": f"Missing required field: {field}"}), 400

        result = positions_handler.create_position(data)
        if result is None:
            return jsonify({"status": "error", "error": "Position already exists (duplicate)"}), 409

        return jsonify({"status": "success", "message": f"Position created for {data['symbol']}"})
    except Exception as e:
        current_app.logger.error(f"Position creation error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@positions_bp.route("/positions/<symbol>", methods=["PATCH"])
def update_position(symbol):
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "error": "Request body required"}), 400

        if "symbol" in data:
            return jsonify({"status": "error", "error": "Cannot change symbol"}), 400

        result = positions_handler.update_position(symbol, data)
        if result is None:
            return jsonify({"status": "error", "error": f"No OPEN position found for {symbol}"}), 404

        return jsonify({"status": "success", "message": f"Position updated for {symbol}"})
    except Exception as e:
        current_app.logger.error(f"Position update error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@positions_bp.route("/positions/<symbol>", methods=["DELETE"])
def close_position(symbol):
    try:
        result = positions_handler.close_position(symbol)
        if result is None:
            return jsonify({"status": "error", "error": f"No OPEN position found for {symbol}"}), 404

        return jsonify({"status": "success", "message": f"Position closed for {symbol}"})
    except Exception as e:
        current_app.logger.error(f"Position close error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@positions_bp.route("/activity-logs", methods=["GET"])
def get_activity_logs():
    try:
        from database import get_mongodb
        from handlers.positions_handler import _serialize_value

        db = get_mongodb()
        symbol_filter = request.args.get("symbol")
        limit = int(request.args.get("limit", 100))
        query = {"symbol": symbol_filter} if symbol_filter else {}
        logs = list(db.activity_logs.find(query).sort("timestamp", -1).limit(limit))
        serialized = [{k: _serialize_value(v) for k, v in log.items()} for log in logs]
        return jsonify({"status": "success", "count": len(serialized), "logs": serialized})
    except Exception as e:
        current_app.logger.error(f"Activity logs error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
