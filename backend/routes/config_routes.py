from flask import Blueprint, current_app, jsonify

from handlers import config_handler

config_bp = Blueprint("config_routes", __name__)


@config_bp.route("/strategies", methods=["GET"])
def list_strategies():
    try:
        strategies = config_handler.get_strategies()
        return jsonify({"status": "success", "count": len(strategies), "strategies": strategies})
    except Exception as e:
        current_app.logger.error(f"Strategies fetch error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@config_bp.route("/settings/trading", methods=["GET"])
def get_trading_config():
    return jsonify({"status": "success", "config": config_handler.get_trading_config()})


@config_bp.route("/cycle-stats", methods=["GET"])
def cycle_stats():
    try:
        stats = config_handler.get_cycle_stats()
        return jsonify({"status": "success", **stats})
    except Exception as e:
        current_app.logger.error(f"Cycle stats error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
