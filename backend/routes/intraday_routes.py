"""
Intraday Trading API Routes
===========================
Endpoints for intraday market status, configuration, live signals, and positions.
"""

from flask import Blueprint, jsonify, request

from handlers import intraday_handler

intraday_bp = Blueprint("intraday", __name__, url_prefix="/intraday")


@intraday_bp.route("/status", methods=["GET"])
def get_intraday_status():
    try:
        return jsonify(intraday_handler.get_status())
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@intraday_bp.route("/config", methods=["GET"])
def get_intraday_config():
    try:
        return jsonify(intraday_handler.get_config())
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@intraday_bp.route("/config", methods=["PATCH"])
def update_intraday_config():
    try:
        payload = request.get_json() or {}
        return jsonify(intraday_handler.update_config(payload))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@intraday_bp.route("/positions", methods=["GET"])
def get_intraday_positions():
    try:
        status_filter = request.args.get("status")
        return jsonify(intraday_handler.get_positions(status_filter=status_filter))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@intraday_bp.route("/positions", methods=["POST"])
def create_intraday_position():
    try:
        data = request.get_json() or {}
        return jsonify(intraday_handler.create_position(data)), 201
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


@intraday_bp.route("/positions/<position_id>/close", methods=["POST"])
def close_intraday_position(position_id: str):
    try:
        data = request.get_json(silent=True) or {}
        reason = data.get("reason", "MANUAL_CLOSE")
        return jsonify(intraday_handler.close_position(position_id, reason=reason))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


@intraday_bp.route("/square-off-all", methods=["POST"])
def square_off_all_positions():
    try:
        return jsonify(intraday_handler.square_off_all())
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@intraday_bp.route("/signals", methods=["GET"])
def get_intraday_signals():
    try:
        simulate = request.args.get("simulate", "false").lower() in ("true", "1")
        return jsonify(intraday_handler.get_signals(allow_simulate=simulate))
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
