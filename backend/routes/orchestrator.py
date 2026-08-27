from flask import Blueprint, current_app, jsonify, request

from handlers import orchestrator_handler

orchestrator_bp = Blueprint("orchestrator", __name__)


@orchestrator_bp.route("/run-orchestrator", methods=["POST"])
def run_orchestrator():
    try:
        req_data = request.get_json() or {}
        mode, error = orchestrator_handler.run(req_data)
        if error:
            return jsonify({"status": "error", "error": error}), 409
        return jsonify({"status": "success", "message": f"Trading cycle started ({mode})."})
    except Exception as e:
        current_app.logger.error(f"Orchestrator trigger failure: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@orchestrator_bp.route("/orchestrator-status", methods=["GET"])
def orchestrator_status():
    return jsonify(orchestrator_handler.get_status())


@orchestrator_bp.route("/stream-logs")
def stream_logs():
    return orchestrator_handler.stream_logs()


@orchestrator_bp.route("/pending-exits", methods=["GET"])
def pending_exits():
    """Get all positions awaiting exit price confirmation from the user."""
    try:
        pending = orchestrator_handler.get_pending_exits()
        return jsonify({"status": "success", "count": len(pending), "pending_exits": pending})
    except Exception as e:
        current_app.logger.error(f"Pending exits fetch error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@orchestrator_bp.route("/confirm-exit", methods=["POST"])
def confirm_exit():
    """User confirms or corrects the exit price for a closed position.

    Request body: { "symbol": "UNIONBANK", "exit_price": 184.84 }
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"status": "error", "error": "Request body required"}), 400

        symbol = data.get("symbol")
        exit_price = data.get("exit_price")

        if not symbol:
            return jsonify({"status": "error", "error": "Missing required field: symbol"}), 400
        if exit_price is None:
            return jsonify({"status": "error", "error": "Missing required field: exit_price"}), 400

        try:
            exit_price = float(exit_price)
        except (ValueError, TypeError):
            return jsonify({"status": "error", "error": "exit_price must be a number"}), 400

        result = orchestrator_handler.confirm_exit(symbol, exit_price)
        if result is None:
            return jsonify({"status": "error", "error": f"No pending exit confirmation found for {symbol}"}), 404

        return jsonify({"status": "success", "data": result})
    except Exception as e:
        current_app.logger.error(f"Confirm exit error: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
