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
