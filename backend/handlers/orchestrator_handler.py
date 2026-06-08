import logging
import queue as queue_module
import threading

from flask import Response

from utils.logger import log_queue, set_verbose

logger = logging.getLogger(__name__)

orchestrator_progress = {"status": "idle", "message": ""}


def run(req_data: dict):
    if orchestrator_progress["status"] == "running":
        return None, "Trading cycle already running."

    mode = req_data.get("mode", "live")
    verbose = req_data.get("verbose", False)

    def run_thread():
        try:
            set_verbose(verbose)
            orchestrator_progress["status"] = "running"
            orchestrator_progress["message"] = "Trading cycle started..."

            if mode == "replay":
                from main_orchestrator import run_replay

                days = req_data.get("replay_days", 5)
                run_replay(int(days))
            elif mode == "date":
                from main_orchestrator import run_single_date

                date_str = req_data.get("date")
                if not date_str:
                    raise ValueError("date is required for date mode")
                run_single_date(date_str)
            else:
                from main_orchestrator import run_trading_cycle

                run_trading_cycle()

            orchestrator_progress["status"] = "completed"
            orchestrator_progress["message"] = "Trading cycle completed."
        except Exception as e:
            orchestrator_progress["status"] = "error"
            orchestrator_progress["message"] = str(e)
            logger.error(f"Orchestrator error: {e}")

    thread = threading.Thread(target=run_thread)
    thread.daemon = True
    thread.start()

    return mode, None


def get_status():
    return orchestrator_progress


def stream_logs():
    def generate():
        while True:
            try:
                msg = log_queue.get(timeout=15)
                yield f"data: {msg}\n\n"
            except queue_module.Empty:
                yield ": keep-alive\n\n"

    return Response(generate(), mimetype="text/event-stream")
