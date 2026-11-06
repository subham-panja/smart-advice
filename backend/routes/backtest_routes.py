"""
Backtest Routes
===============

REST API endpoints for browsing backtest sessions, viewing detailed trade journals,
filtering executions, and inspecting symbol-level performance.
"""

import logging

from flask import Blueprint, jsonify, request

from handlers import backtests_handler

backtests_bp = Blueprint("backtests", __name__)
logger = logging.getLogger(__name__)


@backtests_bp.route("/backtests", methods=["GET"])
def list_sessions():
    """List all backtest sessions."""
    try:
        sessions = backtests_handler.list_backtest_sessions()
        return jsonify(
            {
                "status": "success",
                "count": len(sessions),
                "sessions": sessions,
            }
        )
    except Exception as e:
        logger.error(f"Error listing backtest sessions: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@backtests_bp.route("/backtests/<session_id>", methods=["GET"])
def get_session(session_id: str):
    """Retrieve full details of a specific backtest session."""
    try:
        session = backtests_handler.get_backtest_session(session_id)
        if not session:
            return jsonify({"status": "error", "error": "Backtest session not found"}), 404
        return jsonify(
            {
                "status": "success",
                "session": session,
            }
        )
    except Exception as e:
        logger.error(f"Error fetching backtest session {session_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@backtests_bp.route("/backtests/<session_id>/trades", methods=["GET"])
def get_trades(session_id: str):
    """Query paginated trade executions and journal for a session."""
    try:
        symbol = request.args.get("symbol")
        search = request.args.get("search")
        trade_type = request.args.get("trade_type")
        exit_reason = request.args.get("exit_reason")
        pattern = request.args.get("pattern")
        outcome = request.args.get("outcome")
        page = int(request.args.get("page", 1))
        limit = min(int(request.args.get("limit", 50)), 200)
        sort_by = request.args.get("sort_by", "entry_date")
        sort_dir = request.args.get("sort_dir", "desc")

        result = backtests_handler.get_backtest_trades(
            session_id=session_id,
            symbol=symbol,
            trade_type=trade_type,
            exit_reason=exit_reason,
            pattern=pattern,
            outcome=outcome,
            search=search,
            page=page,
            limit=limit,
            sort_by=sort_by,
            sort_dir=sort_dir,
        )

        return jsonify(
            {
                "status": "success",
                **result,
            }
        )
    except Exception as e:
        logger.error(f"Error fetching trades for session {session_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@backtests_bp.route("/backtests/<session_id>/symbols", methods=["GET"])
def get_symbols(session_id: str):
    """Retrieve performance grouped by stock symbol."""
    try:
        symbols = backtests_handler.get_backtest_symbols(session_id)
        return jsonify(
            {
                "status": "success",
                "count": len(symbols),
                "symbols": symbols,
            }
        )
    except Exception as e:
        logger.error(f"Error fetching symbol breakdown for {session_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@backtests_bp.route("/backtests/<session_id>/yearly", methods=["GET"])
def get_yearly_breakdown(session_id: str):
    """Retrieve year-by-year performance breakdown for a backtest session."""
    try:
        breakdown = backtests_handler.get_backtest_yearly_breakdown(session_id)
        return jsonify(
            {
                "status": "success",
                "count": len(breakdown),
                "breakdown": breakdown,
            }
        )
    except Exception as e:
        logger.error(f"Error fetching yearly breakdown for {session_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@backtests_bp.route("/backtests/<session_id>", methods=["DELETE"])
def delete_session(session_id: str):
    """Delete a backtest session and its trades."""
    try:
        success = backtests_handler.delete_backtest_session(session_id)
        if not success:
            return jsonify({"status": "error", "error": "Session not found or deletion failed"}), 404
        return jsonify(
            {
                "status": "success",
                "message": f"Backtest session {session_id} deleted successfully",
            }
        )
    except Exception as e:
        logger.error(f"Error deleting session {session_id}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@backtests_bp.route("/backtests/run-intraday", methods=["POST"])
def run_intraday_backtest_api():
    """Execute high-speed vectorbt 1-Hour Intraday Ultimate Backtest."""
    try:
        from scripts.vectorbt_intraday_engine import run_intraday_ultimate_backtest

        data = request.get_json(silent=True) or {}
        strategy_name = data.get("strategy", "Intraday_Trading_v1")
        period = data.get("period", "730d")
        symbols = data.get("symbols")

        res = run_intraday_ultimate_backtest(
            strategy_name=strategy_name,
            period=period,
            symbols=symbols,
            save_to_db=True,
        )
        return jsonify(
            {
                "status": "success",
                "message": "Intraday Ultimate Backtest completed",
                "session_id": res.get("session_id"),
                "results": res,
            }
        )
    except Exception as e:
        logger.error(f"Error executing intraday backtest: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
