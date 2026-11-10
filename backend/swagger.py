"""
Smart Advice Trading API — Flask-RESTX Swagger & OpenAPI Integration
=====================================================================
Comprehensive OpenAPI documentation and interactive Swagger UI at /api/docs.
Provides 6 institutional namespaces covering real-time technical analysis,
portfolio positions, automated trade orchestration, trading configuration,
vectorized backtesting, and intraday execution.
"""

import logging
import threading

from flask import current_app, request
from flask_restx import Resource

from handlers import (
    analysis_handler,
    backtests_handler,
    config_handler,
    intraday_handler,
    orchestrator_handler,
    positions_handler,
)
from swagger_base import (
    analysis_ns,
    api,
    backtests_ns,
    config_ns,
    intraday_ns,
    orchestrator_ns,
    positions_ns,
    swagger_bp,
)
from swagger_models import (
    activity_logs_parser,
    activity_logs_response_model,
    backtest_sessions_response_model,
    backtest_trades_response_model,
    candles_parser,
    candles_response_model,
    close_intraday_position_model,
    confirm_exit_model,
    create_intraday_position_model,
    create_position_model,
    cycle_stats_model,
    daily_earnings_parser,
    daily_earnings_response_model,
    dashboard_stats_model,
    error_response_model,
    health_model,
    intraday_config_model,
    intraday_positions_parser,
    intraday_signals_parser,
    intraday_status_model,
    message_response_model,
    orchestrator_status_model,
    pending_exits_response_model,
    positions_query_parser,
    positions_response_model,
    recommendations_response_model,
    run_intraday_backtest_model,
    run_orchestrator_model,
    strategies_response_model,
    trades_query_parser,
    trading_settings_response_model,
    trigger_analysis_model,
    update_intraday_config_model,
    update_position_model,
    update_recommendation_model,
)

logger = logging.getLogger(__name__)

__all__ = ["swagger_bp", "api"]


# ===========================================================================
# 1. ANALYSIS NAMESPACE RESOURCES (6 Endpoints)
# ===========================================================================


@analysis_ns.route("/")
@analysis_ns.route("/health")
class AnalysisHealthResource(Resource):
    @analysis_ns.doc(summary="Backend health check", description="Verify the status of the trading engine")
    @analysis_ns.response(200, "Healthy", health_model)
    def get(self):
        """System health check and server timestamp."""
        return analysis_handler.health_check()


@analysis_ns.route("/stock/<string:symbol>")
@analysis_ns.route("/analyze_stock/<string:symbol>")
@analysis_ns.param("symbol", "Stock ticker symbol (e.g. RELIANCE, TCS)")
class StockAnalysisResource(Resource):
    @analysis_ns.doc(
        summary="Analyze single stock", description="Run full technical and fundamental analysis on a stock symbol"
    )
    @analysis_ns.response(200, "Analysis completed")
    @analysis_ns.response(500, "Analysis error", error_response_model)
    def get(self, symbol):
        """Execute algorithmic analysis for an equity symbol."""
        try:
            result = analysis_handler.analyze(symbol, current_app.config)
            return result
        except Exception as e:
            current_app.logger.error(f"Analysis endpoint failure for {symbol}: {e}")
            return {"status": "error", "error": str(e)}, 500


@analysis_ns.route("/recommendations")
class RecommendationsListResource(Resource):
    @analysis_ns.doc(
        summary="Get stock recommendations",
        description="Retrieve all current system-generated recommendations with analytical scores",
    )
    @analysis_ns.response(200, "Success", recommendations_response_model)
    @analysis_ns.response(500, "Server error", error_response_model)
    def get(self):
        """List active algorithmic stock recommendations."""
        try:
            recs = analysis_handler.get_recommendations()
            return {"status": "success", "count": len(recs), "recommendations": recs}
        except Exception as e:
            current_app.logger.error(f"Recommendations fetch failure: {e}")
            return {"status": "error", "error": str(e)}, 500


@analysis_ns.route("/recommendations/<string:symbol>")
@analysis_ns.param("symbol", "Stock ticker symbol to update")
class RecommendationDetailResource(Resource):
    @analysis_ns.doc(
        summary="Update recommendation", description="Update rating, notes, or target prices of a recommendation"
    )
    @analysis_ns.expect(update_recommendation_model)
    @analysis_ns.response(200, "Updated", message_response_model)
    @analysis_ns.response(404, "Not found", error_response_model)
    @analysis_ns.response(500, "Server error", error_response_model)
    def patch(self, symbol):
        """Update recommendation properties."""
        try:
            data = request.get_json() or {}
            res = analysis_handler.update_recommendation(symbol, data)
            if not res:
                return {"status": "error", "error": f"Recommendation for {symbol} not found"}, 404
            return {"status": "success", "message": f"Recommendation {symbol} updated"}
        except Exception as e:
            current_app.logger.error(f"Recommendation update failure for {symbol}: {e}")
            return {"status": "error", "error": str(e)}, 500


@analysis_ns.route("/stock-candles/<string:symbol>")
@analysis_ns.param("symbol", "Stock ticker symbol (e.g. RELIANCE)")
class StockCandlesResource(Resource):
    @analysis_ns.doc(
        summary="Historical candlestick data", description="Retrieve OHLCV candlestick records for charting"
    )
    @analysis_ns.expect(candles_parser)
    @analysis_ns.response(200, "Success", candles_response_model)
    @analysis_ns.response(404, "Data not found", error_response_model)
    @analysis_ns.response(500, "Server error", error_response_model)
    def get(self, symbol):
        """Fetch historical daily candles for a stock."""
        try:
            args = candles_parser.parse_args()
            days = args.get("days", 35)
            res = analysis_handler.get_stock_candles(symbol, days=days)
            if not res:
                return {"status": "error", "error": f"No candle data for {symbol}"}, 404
            return {"status": "success", **res}
        except Exception as e:
            current_app.logger.error(f"Stock candles fetch error for {symbol}: {e}")
            return {"status": "error", "error": str(e)}, 500


@analysis_ns.route("/trigger-analysis")
class TriggerAnalysisResource(Resource):
    @analysis_ns.doc(
        summary="Trigger batch analysis", description="Run asynchronous background analysis on a watchlist group"
    )
    @analysis_ns.expect(trigger_analysis_model)
    @analysis_ns.response(200, "Started", message_response_model)
    @analysis_ns.response(400, "Bad request", error_response_model)
    @analysis_ns.response(500, "Server error", error_response_model)
    def post(self):
        """Start background technical analysis scan."""
        try:
            req_data = request.get_json()
            if not req_data or "group" not in req_data:
                return {"status": "error", "error": "Missing mandatory field 'group' in payload."}, 400

            thread = threading.Thread(
                target=analysis_handler.run_analysis_thread,
                args=(req_data, current_app.app_context()),
            )
            thread.daemon = True
            thread.start()
            return {"status": "success", "message": "Analysis started."}
        except Exception as e:
            current_app.logger.error(f"Analysis trigger failure: {e}")
            return {"status": "error", "error": str(e)}, 500


# ===========================================================================
# 2. POSITIONS NAMESPACE RESOURCES (5 Endpoints)
# ===========================================================================


@positions_ns.route("")
@positions_ns.route("/")
class PositionsListResource(Resource):
    @positions_ns.doc(summary="List swing positions", description="Retrieve active or closed swing trading positions")
    @positions_ns.expect(positions_query_parser)
    @positions_ns.response(200, "Success", positions_response_model)
    @positions_ns.response(500, "Server error", error_response_model)
    def get(self):
        """List swing positions optionally filtered by status."""
        try:
            args = positions_query_parser.parse_args()
            status_filter = args.get("status")
            positions = positions_handler.list_positions(status_filter)
            return {"status": "success", "count": len(positions), "positions": positions}
        except Exception as e:
            current_app.logger.error(f"Positions fetch error: {e}")
            return {"status": "error", "error": str(e)}, 500

    @positions_ns.doc(summary="Create swing position", description="Manually open a new swing trading position")
    @positions_ns.expect(create_position_model)
    @positions_ns.response(200, "Created", message_response_model)
    @positions_ns.response(400, "Bad request", error_response_model)
    @positions_ns.response(409, "Conflict", error_response_model)
    @positions_ns.response(500, "Server error", error_response_model)
    def post(self):
        """Open a new swing position."""
        try:
            data = request.get_json(force=True, silent=True)
            if not data:
                return {"status": "error", "error": "Request body required"}, 400

            for field in ("symbol", "quantity", "entry_price"):
                if field not in data:
                    return {"status": "error", "error": f"Missing required field: {field}"}, 400

            result = positions_handler.create_position(data)
            if result is None:
                return {"status": "error", "error": "Position already exists (duplicate)"}, 409

            return {"status": "success", "message": f"Position created for {data['symbol']}"}
        except Exception as e:
            current_app.logger.error(f"Position creation error: {e}")
            return {"status": "error", "error": str(e)}, 500


@positions_ns.route("/<string:symbol>")
@positions_ns.param("symbol", "Stock ticker symbol")
class PositionDetailResource(Resource):
    @positions_ns.doc(
        summary="Update position", description="Update entry price, stop-loss, target, or quantity of an open position"
    )
    @positions_ns.expect(update_position_model)
    @positions_ns.response(200, "Updated", message_response_model)
    @positions_ns.response(400, "Bad request", error_response_model)
    @positions_ns.response(404, "Not found", error_response_model)
    @positions_ns.response(500, "Server error", error_response_model)
    def patch(self, symbol):
        """Update active position parameters."""
        try:
            data = request.get_json(force=True, silent=True)
            if not data:
                return {"status": "error", "error": "Request body required"}, 400

            if "symbol" in data:
                return {"status": "error", "error": "Cannot change symbol"}, 400

            result = positions_handler.update_position(symbol, data)
            if result is None:
                return {"status": "error", "error": f"No OPEN position found for {symbol}"}, 404

            return {"status": "success", "message": f"Position updated for {symbol}"}
        except Exception as e:
            current_app.logger.error(f"Position update error: {e}")
            return {"status": "error", "error": str(e)}, 500

    @positions_ns.doc(summary="Close position", description="Close an active swing trading position")
    @positions_ns.response(200, "Closed", message_response_model)
    @positions_ns.response(404, "Not found", error_response_model)
    @positions_ns.response(500, "Server error", error_response_model)
    def delete(self, symbol):
        """Close an open swing position."""
        try:
            result = positions_handler.close_position(symbol)
            if result is None:
                return {"status": "error", "error": f"No OPEN position found for {symbol}"}, 404

            return {"status": "success", "message": f"Position closed for {symbol}"}
        except Exception as e:
            current_app.logger.error(f"Position close error: {e}")
            return {"status": "error", "error": str(e)}, 500


@positions_ns.route("/activity-logs")
class ActivityLogsResource(Resource):
    @positions_ns.doc(summary="Get activity logs", description="Retrieve position audit trails and transition logs")
    @positions_ns.expect(activity_logs_parser)
    @positions_ns.response(200, "Success", activity_logs_response_model)
    @positions_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Retrieve system activity and trading execution logs."""
        try:
            args = activity_logs_parser.parse_args()
            symbol_filter = args.get("symbol")
            limit = args.get("limit", 100)
            logs = positions_handler.get_activity_logs(symbol_filter=symbol_filter, limit=limit)
            return {"status": "success", "count": len(logs), "logs": logs}
        except Exception as e:
            current_app.logger.error(f"Activity logs error: {e}")
            return {"status": "error", "error": str(e)}, 500


# ===========================================================================
# 3. ORCHESTRATOR NAMESPACE RESOURCES (5 Endpoints)
# ===========================================================================


@orchestrator_ns.route("/run")
@orchestrator_ns.route("/run-orchestrator")
class RunOrchestratorResource(Resource):
    @orchestrator_ns.doc(
        summary="Run orchestrator cycle", description="Trigger an automated market analysis and trade execution cycle"
    )
    @orchestrator_ns.expect(run_orchestrator_model)
    @orchestrator_ns.response(200, "Started", message_response_model)
    @orchestrator_ns.response(409, "Conflict (already running)", error_response_model)
    @orchestrator_ns.response(500, "Server error", error_response_model)
    def post(self):
        """Start trading cycle orchestration."""
        try:
            req_data = request.get_json() or {}
            mode, error = orchestrator_handler.run(req_data)
            if error:
                return {"status": "error", "error": error}, 409
            return {"status": "success", "message": f"Trading cycle started ({mode})."}
        except Exception as e:
            current_app.logger.error(f"Orchestrator trigger failure: {e}")
            return {"status": "error", "error": str(e)}, 500


@orchestrator_ns.route("/status")
@orchestrator_ns.route("/orchestrator-status")
class OrchestratorStatusResource(Resource):
    @orchestrator_ns.doc(
        summary="Orchestrator status", description="Get current execution state of the trading orchestrator"
    )
    @orchestrator_ns.response(200, "Success", orchestrator_status_model)
    def get(self):
        """Get current orchestrator status and message."""
        return orchestrator_handler.get_status()


@orchestrator_ns.route("/stream-logs")
class StreamLogsResource(Resource):
    @orchestrator_ns.doc(
        summary="Stream live logs",
        description="Real-time Server-Sent Events (SSE) log stream from active trading cycles",
    )
    def get(self):
        """Stream trading cycle logs via Server-Sent Events."""
        return orchestrator_handler.stream_logs()


@orchestrator_ns.route("/pending-exits")
class PendingExitsResource(Resource):
    @orchestrator_ns.doc(
        summary="List pending exit confirmations",
        description="Positions waiting for trader confirmation of execution exit price",
    )
    @orchestrator_ns.response(200, "Success", pending_exits_response_model)
    @orchestrator_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Get positions pending exit price confirmation."""
        try:
            pending = orchestrator_handler.get_pending_exits()
            return {"status": "success", "count": len(pending), "pending_exits": pending}
        except Exception as e:
            current_app.logger.error(f"Pending exits fetch error: {e}")
            return {"status": "error", "error": str(e)}, 500


@orchestrator_ns.route("/confirm-exit")
class ConfirmExitResource(Resource):
    @orchestrator_ns.doc(
        summary="Confirm position exit price",
        description="Confirm or update the actual exit execution price for a closed position",
    )
    @orchestrator_ns.expect(confirm_exit_model)
    @orchestrator_ns.response(200, "Success")
    @orchestrator_ns.response(400, "Bad request", error_response_model)
    @orchestrator_ns.response(404, "Not found", error_response_model)
    @orchestrator_ns.response(500, "Server error", error_response_model)
    def post(self):
        """Confirm or correct exit price for a position."""
        try:
            data = request.get_json(force=True, silent=True)
            if not data:
                return {"status": "error", "error": "Request body required"}, 400

            symbol = data.get("symbol")
            exit_price = data.get("exit_price")

            if not symbol:
                return {"status": "error", "error": "Missing required field: symbol"}, 400
            if exit_price is None:
                return {"status": "error", "error": "Missing required field: exit_price"}, 400

            try:
                exit_price = float(exit_price)
            except (ValueError, TypeError):
                return {"status": "error", "error": "exit_price must be a number"}, 400

            result = orchestrator_handler.confirm_exit(symbol, exit_price)
            if result is None:
                return {"status": "error", "error": f"No pending exit confirmation found for {symbol}"}, 404

            return {"status": "success", "data": result}
        except Exception as e:
            current_app.logger.error(f"Confirm exit error: {e}")
            return {"status": "error", "error": str(e)}, 500


# ===========================================================================
# 4. CONFIG NAMESPACE RESOURCES (5 Endpoints)
# ===========================================================================


@config_ns.route("/strategies")
class StrategiesResource(Resource):
    @config_ns.doc(
        summary="List trading strategies", description="Retrieve all configured strategies (swing, intraday, portfolio)"
    )
    @config_ns.response(200, "Success", strategies_response_model)
    @config_ns.response(500, "Server error", error_response_model)
    def get(self):
        """List registered JSON-configured trading strategies."""
        try:
            strategies = config_handler.get_strategies()
            return {"status": "success", "count": len(strategies), "strategies": strategies}
        except Exception as e:
            current_app.logger.error(f"Strategies fetch error: {e}")
            return {"status": "error", "error": str(e)}, 500


@config_ns.route("/settings/trading")
@config_ns.route("/trading-settings")
class TradingSettingsResource(Resource):
    @config_ns.doc(
        summary="Trading configuration",
        description="Get current global trading settings, capital allocation, and risk limits",
    )
    @config_ns.response(200, "Success", trading_settings_response_model)
    def get(self):
        """Get active trading settings."""
        return {"status": "success", "config": config_handler.get_trading_config()}


@config_ns.route("/cycle-stats")
class CycleStatsResource(Resource):
    @config_ns.doc(
        summary="Trading cycle statistics",
        description="Get active cycle capital stats, unsettled funds, and equity values",
    )
    @config_ns.response(200, "Success", cycle_stats_model)
    @config_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Get current cycle financial statistics."""
        try:
            stats = config_handler.get_cycle_stats()
            return {"status": "success", **stats}
        except Exception as e:
            current_app.logger.error(f"Cycle stats error: {e}")
            return {"status": "error", "error": str(e)}, 500


@config_ns.route("/dashboard-stats")
class DashboardStatsResource(Resource):
    @config_ns.doc(
        summary="Dashboard overview metrics",
        description="Aggregated metrics for portfolio balance, win rates, and active positions",
    )
    @config_ns.response(200, "Success", dashboard_stats_model)
    @config_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Get aggregated dashboard metrics."""
        try:
            stats = config_handler.get_dashboard_stats()
            return {"status": "success", **stats}
        except Exception as e:
            current_app.logger.error(f"Dashboard stats error: {e}")
            return {"status": "error", "error": str(e)}, 500


@config_ns.route("/daily-earnings")
class DailyEarningsResource(Resource):
    @config_ns.doc(
        summary="Daily earnings breakdown", description="Historical daily realized P&L and earnings breakdown"
    )
    @config_ns.expect(daily_earnings_parser)
    @config_ns.response(200, "Success", daily_earnings_response_model)
    @config_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Retrieve daily earnings breakdown."""
        try:
            args = daily_earnings_parser.parse_args()
            data = config_handler.get_daily_earnings(
                session_id=args.get("session_id"),
                days=args.get("days"),
                start_date=args.get("start_date"),
                end_date=args.get("end_date"),
            )
            return {"status": "success", **data}
        except Exception as e:
            current_app.logger.error(f"Daily earnings error: {e}")
            return {"status": "error", "error": str(e)}, 500


# ===========================================================================
# 5. BACKTESTS NAMESPACE RESOURCES (7 Endpoints)
# ===========================================================================


@backtests_ns.route("")
@backtests_ns.route("/")
class BacktestSessionsListResource(Resource):
    @backtests_ns.doc(
        summary="List backtest sessions", description="Retrieve all historical backtest sessions with summary metrics"
    )
    @backtests_ns.response(200, "Success", backtest_sessions_response_model)
    @backtests_ns.response(500, "Server error", error_response_model)
    def get(self):
        """List all completed backtest sessions."""
        try:
            sessions = backtests_handler.list_backtest_sessions()
            return {"status": "success", "count": len(sessions), "sessions": sessions}
        except Exception as e:
            logger.error(f"Error listing backtest sessions: {e}")
            return {"status": "error", "error": str(e)}, 500


@backtests_ns.route("/<string:session_id>")
@backtests_ns.param("session_id", "Backtest session unique identifier")
class BacktestSessionDetailResource(Resource):
    @backtests_ns.doc(
        summary="Get backtest session details",
        description="Retrieve full configuration and comprehensive metrics of a session",
    )
    @backtests_ns.response(200, "Success")
    @backtests_ns.response(404, "Not found", error_response_model)
    @backtests_ns.response(500, "Server error", error_response_model)
    def get(self, session_id):
        """Retrieve backtest session details."""
        try:
            session = backtests_handler.get_backtest_session(session_id)
            if not session:
                return {"status": "error", "error": "Backtest session not found"}, 404
            return {"status": "success", "session": session}
        except Exception as e:
            logger.error(f"Error fetching backtest session {session_id}: {e}")
            return {"status": "error", "error": str(e)}, 500

    @backtests_ns.doc(
        summary="Delete backtest session", description="Delete a backtest session and all of its associated trade logs"
    )
    @backtests_ns.response(200, "Deleted", message_response_model)
    @backtests_ns.response(404, "Not found", error_response_model)
    @backtests_ns.response(500, "Server error", error_response_model)
    def delete(self, session_id):
        """Delete backtest session."""
        try:
            success = backtests_handler.delete_backtest_session(session_id)
            if not success:
                return {"status": "error", "error": "Session not found or deletion failed"}, 404
            return {"status": "success", "message": f"Backtest session {session_id} deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return {"status": "error", "error": str(e)}, 500


@backtests_ns.route("/<string:session_id>/trades")
@backtests_ns.param("session_id", "Backtest session unique identifier")
class BacktestTradesResource(Resource):
    @backtests_ns.doc(
        summary="Query trade executions journal",
        description="Paginated trade records with filters for symbol, outcome, pattern, reason",
    )
    @backtests_ns.expect(trades_query_parser)
    @backtests_ns.response(200, "Success", backtest_trades_response_model)
    @backtests_ns.response(500, "Server error", error_response_model)
    def get(self, session_id):
        """Get paginated backtest trades journal."""
        try:
            args = trades_query_parser.parse_args()
            limit = min(args.get("limit", 50), 200)
            result = backtests_handler.get_backtest_trades(
                session_id=session_id,
                symbol=args.get("symbol"),
                trade_type=args.get("trade_type"),
                exit_reason=args.get("exit_reason"),
                pattern=args.get("pattern"),
                outcome=args.get("outcome"),
                search=args.get("search"),
                page=args.get("page", 1),
                limit=limit,
                sort_by=args.get("sort_by", "entry_date"),
                sort_dir=args.get("sort_dir", "desc"),
            )
            return {"status": "success", **result}
        except Exception as e:
            logger.error(f"Error fetching trades for session {session_id}: {e}")
            return {"status": "error", "error": str(e)}, 500


@backtests_ns.route("/<string:session_id>/symbols")
@backtests_ns.param("session_id", "Backtest session unique identifier")
class BacktestSymbolsResource(Resource):
    @backtests_ns.doc(
        summary="Symbol performance breakdown", description="Performance metrics grouped per traded stock symbol"
    )
    @backtests_ns.response(200, "Success")
    @backtests_ns.response(500, "Server error", error_response_model)
    def get(self, session_id):
        """Retrieve symbol-level performance breakdown."""
        try:
            symbols = backtests_handler.get_backtest_symbols(session_id)
            return {"status": "success", "count": len(symbols), "symbols": symbols}
        except Exception as e:
            logger.error(f"Error fetching symbol breakdown for {session_id}: {e}")
            return {"status": "error", "error": str(e)}, 500


@backtests_ns.route("/<string:session_id>/yearly")
@backtests_ns.param("session_id", "Backtest session unique identifier")
class BacktestYearlyResource(Resource):
    @backtests_ns.doc(
        summary="Yearly performance breakdown", description="Annualized returns, win rates, and drawdown breakdown"
    )
    @backtests_ns.response(200, "Success")
    @backtests_ns.response(500, "Server error", error_response_model)
    def get(self, session_id):
        """Retrieve year-by-year performance metrics."""
        try:
            breakdown = backtests_handler.get_backtest_yearly_breakdown(session_id)
            return {"status": "success", "count": len(breakdown), "breakdown": breakdown}
        except Exception as e:
            logger.error(f"Error fetching yearly breakdown for {session_id}: {e}")
            return {"status": "error", "error": str(e)}, 500


@backtests_ns.route("/run-intraday")
class RunIntradayBacktestResource(Resource):
    @backtests_ns.doc(
        summary="Execute Intraday Ultimate Backtest",
        description="Run high-speed vectorbt 1-Hour Intraday simulation across symbols",
    )
    @backtests_ns.expect(run_intraday_backtest_model)
    @backtests_ns.response(200, "Completed successfully")
    @backtests_ns.response(500, "Server error", error_response_model)
    def post(self):
        """Run vectorized intraday backtest."""
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
            return {
                "status": "success",
                "message": "Intraday Ultimate Backtest completed",
                "session_id": res.get("session_id"),
                "results": res,
            }
        except Exception as e:
            logger.error(f"Error executing intraday backtest: {e}")
            return {"status": "error", "error": str(e)}, 500


# ===========================================================================
# 6. INTRADAY NAMESPACE RESOURCES (8 Endpoints)
# ===========================================================================


@intraday_ns.route("/status")
class IntradayStatusResource(Resource):
    @intraday_ns.doc(
        summary="Intraday market status",
        description="Current market clock, session phase (e.g. ACTIVE_TRADING), and day statistics",
    )
    @intraday_ns.response(200, "Success", intraday_status_model)
    @intraday_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Get intraday market clock and day statistics."""
        try:
            return intraday_handler.get_status()
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500


@intraday_ns.route("/config")
class IntradayConfigResource(Resource):
    @intraday_ns.doc(
        summary="Get intraday configuration",
        description="Current intraday engine parameters, capital limits, and square-off times",
    )
    @intraday_ns.response(200, "Success", intraday_config_model)
    @intraday_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Get current intraday configuration."""
        try:
            return intraday_handler.get_config()
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500

    @intraday_ns.doc(
        summary="Update intraday configuration",
        description="Update intraday risk limits, max positions, and target risk per trade",
    )
    @intraday_ns.expect(update_intraday_config_model)
    @intraday_ns.response(200, "Success")
    @intraday_ns.response(500, "Server error", error_response_model)
    def patch(self):
        """Update intraday trading engine configuration."""
        try:
            payload = request.get_json() or {}
            return intraday_handler.update_config(payload)
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500


@intraday_ns.route("/positions")
class IntradayPositionsResource(Resource):
    @intraday_ns.doc(summary="List intraday positions", description="Retrieve active or closed intraday positions")
    @intraday_ns.expect(intraday_positions_parser)
    @intraday_ns.response(200, "Success")
    @intraday_ns.response(500, "Server error", error_response_model)
    def get(self):
        """List intraday positions optionally filtered by status."""
        try:
            args = intraday_positions_parser.parse_args()
            status_filter = args.get("status")
            return intraday_handler.get_positions(status_filter=status_filter)
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500

    @intraday_ns.doc(
        summary="Create intraday position",
        description="Execute an intraday position order with pre-calculated stop-loss and targets",
    )
    @intraday_ns.expect(create_intraday_position_model)
    @intraday_ns.response(201, "Created")
    @intraday_ns.response(400, "Bad request", error_response_model)
    def post(self):
        """Open a new intraday position."""
        try:
            data = request.get_json() or {}
            return intraday_handler.create_position(data), 201
        except Exception as e:
            return {"status": "error", "error": str(e)}, 400


@intraday_ns.route("/positions/<string:position_id>/close")
@intraday_ns.param("position_id", "Intraday position unique identifier")
class IntradayPositionCloseResource(Resource):
    @intraday_ns.doc(summary="Close intraday position", description="Close an active intraday position by ID")
    @intraday_ns.expect(close_intraday_position_model)
    @intraday_ns.response(200, "Closed")
    @intraday_ns.response(400, "Error closing", error_response_model)
    def post(self, position_id):
        """Close an open intraday position."""
        try:
            data = request.get_json(silent=True) or {}
            reason = data.get("reason", "MANUAL_CLOSE")
            return intraday_handler.close_position(position_id, reason=reason)
        except Exception as e:
            return {"status": "error", "error": str(e)}, 400


@intraday_ns.route("/square-off-all")
class IntradaySquareOffAllResource(Resource):
    @intraday_ns.doc(
        summary="Square off all intraday positions",
        description="Immediately square off and close all active intraday positions",
    )
    @intraday_ns.response(200, "All positions squared off")
    @intraday_ns.response(500, "Server error", error_response_model)
    def post(self):
        """Square off all intraday positions."""
        try:
            return intraday_handler.square_off_all()
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500


@intraday_ns.route("/signals")
class IntradaySignalsResource(Resource):
    @intraday_ns.doc(
        summary="Live intraday scanner signals",
        description="Scan intraday watchlists and return high-conviction buy/sell signals",
    )
    @intraday_ns.expect(intraday_signals_parser)
    @intraday_ns.response(200, "Success")
    @intraday_ns.response(500, "Server error", error_response_model)
    def get(self):
        """Get live intraday scanner signals."""
        try:
            args = intraday_signals_parser.parse_args()
            simulate = args.get("simulate", False)
            return intraday_handler.get_signals(allow_simulate=simulate)
        except Exception as e:
            return {"status": "error", "error": str(e)}, 500
