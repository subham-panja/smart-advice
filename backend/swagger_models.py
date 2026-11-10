"""
Flask-RESTX API Models and Request Parsers for Smart Advice Trading API.
"""

from flask_restx import fields, inputs, reqparse

from swagger_base import api

# ---------------------------------------------------------------------------
# Common / Generic Models
# ---------------------------------------------------------------------------

health_model = api.model(
    "HealthStatus",
    {
        "status": fields.String(example="ok", description="Server health status"),
        "timestamp": fields.String(example="2026-11-10T18:00:00.000000", description="Current server time (ISO)"),
    },
)

message_response_model = api.model(
    "MessageResponse",
    {
        "status": fields.String(example="success"),
        "message": fields.String(example="Operation completed successfully."),
    },
)

error_response_model = api.model(
    "ErrorResponse",
    {
        "status": fields.String(example="error"),
        "error": fields.String(example="Detailed error description"),
    },
)


# ---------------------------------------------------------------------------
# Analysis Namespace Models & Parsers
# ---------------------------------------------------------------------------

candle_model = api.model(
    "Candle",
    {
        "date": fields.String(example="2026-11-10", description="Date formatted as YYYY-MM-DD"),
        "open": fields.Float(example=2540.0, description="Opening price"),
        "high": fields.Float(example=2575.5, description="Day high"),
        "low": fields.Float(example=2530.0, description="Day low"),
        "close": fields.Float(example=2568.0, description="Closing price"),
        "volume": fields.Integer(example=1250000, description="Trading volume"),
    },
)

candles_response_model = api.model(
    "CandlesResponse",
    {
        "status": fields.String(example="success"),
        "symbol": fields.String(example="RELIANCE"),
        "candles": fields.List(fields.Nested(candle_model)),
    },
)

recommendation_model = api.model(
    "Recommendation",
    {
        "symbol": fields.String(example="RELIANCE"),
        "company_name": fields.String(example="Reliance Industries Ltd."),
        "technical_score": fields.Float(example=84.5),
        "fundamental_score": fields.Float(example=76.0),
        "sentiment_score": fields.Float(example=68.0),
        "combined_score": fields.Float(example=78.2),
        "is_recommended": fields.Boolean(example=True),
        "recommendation_strength": fields.String(example="STRONG_BUY"),
        "reason": fields.String(example="EMA Golden Cross + Squeeze Momentum breakout"),
        "recommendation_date": fields.String(example="2026-11-10"),
        "buy_price": fields.Float(example=2540.0),
        "sell_price": fields.Float(example=2680.0),
        "expected_return_percent": fields.Float(example=5.51),
        "market_regime": fields.String(example="BULLISH_TREND"),
        "strategy_name": fields.String(example="Swing_Trading_v2"),
        "entry_atr": fields.Float(example=35.5),
        "rr_ratio": fields.Float(example=2.1),
        "note": fields.String(example=""),
        "backtest_metrics": fields.Raw(example={}),
        "detailed_analysis": fields.Raw(example={}),
        "sector_analysis": fields.Raw(example={}),
    },
)

recommendations_response_model = api.model(
    "RecommendationsResponse",
    {
        "status": fields.String(example="success"),
        "count": fields.Integer(example=12),
        "recommendations": fields.List(fields.Nested(recommendation_model)),
    },
)

update_recommendation_model = api.model(
    "UpdateRecommendationRequest",
    {
        "is_recommended": fields.Boolean(example=True, description="Whether to include in active recommendations"),
        "recommendation_strength": fields.String(
            example="STRONG_BUY", description="Recommendation rating (BUY, STRONG_BUY, HOLD, SELL)"
        ),
        "note": fields.String(example="Approved by trader", description="User annotations"),
        "buy_price": fields.Float(example=2540.0),
        "sell_price": fields.Float(example=2680.0),
    },
)

trigger_analysis_model = api.model(
    "TriggerAnalysisRequest",
    {
        "group": fields.String(
            required=True, example="NIFTY50", description="Watchlist or symbol group to scan (e.g. NIFTY50, ALL)"
        ),
        "verbose": fields.Boolean(default=False, description="Enable verbose logs"),
        "max_stocks": fields.Integer(example=50, description="Max stocks to analyze"),
        "all": fields.Boolean(default=False, description="Scan entire universe ignoring filters"),
    },
)

candles_parser = reqparse.RequestParser()
candles_parser.add_argument("days", type=int, default=35, help="Number of historical days to fetch (default 35)")


# ---------------------------------------------------------------------------
# Positions Namespace Models & Parsers
# ---------------------------------------------------------------------------

position_model = api.model(
    "Position",
    {
        "symbol": fields.String(example="RELIANCE"),
        "status": fields.String(example="OPEN", description="OPEN or CLOSED"),
        "quantity": fields.Integer(example=50),
        "entry_price": fields.Float(example=2540.0),
        "current_price": fields.Float(example=2575.0),
        "stop_loss": fields.Float(example=2470.0),
        "target": fields.Float(example=2680.0),
        "current_stop_loss": fields.Float(example=2470.0),
        "current_target": fields.Float(example=2680.0),
        "total_investment": fields.Float(example=127254.0),
        "unrealized_pnl": fields.Float(example=1750.0),
        "unrealized_pnl_pct": fields.Float(example=1.38),
        "strategy_name": fields.String(example="Swing_Trading_v2"),
        "entry_date": fields.String(example="2026-11-10T09:30:00"),
        "is_paper": fields.Boolean(example=True),
    },
)

positions_response_model = api.model(
    "PositionsResponse",
    {
        "status": fields.String(example="success"),
        "count": fields.Integer(example=4),
        "positions": fields.List(fields.Nested(position_model)),
    },
)

create_position_model = api.model(
    "CreatePositionRequest",
    {
        "symbol": fields.String(required=True, example="RELIANCE", description="Stock ticker symbol"),
        "quantity": fields.Integer(required=True, example=50, description="Number of shares/units"),
        "entry_price": fields.Float(required=True, example=2540.0, description="Executed entry price"),
        "stop_loss": fields.Float(example=2470.0, description="Initial stop loss price"),
        "target": fields.Float(example=2680.0, description="Initial target price"),
        "strategy_name": fields.String(default="Manual", example="Manual"),
        "note": fields.String(example="Technical breakout entry"),
    },
)

update_position_model = api.model(
    "UpdatePositionRequest",
    {
        "entry_price": fields.Float(example=2540.0, description="Updated entry price"),
        "quantity": fields.Integer(example=50, description="Updated quantity"),
        "stop_loss": fields.Float(example=2470.0, description="Updated stop loss level"),
        "target": fields.Float(example=2680.0, description="Updated target level"),
        "current_price": fields.Float(example=2575.0, description="Manual current price update"),
        "notes": fields.String(example="Trailing stop adjusted"),
    },
)

activity_log_model = api.model(
    "ActivityLog",
    {
        "symbol": fields.String(example="RELIANCE"),
        "action": fields.String(example="POSITION_OPENED"),
        "timestamp": fields.String(example="2026-11-10T09:30:00"),
        "details": fields.Raw(example={"entry_price": 2540.0, "quantity": 50}),
    },
)

activity_logs_response_model = api.model(
    "ActivityLogsResponse",
    {
        "status": fields.String(example="success"),
        "count": fields.Integer(example=25),
        "logs": fields.List(fields.Nested(activity_log_model)),
    },
)

positions_query_parser = reqparse.RequestParser()
positions_query_parser.add_argument(
    "status", type=str, choices=["OPEN", "CLOSED"], help="Filter by status (OPEN or CLOSED)"
)

activity_logs_parser = reqparse.RequestParser()
activity_logs_parser.add_argument("symbol", type=str, help="Filter activity logs by stock symbol")
activity_logs_parser.add_argument("limit", type=int, default=100, help="Maximum number of logs to return (default 100)")


# ---------------------------------------------------------------------------
# Orchestrator Namespace Models
# ---------------------------------------------------------------------------

run_orchestrator_model = api.model(
    "RunOrchestratorRequest",
    {
        "mode": fields.String(
            default="live", enum=["live", "replay", "date", "full", "quick"], description="Execution mode"
        ),
        "verbose": fields.Boolean(default=False, description="Enable detailed terminal logs"),
        "replay_days": fields.Integer(example=5, description="Days to simulate when mode=replay"),
        "date": fields.String(example="2026-11-10", description="Target date (YYYY-MM-DD) when mode=date"),
    },
)

orchestrator_status_model = api.model(
    "OrchestratorStatusResponse",
    {
        "status": fields.String(example="idle", description="Execution state: idle, running, completed, error"),
        "message": fields.String(example="Trading cycle completed."),
    },
)

pending_exit_model = api.model(
    "PendingExit",
    {
        "symbol": fields.String(example="UNIONBANK"),
        "quantity": fields.Integer(example=100),
        "entry_price": fields.Float(example=175.5),
        "exit_reason": fields.String(example="TARGET_REACHED"),
        "detected_exit_price": fields.Float(example=184.84),
        "created_at": fields.String(example="2026-11-10T15:20:00"),
    },
)

pending_exits_response_model = api.model(
    "PendingExitsResponse",
    {
        "status": fields.String(example="success"),
        "count": fields.Integer(example=1),
        "pending_exits": fields.List(fields.Nested(pending_exit_model)),
    },
)

confirm_exit_model = api.model(
    "ConfirmExitRequest",
    {
        "symbol": fields.String(required=True, example="UNIONBANK", description="Symbol of position being confirmed"),
        "exit_price": fields.Float(required=True, example=184.84, description="Final confirmed execution price"),
    },
)


# ---------------------------------------------------------------------------
# Config Namespace Models & Parsers
# ---------------------------------------------------------------------------

strategy_model = api.model(
    "StrategyConfig",
    {
        "name": fields.String(example="Swing_Trading_v2"),
        "version": fields.String(example="2.0.0"),
        "category": fields.String(example="swing"),
        "enabled": fields.Boolean(example=True),
        "description": fields.String(example="Momentum swing trading strategy with ATR risk management"),
        "file_name": fields.String(example="swing_trading_v2.json"),
    },
)

strategies_response_model = api.model(
    "StrategiesResponse",
    {
        "status": fields.String(example="success"),
        "count": fields.Integer(example=3),
        "strategies": fields.List(fields.Nested(strategy_model)),
    },
)

trading_settings_response_model = api.model(
    "TradingSettingsResponse",
    {
        "status": fields.String(example="success"),
        "config": fields.Raw(example={"initial_capital": 100000.0, "max_positions": 5, "brokerage_charges": 0.002}),
    },
)

cycle_stats_model = api.model(
    "CycleStatsResponse",
    {
        "status": fields.String(example="success"),
        "open_positions": fields.Integer(example=4),
        "total_invested": fields.Float(example=85000.0),
        "cash_remaining": fields.Float(example=15000.0),
        "unsettled_funds": fields.Float(example=0.0),
        "total_equity": fields.Float(example=103500.0),
        "pnl_pct": fields.Float(example=3.5),
        "initial_capital": fields.Float(example=100000.0),
    },
)

dashboard_stats_model = api.model(
    "DashboardStatsResponse",
    {
        "status": fields.String(example="success"),
        "capital": fields.Raw(example={"initial": 100000.0, "current": 103500.0}),
        "positions": fields.Raw(example={"open": 4, "closed": 18}),
        "performance": fields.Raw(example={"win_rate": 66.7, "pnl": 3500.0}),
    },
)

daily_earnings_response_model = api.model(
    "DailyEarningsResponse",
    {
        "status": fields.String(example="success"),
        "total_realized_pnl": fields.Float(example=4250.0),
        "earnings": fields.Raw(example=[]),
    },
)

daily_earnings_parser = reqparse.RequestParser()
daily_earnings_parser.add_argument("days", type=int, help="Number of recent days to include")
daily_earnings_parser.add_argument("start_date", type=str, help="Start date filter (YYYY-MM-DD)")
daily_earnings_parser.add_argument("end_date", type=str, help="End date filter (YYYY-MM-DD)")
daily_earnings_parser.add_argument("session_id", type=str, help="Filter by specific session ID")


# ---------------------------------------------------------------------------
# Backtests Namespace Models & Parsers
# ---------------------------------------------------------------------------

backtest_session_model = api.model(
    "BacktestSession",
    {
        "session_id": fields.String(example="bts_20261110_01"),
        "strategy_name": fields.String(example="Intraday_Trading_v1"),
        "session_name": fields.String(example="Intraday 1H Ultimate Backtest"),
        "created_at": fields.String(example="2026-11-10T12:00:00"),
        "total_return_pct": fields.Float(example=24.5),
        "cagr": fields.Float(example=18.2),
        "sharpe_ratio": fields.Float(example=1.65),
        "max_drawdown_pct": fields.Float(example=-8.4),
        "win_rate": fields.Float(example=64.2),
        "profit_factor": fields.Float(example=1.92),
        "total_trades": fields.Integer(example=142),
        "initial_capital": fields.Float(example=100000.0),
        "final_portfolio_value": fields.Float(example=124500.0),
    },
)

backtest_sessions_response_model = api.model(
    "BacktestSessionsResponse",
    {
        "status": fields.String(example="success"),
        "count": fields.Integer(example=5),
        "sessions": fields.List(fields.Nested(backtest_session_model)),
    },
)

backtest_trades_response_model = api.model(
    "BacktestTradesResponse",
    {
        "status": fields.String(example="success"),
        "page": fields.Integer(example=1),
        "limit": fields.Integer(example=50),
        "total": fields.Integer(example=142),
        "total_pages": fields.Integer(example=3),
        "trades": fields.List(fields.Raw),
    },
)

run_intraday_backtest_model = api.model(
    "RunIntradayBacktestRequest",
    {
        "strategy": fields.String(default="Intraday_Trading_v1", example="Intraday_Trading_v1"),
        "period": fields.String(default="730d", example="730d"),
        "symbols": fields.List(fields.String, example=["RELIANCE", "TCS", "INFY"]),
    },
)

trades_query_parser = reqparse.RequestParser()
trades_query_parser.add_argument("symbol", type=str, help="Filter trades by stock symbol")
trades_query_parser.add_argument("search", type=str, help="Search query string")
trades_query_parser.add_argument("trade_type", type=str, help="Filter by trade type (e.g. LONG_BUY)")
trades_query_parser.add_argument("exit_reason", type=str, help="Filter by exit reason (e.g. TARGET_REACHED, STOP_LOSS)")
trades_query_parser.add_argument("pattern", type=str, help="Filter by trigger chart pattern")
trades_query_parser.add_argument("outcome", type=str, help="Filter by outcome (WIN, LOSS)")
trades_query_parser.add_argument("page", type=int, default=1, help="Page number (default 1)")
trades_query_parser.add_argument("limit", type=int, default=50, help="Number of trades per page (default 50, max 200)")
trades_query_parser.add_argument(
    "sort_by", type=str, default="entry_date", help="Field to sort by (default entry_date)"
)
trades_query_parser.add_argument(
    "sort_dir", type=str, default="desc", choices=["asc", "desc"], help="Sort direction (asc, desc)"
)


# ---------------------------------------------------------------------------
# Intraday Namespace Models & Parsers
# ---------------------------------------------------------------------------

intraday_status_model = api.model(
    "IntradayStatusResponse",
    {
        "status": fields.String(example="success"),
        "clock": fields.Raw(
            example={"current_time_ist": "10:15:00", "phase": "ACTIVE_TRADING", "is_market_open": True}
        ),
        "day_stats": fields.Raw(example={"total_trades": 3, "open_positions": 1, "realized_pnl": 1200.0}),
    },
)

intraday_config_model = api.model(
    "IntradayConfigResponse",
    {
        "status": fields.String(example="success"),
        "config": fields.Raw(
            example={"capital": 100000.0, "max_positions": 5, "risk_per_trade_pct": 1.0, "auto_square_off": "15:15"}
        ),
    },
)

update_intraday_config_model = api.model(
    "UpdateIntradayConfigRequest",
    {
        "capital": fields.Float(example=100000.0),
        "max_positions": fields.Integer(example=5),
        "risk_per_trade_pct": fields.Float(example=1.0),
        "auto_square_off": fields.String(example="15:15"),
    },
)

create_intraday_position_model = api.model(
    "CreateIntradayPositionRequest",
    {
        "symbol": fields.String(required=True, example="TATASTEEL", description="Stock symbol"),
        "entry_price": fields.Float(required=True, example=155.0, description="Entry execution price"),
        "quantity": fields.Integer(required=True, example=100, description="Quantity of shares"),
        "stop_loss": fields.Float(required=True, example=152.0, description="Stop loss level"),
        "target_1": fields.Float(required=True, example=159.0, description="Target 1 level"),
        "target_2": fields.Float(example=162.0, description="Target 2 level"),
        "pattern": fields.String(example="ORB_BREAKOUT", description="Pattern name"),
        "reason": fields.String(example="15m breakout above day high", description="Entry rationale"),
        "note": fields.String(example="Volume surge confirmed", description="Notes"),
    },
)

close_intraday_position_model = api.model(
    "CloseIntradayPositionRequest",
    {
        "reason": fields.String(
            default="MANUAL_CLOSE", example="MANUAL_CLOSE", description="Reason for closing position"
        ),
    },
)

intraday_positions_parser = reqparse.RequestParser()
intraday_positions_parser.add_argument("status", type=str, choices=["OPEN", "CLOSED"], help="Filter by status")

intraday_signals_parser = reqparse.RequestParser()
intraday_signals_parser.add_argument(
    "simulate", type=inputs.boolean, default=False, help="Generate simulated signals when market is closed"
)
