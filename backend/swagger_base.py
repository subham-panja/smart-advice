"""
Base Blueprint, Api, and Namespaces for Swagger / OpenAPI documentation.
"""

from flask import Blueprint, jsonify
from flask_restx import Api

from handlers import analysis_handler

swagger_bp = Blueprint("swagger_api", __name__, url_prefix="/api")

api = Api(
    swagger_bp,
    title="Smart Advice Trading API",
    version="1.0.0",
    description=(
        "Institutional-grade Stock Analysis & Trading Platform API for Indian Equity Market (NSE). "
        "Provides real-time technical analysis, automated trade orchestration, swing/intraday "
        "position tracking, and vectorized backtesting."
    ),
    doc="/docs",
)


# Root level health check on the blueprint
@swagger_bp.route("/health")
@swagger_bp.route("/")
def root_api_health():
    return jsonify(analysis_handler.health_check())


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------

analysis_ns = api.namespace("analysis", description="Stock analysis, recommendations, and candle charts")
positions_ns = api.namespace("positions", description="Portfolio positions and activity logs")
orchestrator_ns = api.namespace("orchestrator", description="Trading cycle orchestration and exit confirmations")
config_ns = api.namespace("config", description="Trading settings, cycle stats, and dashboard metrics")
backtests_ns = api.namespace(
    "backtests", description="Vectorized backtesting sessions, trades journal, and performance"
)
intraday_ns = api.namespace("intraday", description="Intraday market status, signals, config, and positions")
