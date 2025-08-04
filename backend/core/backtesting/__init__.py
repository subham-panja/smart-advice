"""
Backtesting Package
==================

This package contains backtesting-related modules for testing trading strategies.

Modules:
- backtest_engine: Core backtesting functionality
- backtest_metrics: Calculation of performance metrics
- portfolio_simulator: Portfolio simulation logic
"""

from .backtest_engine import BacktestEngine
from .backtest_metrics import BacktestMetrics
from .portfolio_simulator import PortfolioSimulator

__all__ = [
    'BacktestEngine',
    'BacktestMetrics',
    'PortfolioSimulator'
]
