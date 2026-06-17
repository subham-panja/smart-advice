# GitHub Copilot Instructions

## Project Overview
Stock analysis and trading platform for the Indian Equity Market (NSE) with technical analysis, ML sentiment analysis, and portfolio backtesting.

## Before Writing Code
Always check AGENT.md for:
- Technology stack and architecture
- Development rules and constraints
- Directory structure
- Workflow guides in .agent/workflows/

## Coding Standards
- Use vectorized operations with numpy/pandas/vectorbt (avoid Python loops over DataFrames)
- Strategies are JSON-configured in backend/strategies/ (not hardcoded)
- New indicator modules inherit from BaseStrategy
- Data is cached as date-stamped parquet files in backend/data/historical/
- Never bypass circuit_breaker, position limits, or stop-loss logic

## Key Technologies
- Backend: Python (Flask), MongoDB, Redis, TA-Lib, vectorbt
- Frontend: Next.js 15.5, React 19, Tailwind CSS v4
- ML: PyTorch, HuggingFace Transformers, stable-baselines3

## File Locations
- Strategies: backend/strategies/*.json
- Indicators: backend/scripts/strategies/
- Backtests: backend/scripts/run_ultimate_backtest.py
- Workflows: .agent/workflows/
