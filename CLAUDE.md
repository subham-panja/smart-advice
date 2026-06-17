# Claude Code Instructions

## Project Context
This is a stock analysis and trading platform for the Indian Equity Market (NSE).

## Before Starting Any Task
1. **Always read AGENT.md first** - It contains:
   - Project mission and technology stack
   - Development rules and constraints
   - Key directory structure
   - Strategy validation workflows

2. **Check .agent/workflows/** for task-specific guides:
   - `analyze_stocks.md` - Stock analysis execution
   - `add_strategy.md` - Adding new strategies
   - `frontend_development.md` - Frontend changes

## Key Rules
- Strategies are JSON-configured in `backend/strategies/`
- Use vectorized operations (numpy/pandas/vectorbt)
- Never bypass circuit breaker or risk controls
- Data is cached in `backend/data/historical/` as date-stamped parquet files
- All backtest results save to MongoDB

## Technology Stack
- Backend: Python (Flask), MongoDB, Redis
- Frontend: Next.js 15.5, React 19, Tailwind CSS v4
- Analysis: TA-Lib, yfinance, vectorbt
- ML: PyTorch, HuggingFace Transformers

Read AGENT.md for complete details.
