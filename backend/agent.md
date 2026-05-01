# Agent Instructions & Project Context

This file contains persistent rules and context that every agent (AI assistant) should follow when working on this repository.

## Core Principles
1. **Institutional Grade**: All trading logic must maintain pre-defined risk parameters (Market Cap > 5000Cr, Delivery > 55%).
2. **Performance First**: Favor vectorized operations (numpy/pandas) over iterative loops to protect Apple Silicon hardware.
3. **Macro Awareness**: Never suggest or execute individual stock analysis without verifying the NIFTY 50 macro-gate first.
4. **Circuit Breaker Respect**: If `TRADING_OPTIONS["circuit_breaker"]` is True, ALL trading and analysis activity must halt.
5. **Strategy JSON First**: Trading strategies are defined as JSON files in `backend/strategies/`, not hardcoded in Python.

## Working with this Pipeline
- **Primary Analysis Script**: `run_analysis.py` (strategy-config-driven)
- **Unified Trading Cycle**: `main_orchestrator.py` (analysis + execution)
- **Configuration**: `backend/config.py` (infrastructure only)
- **Strategy Configs**: `backend/strategies/*.json` (trading logic)
- **Indicator Modules**: `backend/scripts/strategies/` (50+ TA-Lib modules)
- **Database**: MongoDB (`super_advice`)
- **Telegram Bot**: `telegram_bot.py` (remote control)

## Key Modules
- **Execution Engine**: `scripts/execution_engine_paper.py` (BUY/SELL with pyramiding)
- **Portfolio Monitor**: `scripts/portfolio_monitor_paper.py` (SL, targets, trailing SL, time-stop)
- **Strategy Loader**: `utils/strategy_loader.py` (dynamic JSON strategy loading)
- **Swing Signals**: `scripts/swing_trading_signals.py` (gates + entry patterns)
- **Risk Manager**: `scripts/risk_management.py` (position sizing, ATR-based stops)

## Instruction Checklist for Agents
- [ ] Read `backend/config.py` before suggesting any infrastructure changes.
- [ ] Read relevant `backend/strategies/*.json` before modifying trading logic.
- [ ] Check Nifty-50 gate logic in `run_analysis.py` before modifying the pipeline.
- [ ] Ensure all new indicator modules are vectorized.
- [ ] Verify that `circuit_breaker` logic is preserved in any execution-related changes.
- [ ] Ensure `strategy_name` is tracked in all recommendations and positions.
