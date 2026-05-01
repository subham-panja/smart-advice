# Agent Instructions & Project Context

This file contains persistent rules and context that every agent (AI assistant) should follow when working on this repository.

## Core Principles
1. **Institutional Grade**: All trading logic must maintain pre-defined risk parameters (Market Cap > 500Cr, Delivery > 55%).
2. **Performance First**: Favor vectorized operations (numpy/pandas) over iterative loops to protect Apple Silicon hardware.
3. **Macro Awareness**: Never suggest or execute individual stock analysis without verifying the NIFTY 50 macro-gate first.
4. **Circuit Breaker Respect**: If `TRADING_OPTIONS["circuit_breaker"]` is True, ALL trading and analysis activity must halt.
5. **Strategy JSON First**: Trading strategies are defined as JSON files in `backend/strategies/`, not hardcoded in Python.

## Working with this Pipeline
- **Primary Analysis Script**: `run_analysis.py` (strategy-config-driven)
- **Unified Trading Cycle**: `main_orchestrator.py` (analysis + execution + portfolio backtest)
- **Configuration**: `backend/config.py` (infrastructure only)
- **Strategy Configs**: `backend/strategies/*.json` (trading logic)
  - `hybrid_trading.json` (enabled) — Combined multi-factor strategy
  - `momentum_trading.json` (disabled) — Momentum-based entries
  - `swing_trading.json` (disabled) — Classic swing trading with gates
- **Indicator Modules**: `backend/scripts/strategies/` (55+ TA-Lib modules, all inherit from `BaseStrategy`)
- **Database**: MongoDB (`super_advice`)
- **Telegram Bot**: `telegram_bot.py` (remote control)

## Key Modules
- **Execution Engine**: `scripts/execution_engine_paper.py` (BUY/SELL with pyramiding)
- **Portfolio Monitor**: `scripts/portfolio_monitor_paper.py` (SL, targets, trailing SL, time-stop)
- **Portfolio Backtest**: `scripts/portfolio_backtest_engine.py` + `scripts/run_portfolio_backtest.py` (historical simulation)
- **Strategy Loader**: `utils/strategy_loader.py` (dynamic JSON strategy loading)
- **Swing Signals**: `scripts/swing_trading_signals.py` (gates + entry patterns)
- **Risk Manager**: `scripts/risk_management.py` (position sizing, ATR-based stops)
- **Chartink Filter**: `scripts/chartink_filter.py` (rapid NSE screening)
- **Confluence Engine**: `scripts/confluence_engine.py` (multi-timeframe signal confluence)
- **Options Analyzer**: `scripts/options_analyzer.py` (OI analysis, PCR, unwinding)
- **Market Regime**: `scripts/market_regime_detection.py` (HMM-based regime detection)
- **Smart Money Tracker**: `scripts/smart_money_tracker.py` (FII/DII flows)
- **Screener Filter**: `scripts/screener_filter.py` (fundamental screening via Screener.in)
- **Sentiment Analysis**: `scripts/sentiment_analysis.py` (NLP-based news sentiment)
- **Sector Analysis**: `scripts/sector_analysis.py` (industry comparisons)
- **Deep Learning**: `scripts/deep_learning_models.py` (LSTM price prediction)
- **RL Trading Agent**: `scripts/rl_trading_agent.py` (reinforcement learning)
- **Fundamental Analysis**: `scripts/fundamental_analysis.py` (financial metrics)
- **Backtesting**: `scripts/backtesting.py` + `scripts/backtesting_runner.py` (per-strategy backtest)
- **ML Pipeline**: `ml/` (classifier_trainer, feature_extractor, secondary_ranker)
- **Data Models**: `models/` (Pydantic: Recommendation, Stock)

## Instruction Checklist for Agents
- [ ] Read `backend/config.py` before suggesting any infrastructure changes.
- [ ] Read relevant `backend/strategies/*.json` before modifying trading logic.
- [ ] Check Nifty-50 gate logic in `run_analysis.py` before modifying the pipeline.
- [ ] Ensure all new indicator modules are vectorized (numpy/pandas, no row loops).
- [ ] Verify that `circuit_breaker` logic is preserved in any execution-related changes.
- [ ] Ensure `strategy_name` is tracked in all recommendations and positions.
- [ ] New indicator modules must inherit from `BaseStrategy` in `scripts/strategies/base_strategy.py`.
- [ ] Test new modules: `python -c "from scripts.strategies.your_module import YourClass; print('OK')"`.
