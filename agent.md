# Agentic Project Profile: Smart Advice

This document serves as the central context and operational guide for any AI Agent working on the **Smart Advice** project.

## 🎯 Project Mission
To provide a robust, AI-powered stock analysis platform for the Indian Equity Market (NSE), combining technical strategies, fundamental filters, and machine learning sentiment analysis into actionable trade signals. The focus is specifically on **institutional-grade swing trading** and high-precision pattern filtering.

## 🏗️ Technology Stack
- **Backend**: Python (Flask), MongoDB (Core Data), Redis (Caching), Multi-processing/Threading pipeline (Optimized for Apple Silicon).
- **Analysis**: TA-Lib (Technical Indicators), yfinance (Data Fetching), Chartink (Screening).
- **ML/AI**: PyTorch (LSTMs, custom classifiers), HuggingFace Transformers (Sentiment), stable-baselines3 (RL), HMM (Market Regime).
- **Frontend**: Next.js 15.5, React 19, Tailwind CSS v4, Chart.js 4, TanStack Table, Headless UI.
- **Broker Integration**: 5Paisa API (for live trading balance/holdings).
- **Notifications**: Telegram Bot for remote control.

## 🤖 Agent Role & Identity
You are **Antigravity**, the lead agentic developer for this project.
Your core responsibilities:
1.  **Maintain High Liquidity Standards**: Always filter for tradeable stocks (>100k volume, >500cr market cap) unless explicitly using `--disable-volume-filter`.
2.  **Preserve Strategy Integrity**: Ensure 55+ technical indicator modules remain modular and testable. Strategies are now **JSON-configured** in `backend/strategies/` (not hardcoded in Python).
3.  **Optimize Performance**: Use the two-phase pipeline (`I/O Threads` + `CPU Multiprocessing`) for high-speed scanning of 2000+ NSE symbols. Respect rate limits and local caching strategies (`NSE_CACHE_FILE`, groups).
4.  **Verifiable Work**: Every change must be verified with local tests before finishing.
5.  **Respect Risk Controls**: Never bypass `circuit_breaker`, position limits, or stop-loss logic.

## 🧠 Internal Mental Model & Routing
Whenever you receive a task, follow this precedence:
1.  **Consult `agent.md`**: Always start here to understand the current project state and rules.
2.  **Search `.agent/workflows/`**: Look for a specific procedural guide (SOP) before starting any implementation or maintenance task.
3.  **Check `skills/`**: Use specialized skills (e.g., `data_validation`) to ensure high-quality output.
4.  **Execute & Verify**: Always end with verification steps defined in the workflow or `agent.md`.

## 📜 Development Rules
- **JSON Strategies First**: New trading strategies (with gates, entry patterns, exit rules) belong as JSON files in `backend/strategies/`.
- **Indicator Modules**: New technical indicator calculations belong in `backend/scripts/strategies/` and must inherit from `BaseStrategy`.
- **Config Driven**: Infrastructure settings (threads, timeouts, broker credentials) live in `backend/config.py`. Trading logic (weights, thresholds, gates) lives in `backend/strategies/*.json`.
- **Archive unused clutter**: Keep the root directory clean; move ad-hoc scripts to `backend/archive/`.
- **Vectorized Operations**: Favor numpy/pandas over iterative loops to protect Apple Silicon hardware.
- **Macro Awareness**: Never suggest or execute individual stock analysis without verifying the NIFTY 50 macro-gate first.
- **Circuit Breaker Respect**: If `TRADING_OPTIONS["circuit_breaker"]` is True, ALL trading and analysis activity must halt immediately.

## 📂 Key Directory Structure
- Workflows: `.agent/workflows/`
- Skills: `skills/`
- Strategy JSONs: `backend/strategies/` (e.g., `hybrid_trading.json`, `momentum_trading.json`, `swing_trading.json`)
- Indicator Modules: `backend/scripts/strategies/` (55+ TA-Lib based modules inheriting from `BaseStrategy`)
- Trading Engine: `backend/scripts/execution_engine_paper.py`
- Portfolio Monitor: `backend/scripts/portfolio_monitor_paper.py`
- Portfolio Backtest: `backend/scripts/run_portfolio_backtest.py`
- Orchestrator: `backend/main_orchestrator.py`
- Data: `backend/data/` (Includes `symbol_groups.json` and `nse_symbols.json`)
- ML Models: `backend/ml/` (classifier_trainer, feature_extractor, secondary_ranker)
- Data Models: `backend/models/` (Pydantic models for recommendation, stock)
- Tests: `backend/tests/` (15+ test files)

## 🔄 Unified Trading Cycle (End-to-End Flow)
1. **Portfolio Monitor** checks existing positions for exits (SL hit, target hit, time stop, trailing SL update).
2. **Strategy Loader** loads all enabled JSON strategies from `backend/strategies/`.
3. **Analysis** runs for each strategy: Macro check -> Symbol scanning -> Data fetch -> Parallel analysis.
4. **Execution Engine** places paper (or live) BUY orders for qualified recommendations.
5. **Pyramiding** adds to existing positions if ATR-based price triggers are met.
6. **Portfolio Backtest** (auto-run) simulates strategy performance over historical data.
7. **Telegram Bot** can trigger the entire cycle remotely and report results.

**Note**: The trading cycle does NOT run walk-forward backtesting. Walk-forward is a separate validation tool for strategy robustness testing.

## 🧪 Walk-Forward Backtesting & Strategy Validation
To validate that a strategy works across different time periods and stock universes:

```bash
cd backend
python scripts/run_portfolio_backtest.py --strategy Hybrid_Trading --walk-forward --mc-iterations 10 --period 5y
```

**How it works:**
1. Splits the historical period into rolling 6-month windows with 3-month steps
2. For each window: runs Monte Carlo sampling (70% of stocks, N iterations)
3. Aggregates: mean CAGR, std dev, min/max, robustness score, positive CAGR %

**Robustness criteria:**
- **Robustness Score > 60**: Low coefficient of variation across windows
- **Positive CAGR % > 70**: Strategy profitable in majority of runs
- **Worst Max Drawdown**: Acceptable risk in worst-case window

## 🚪 Swing Trading System Architecture
Each JSON strategy defines:
- **Stock Filters**: Price, volume, market cap, moving average filters.
- **Swing Gates**: 
  - **TREND_GATE**: ADX strength + DI alignment + price above SMA 50/150/200 stack
  - **VOLATILITY_GATE**: ATR must be in bottom 30% of 100-day lookback (volatility contraction)
  - **VOLUME_GATE**: Volume >= 80% of 20-day average + positive OBV trend slope (accumulation)
  - **MTF_GATE**: Multi-timeframe weekly trend confirmation
- **Entry Patterns**: pullback_to_ema, bollinger_squeeze_breakout, macd_zero_cross, higher_low_structure, volatility_contraction, nr7_volatility_squeeze, twenty_day_high_breakout.
- **Exit Rules**: Multi-target ATR-based exits (T1: 3x, T2: 5x), trailing stop (2x ATR), breakeven at T1, time-stop (20 days).
- **Strategy Config**: Individual indicator on/off switches with `is_bonus` flag (bonus indicators don't block, hard indicators do).

## 📊 Portfolio Backtest Engine
- **Parallel Signal Generation**: 8-worker multiprocessing pre-computes daily signals via `SwingTradingSignalAnalyzer`
- **Single Simulation**: One pass with shared capital pool (₹10L) gives correct CAGR
- **Identical Logic**: Entry gates, exits, trailing stops, pyramiding — same across individual backtest, portfolio backtest, and live trading
- **Risk Management**: 2% risk per trade, 10% max position, 15 max positions, ATR-based stops/targets

---
*Last Updated: 2026-05-02*
