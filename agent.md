# Agentic Project Profile: Smart Advice

This document serves as the central context and operational guide for any AI Agent working on the **Smart Advice** project.

## Project Mission
To provide a robust, AI-powered stock analysis platform for the Indian Equity Market (NSE), combining technical strategies, fundamental filters, and machine learning sentiment analysis into actionable trade signals. The focus is specifically on **institutional-grade swing trading** and high-precision pattern filtering.

## Technology Stack
- **Backend**: Python (Flask), MongoDB (Core Data), Redis (Caching), Multi-processing/Threading pipeline (Optimized for Apple Silicon).
- **Analysis**: TA-Lib (Technical Indicators), yfinance (Data Fetching), Chartink (Screening), vectorbt (Vectorized Backtesting).
- **ML/AI**: PyTorch (LSTMs, custom classifiers), HuggingFace Transformers (Sentiment), stable-baselines3 (RL), HMM (Market Regime).
- **Frontend**: Next.js 15.5, React 19, Tailwind CSS v4, Chart.js 4, TanStack Table, Headless UI.
- **Broker Integration**: 5Paisa API (for live trading balance/holdings).
- **Notifications**: Telegram Bot for remote control.

## Agent Role & Identity
You are **Antigravity**, the lead agentic developer for this project.
Your core responsibilities:
1.  **Maintain High Liquidity Standards**: Always filter for tradeable stocks (>100k volume, >500cr market cap) unless explicitly using `--disable-volume-filter`.
2.  **Preserve Strategy Integrity**: Ensure 55+ technical indicator modules remain modular and testable. Strategies are **JSON-configured** in `backend/strategies/` (not hardcoded in Python).
3.  **Optimize Performance**: Use vectorized operations (numpy/pandas/vectorbt). The backtest engine uses vectorbt IndicatorStore for O(1) indicator lookups — no TA-Lib calls during simulation.
4.  **Verifiable Work**: Every change must be verified with local tests before finishing.
5.  **Respect Risk Controls**: Never bypass `circuit_breaker`, position limits, or stop-loss logic.

## Internal Mental Model & Routing
Whenever you receive a task, follow this precedence:
1.  **Consult `agent.md`**: Always start here to understand the current project state and rules.
2.  **Search `.agent/workflows/`**: Look for a specific procedural guide (SOP) before starting any implementation or maintenance task.
3.  **Check `skills/`**: Use specialized skills (e.g., `data_validation`) to ensure high-quality output.
4.  **Execute & Verify**: Always end with verification steps defined in the workflow or `agent.md`.

## Development Rules
- **JSON Strategies First**: New trading strategies (with gates, entry patterns, exit rules) belong as JSON files in `backend/strategies/`.
- **Indicator Modules**: New technical indicator calculations belong in `backend/scripts/strategies/` and must inherit from `BaseStrategy`.
- **Config Driven**: Infrastructure settings (threads, timeouts, broker credentials) live in `backend/config.py`. Trading logic (weights, thresholds, gates) lives in `backend/strategies/*.json`.
- **Archive unused clutter**: Keep the root directory clean; move ad-hoc scripts to `backend/archive/`.
- **Vectorized Operations**: Favor numpy/pandas/vectorbt over iterative loops to protect Apple Silicon hardware.
- **Macro Awareness**: Never suggest or execute individual stock analysis without verifying the NIFTY 50 macro-gate first.
- **Circuit Breaker Respect**: If `TRADING_OPTIONS["circuit_breaker"]` is True, ALL trading and analysis activity must halt immediately.

## Key Directory Structure
- Workflows: `.agent/workflows/`
- Skills: `skills/`
- Strategy JSONs: `backend/strategies/` (4 strategies defined)
- Indicator Modules: `backend/scripts/strategies/` (55+ TA-Lib based modules inheriting from `BaseStrategy`)
- Trading Engine: `backend/scripts/execution_engine_paper.py`
- Portfolio Monitor: `backend/scripts/portfolio_monitor_paper.py`
- Portfolio Backtest: `backend/scripts/portfolio_backtest_engine.py`
- Ultimate Backtest: `backend/scripts/run_ultimate_backtest.py` (6-phase validation)
- Vectorbt Indicators: `backend/scripts/vectorbt_indicator_batch.py` (batch pre-computation + IndicatorStore)
- Orchestrator: `backend/main_orchestrator.py`
- Data: `backend/data/` (Includes `symbol_groups.json` and `nse_symbols.json`)
- ML Models: `backend/ml/` (classifier_trainer, feature_extractor, secondary_ranker)
- Data Models: `backend/models/` (Pydantic models for recommendation, stock)
- Tests: `backend/tests/` (15+ test files)

## Unified Trading Cycle (End-to-End Flow)
1. **Portfolio Monitor** checks existing positions for exits (SL hit, target hit, time stop, trailing SL update).
2. **Strategy Loader** loads all enabled JSON strategies from `backend/strategies/`.
3. **Analysis** runs for each strategy: Macro check -> Symbol scanning -> Data fetch -> Parallel analysis.
4. **Execution Engine** places paper (or live) BUY orders for qualified recommendations.
5. **Pyramiding** adds to existing positions if ATR-based price triggers are met.
6. **Portfolio Backtest** (auto-run) simulates strategy performance over historical data.
7. **Telegram Bot** can trigger the entire cycle remotely and report results.

**Note**: The trading cycle does NOT run walk-forward backtesting. Walk-forward is a separate validation tool for strategy robustness testing.

## Strategy Validation (Ultimate Backtest)

The ultimate backtest runs 6 phases to validate strategy robustness:

```bash
cd backend
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --max-stocks 10 --mc-iterations 8
```

**Phase 1**: Historical backtest with realistic costs (gap risk, STT, slippage, brokerage)
**Phase 1b**: Save results to MongoDB
**Phase 2**: Statistical validation (DSR, Monte Carlo permutation, MLRS)
**Phase 3**: Walk-Forward Monte Carlo (8 rolling windows, sequential execution)
**Phase 4**: Stress tests (regime, param sensitivity +/-20%, cost sensitivity)
**Phase 5**: Trade diagnostics (hold times, exit reasons, concurrent positions)
**Phase 6**: Composite confidence score (0-100, realistic CAGR projection)

**Key optimizations:**
- vectorbt indicator batch pre-computation (one-time, all symbols x all dates)
- IndicatorStore for O(1) lookups during simulation (no TA-Lib during loop)
- Sequential walk-forward (no multiprocessing overhead)
- Chartink results cached across all phases
- Data fetched once with `period="max"`, simulation range controlled by `--months`

**Performance**: ~2 minutes for 10y/10-stock/8-iter with all phases.

**Strategy Benchmark Results (all 10y, 10 stocks, 8 iter, no skips):**

| Strategy | CAGR | Final Value | Max DD | Win Rate | Profit Factor | Confidence |
|----------|------|-------------|--------|----------|---------------|------------|
| Swing_Trading | 26.13% | ₹1,017,225 | -12.25% | 51.9% | 19.63 | 55/100 (Moderate) |
| Hybrid_Trading | 1.13% | ₹111,912 | -8.51% | 42.1% | 1.23 | 48/100 (Low) |
| Momentum_Trading | 3.51% | ₹141,084 | -17.21% | 49.3% | 1.47 | 37/100 (Low) |
| Nitin_Triple_Confirm | 0.14% | ₹101,379 | -12.96% | 41.4% | 0.42 | 33/100 (Low) |

**Confidence Score Calculation:**
- Walk-Forward Robustness (20%), DSR (15%), MC Permutation (15%), Stress Tests (15%), Param Stability (10%), Cost Resilience (10%), Data Sufficiency (5%)
- Edge Verified: total_score >= 65
- Realistic CAGR = base_cagr * haircut (Moderate=0.50, Low=0.25, High=0.70, Very High=0.85)

## Swing Trading System Architecture
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

## Portfolio Backtest Engine
- **Vectorbt Indicator Pre-computation**: All indicators computed once in batch before simulation
- **IndicatorStore**: O(1) lookups during day-by-day simulation loop (no TA-Lib during loop)
- **Single Simulation**: One pass with shared capital pool gives correct CAGR
- **Identical Logic**: Entry gates, exits, trailing stops, pyramiding — same across individual backtest, portfolio backtest, and live trading
- **Risk Management**: 2% risk per trade, 10% max position, 8 max positions, ATR-based stops/targets
- **Realistic Costs**: Gap risk, STT, stamp duty, SEBI charges, slippage on entry/exit, brokerage

---
*Last Updated: 2026-05-17*
