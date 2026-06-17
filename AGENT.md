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
1.  **Consult `AGENT.md`**: Always start here to understand the current project state and rules.
2.  **Search `.agent/workflows/`**: Look for a specific procedural guide (SOP) before starting any implementation or maintenance task.
3.  **Read relevant `skills/`**: **MANDATORY** - Before any research, analysis, or implementation work, read the relevant skill file(s) below. Skills contain domain-specific knowledge, best practices, and validation checklists that prevent costly mistakes.
4.  **Execute & Verify**: Always end with verification steps defined in the workflow, skill, or `AGENT.md`.

## Available Skills (Read Before Working)

### Core Trading Skills
- **`skills/strategy_analysis.md`**: How to analyze strategy performance, interpret confidence scores, and make data-driven improvements. **Read before**: Modifying strategy JSON, adjusting gates/entry patterns, or optimizing parameters.
  
- **`skills/backtest_validation.md`**: How to validate backtest results, interpret each phase, and identify red flags. **Read before**: Running ultimate backtest, analyzing results, or comparing strategies.

- **`skills/entry_pattern_optimization.md`**: How to optimize entry patterns, tune gate thresholds, and add new patterns. **Read before**: Modifying entry patterns, adjusting gate parameters, or adding new technical signals.

- **`skills/risk_management.md`**: How to configure position sizing, stop losses, and risk controls. **Read before**: Adjusting risk parameters, investigating high drawdown, or setting up new strategies.

- **`skills/performance_debugging.md`**: How to debug slow backtests, optimize performance, and fix cache issues. **Read before**: Investigating slow backtests, high memory usage, or repeated yfinance downloads.

### Data Quality Skills
- **`skills/data_validation.md`**: How to validate stock data quality, filter noisy data, and ensure clean inputs. **Read before**: Working with stock data, adding new data sources, or investigating data quality issues.

## When to Read Skills

**ALWAYS read skills when:**
- Starting a new feature or modification
- Analyzing backtest results
- Optimizing strategy parameters
- Debugging performance issues
- Making risk management changes
- Working with stock data

**Skill reading workflow:**
1. Identify relevant skill(s) for your task
2. Read the entire skill file (10-15 minutes)
3. Follow the workflow/checklist in the skill
4. Apply the best practices and validation steps
5. Document any deviations or learnings

**Example:**
```
Task: "Optimize swing trading entry patterns"
→ Read: skills/entry_pattern_optimization.md
→ Follow: Optimization Workflow (steps 1-5)
→ Apply: Parameter Tuning Guidelines
→ Verify: Backtesting Parameter Changes checklist
```

**Don't skip skills** - they contain hard-won knowledge about what works, what doesn't, and common pitfalls. Reading a skill upfront saves hours of trial-and-error.

## Finding Work & Understanding Tasks
When starting work on this project, check these locations in order:

1.  **Current Git Status**: Run `git status` to see uncommitted changes or work in progress.
2.  **Recent Commits**: Run `git log --oneline -20` to understand recent changes and context.
3.  **Test Failures**: Run `pytest backend/tests/` to identify broken functionality that needs fixing.
4.  **TODO Comments**: Search for `TODO`, `FIXME`, or `XXX` comments in the codebase using `grep -r "TODO\|FIXME" backend/`.
5.  **Workflow Gaps**: Review `.agent/workflows/` to identify missing or incomplete procedures.
6.  **Strategy Performance**: Check MongoDB `backtest_sessions` collection for strategies with low confidence scores that need improvement.
7.  **User Requests**: Clarify the task scope with the user if the request is ambiguous (e.g., "Which strategy?", "What time period?", "Full NSE or subset?").

**Task Categories:**
- **Bug Fix**: Check test failures, logs in `backend/logs/`, and error messages.
- **New Feature**: Check `.agent/workflows/` for relevant SOP, then implement following the workflow.
- **Performance Optimization**: Profile with `cProfile`, check for non-vectorized operations, review backtest execution time.
- **Strategy Improvement**: Run ultimate backtest, analyze confidence score components, adjust gates/entry patterns in JSON config.
- **Documentation**: Update `AGENT.md` or `.agent/workflows/*.md` if processes have changed.

## Multi-Agent Mode (When to Use)
For complex tasks, use multiple agents in parallel when:

**Use Multi-Agent When:**
- Task involves **independent subtasks** (e.g., "Update all 4 strategy JSON files" - each agent handles one strategy)
- Need to **compare approaches** (e.g., "Test 3 different entry patterns" - each agent tests one)
- **Large-scale refactoring** across multiple files (e.g., "Update all indicator modules to use vectorbt" - parallelize by indicator type)
- **Data processing pipelines** (e.g., "Fetch data for 500 stocks" - split into batches)

**Don't Use Multi-Agent When:**
- Tasks have **sequential dependencies** (e.g., "Run backtest then analyze results")
- **Small, focused changes** (e.g., "Fix bug in one file")
- **Exploration/research** tasks that need context accumulation

**Multi-Agent Best Practices:**
1.  **Clear Boundaries**: Each agent should work on independent files/data to avoid conflicts.
2.  **Merge Strategy**: Designate one agent to merge results, or use git worktrees for isolation.
3.  **Progress Tracking**: Use `TodoWrite` to track which agents completed which subtasks.
4.  **Resource Awareness**: Don't spawn more than 4-6 parallel agents to avoid overwhelming the system.

**Example Multi-Agent Scenarios:**
```
Task: "Update all strategy configs to use new volume filter"
→ Agent 1: Update Swing_Trading.json
→ Agent 2: Update Hybrid_Trading.json
→ Agent 3: Update Momentum_Trading.json
→ Agent 4: Update Nitin_Triple_Confirm.json
→ Main Agent: Merge changes, run tests, commit
```

## Development Rules
- **JSON Strategies First**: New trading strategies (with gates, entry patterns, exit rules) belong as JSON files in `backend/strategies/`.
- **Indicator Modules**: New technical indicator calculations belong in `backend/scripts/strategies/` and must inherit from `BaseStrategy`.
- **Config Driven**: Infrastructure settings (threads, timeouts, broker credentials) live in `backend/config.py`. Trading logic (weights, thresholds, gates) lives in `backend/strategies/*.json`.
- **Archive unused clutter**: Keep the root directory clean; move ad-hoc scripts to `backend/archive/`.
- **Vectorized Operations**: Favor numpy/pandas/vectorbt over iterative loops to protect Apple Silicon hardware.
- **Macro Awareness**: Never suggest or execute individual stock analysis without verifying the NIFTY 50 macro-gate first.
- **Circuit Breaker Respect**: If `TRADING_OPTIONS["circuit_breaker"]` is True, ALL trading and analysis activity must halt immediately.
- **Test Coverage**: When adding new features or modifying existing code, always update or add corresponding test cases in `backend/tests/`. Run `pytest backend/tests/` to verify all tests pass before committing.
- **Documentation Updates**: If you change architecture, add new workflows, or modify key systems, update this `AGENT.md` file and relevant `.agent/workflows/*.md` files to keep documentation current.

## Key Directory Structure
- Workflows: `.agent/workflows/`
- Skills: `skills/` (6 specialized skill files - see "Available Skills" section above)
  - `strategy_analysis.md` - Strategy performance analysis and improvement
  - `backtest_validation.md` - Backtest result validation and interpretation
  - `entry_pattern_optimization.md` - Entry pattern tuning and gate optimization
  - `risk_management.md` - Position sizing, stops, and risk controls
  - `performance_debugging.md` - Backtest performance debugging and optimization
  - `data_validation.md` - Stock data quality validation
- Strategy JSONs: `backend/strategies/` (4 strategies defined)
- Indicator Modules: `backend/scripts/strategies/` (55+ TA-Lib based modules inheriting from `BaseStrategy`)
- Trading Engine: `backend/scripts/execution_engine_paper.py`
- Portfolio Monitor: `backend/scripts/portfolio_monitor_paper.py`
- Portfolio Backtest: `backend/scripts/portfolio_backtest_engine.py`
- Ultimate Backtest: `backend/scripts/run_ultimate_backtest.py` (6-phase validation)
- Vectorbt Indicators: `backend/scripts/vectorbt_indicator_batch.py` (batch pre-computation + IndicatorStore)
- Vectorbt Signals: `backend/scripts/vectorbt_signal_generator.py` (stock prefilter computation)
- Data Cache: `backend/data/historical/` (date-stamped parquet files: `{SYMBOL}_{YYYY-MM-DD}.parquet`)
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
# Full run with walk-forward (12 MC iterations)
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --mc-iterations 12 --telegram

# Quick run without walk-forward (faster, skips Phase 3)
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --telegram
```

**Phase 1**: Historical backtest with realistic costs (gap risk, STT, slippage, brokerage) — uses full NSE universe
**Phase 1b**: Save results to MongoDB (trades, daily snapshots, summary metrics)
**Phase 2**: Statistical validation (DSR, Monte Carlo permutation, MLRS)
**Phase 3**: Walk-Forward Monte Carlo (rolling windows, only if `--mc-iterations` is passed)
**Phase 4**: Stress tests (regime, param sensitivity +/-20%, cost sensitivity)
**Phase 5**: Trade diagnostics (hold times, exit reasons, concurrent positions)
**Phase 6**: Composite confidence score (0-100, realistic CAGR projection)

**All phases (2-6) save to MongoDB** in the `backtest_sessions` collection under `ultimate_phases`.

**Key optimizations:**
- vectorbt indicator batch pre-computation (one-time, all symbols x all dates)
- IndicatorStore for O(1) lookups during simulation (no TA-Lib during loop)
- Sequential walk-forward (no multiprocessing overhead)
- Chartink results cached across all phases
- **Date-stamped parquet cache**: `{symbol}_{YYYY-MM-DD}.parquet` in `backend/data/historical/` — if today's cache exists with sufficient rows, no yfinance download occurs
- Stock prefilter computed once with vectorbt (no per-stock loops)

**Data Caching:**
- Cache location: `backend/data/historical/`
- Format: `{SYMBOL}_{YYYY-MM-DD}.parquet` (e.g., `RELIANCE_2026-06-17.parquet`)
- Cache check: Requires today's date in filename AND sufficient rows for the requested period
- If cache is missing or insufficient, data is fetched from yfinance and cached for the day
- Symbols with < 250 trading days are excluded from simulation

**Performance**: ~2 minutes for Phase 1 (full NSE universe, 10y). Walk-forward adds significant compute time depending on MC iterations.

**Latest Benchmark Results (Swing_Trading, 10y, full NSE universe, 2026-06-17):**

| Metric | Value |
|--------|-------|
| CAGR | 22.92% |
| Total Return | +686.97% |
| Final Value | ₹78,697 (from ₹10,000) |
| Max Drawdown | -20.16% |
| Sharpe Ratio | 0.43 |
| Win Rate | 37.2% |
| Profit Factor | 2.13 |
| Total Trades | 1,351 |
| Expectancy | ₹76.74/trade |

**Historical Benchmark Results (10y, 10 stocks, 8 iter, no skips):**

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
- **Stock Prefilter**: Vectorbt-based prefilter identifies candidate stocks per date (no per-stock loops)
- **Single Simulation**: One pass with shared capital pool gives correct CAGR
- **Full NSE Universe**: Tests against all ~2100+ NSE stocks (no artificial cap)
- **Identical Logic**: Entry gates, exits, trailing stops, pyramiding — same across individual backtest, portfolio backtest, and live trading
- **Risk Management**: 2% risk per trade, 10% max position, 8 max positions, ATR-based stops/targets
- **Realistic Costs**: Gap risk, STT, stamp duty, SEBI charges, slippage on entry/exit, brokerage

---
*Last Updated: 2026-06-17 (Added: 6 swing trading skills, mandatory skill reading instructions)*
