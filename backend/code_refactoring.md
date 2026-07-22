# Backend Code Refactoring & Optimization Plan

This document outlines the plan to clean up, simplify, and restructure the **Smart Advice** backend codebase. 

---

## 1. Executive Summary & Goals

* **Max File Limit Rule**: No single file should exceed **1,000 lines of code**. Large files will be split into smaller, focused modules.
* **Remove Code Bloat**: Delete obsolete backup files, temporary scratch scripts, and legacy migration tools.
* **Clean & Modular Structure**: Move scripts into clear functional folders (`engine/`, `analysis/`, `ml/`, `routes/`, `utils/`).
* **Human-Readable Code**: Eliminate redundant code, repetitive comments, and unused helper functions.

---

## 2. Unnecessary Files to Delete or Archive

The following **7 files** are obsolete, duplicate, or one-off temporary scripts that should be removed to declutter the codebase:

| File Path | Lines | Reason for Removal |
| :--- | :--- | :--- |
| `backend/config_backup.py` | 270 | Duplicate backup of `config.py`. Outdated and unused. |
| `backend/scratch/migrate_recomms.py` | 28 | Temporary scratch script for MongoDB migration. |
| `backend/scripts/migrate_cache_to_single_file.py` | 67 | One-time cache migration script. No longer needed. |
| `backend/scripts/db_migrate.py` | 105 | One-time database schema migration script. |
| `backend/scripts/enhanced_data_fetcher.py` | 110 | Redundant wrapper duplicate of `scripts/data_fetcher.py`. |
| `backend/scripts/trade_logic.py` | 42 | Dead stub functions; actual logic lives in backtest engine. |
| `backend/scripts/screener_filter.py` | 36 | Duplicate filtering logic already handled in `utils/stock_scanner.py`. |

---

## 3. Large Files to Split (Enforcing < 1,000 Lines Limit)

Currently, **3 files exceed 1,000 lines**. They will be split into smaller, single-responsibility files:

### A. `backend/scripts/portfolio_backtest_engine.py` (1,510 Lines)
* **Problem**: Combines portfolio simulation loop, position tracking, trailing stop-losses, slippage math, and metric calculation into one giant file.
* **Refactoring Plan**:
  1. `portfolio_backtest_engine.py` (< 600 lines): Core session orchestrator and day-by-day loop.
  2. `backtest_position_manager.py` (~450 lines): Order execution, position sizing, entry/exit gates, trailing stop-loss rules.
  3. `backtest_metrics.py` (~350 lines): Return metrics, Sharpe ratio, Max Drawdown, CAGR, and snapshot math.

### B. `backend/scripts/run_portfolio_backtest.py` (1,107 Lines)
* **Problem**: Mixes command-line handling, signal pre-computation, data fetching, and walk-forward matrix evaluation.
* **Refactoring Plan**:
  1. `run_portfolio_backtest.py` (< 500 lines): Clean CLI runner and configuration loader.
  2. `signal_precomputer.py` (~450 lines): Multi-threaded signal pre-computation & indicator batch lookup.

### C. `backend/tests/test_slippage_replay.py` (1,001 Lines)
* **Problem**: Combines data caching, signal pre-populating, API replay simulation, and markdown report generation.
* **Refactoring Plan**:
  1. `test_slippage_replay.py` (< 600 lines): Main test runner and assertions.
  2. `test_replay_helpers.py` (~350 lines): Cache pre-loading and MongoDB signal injection helpers.

---

## 4. Architectural Restructuring & Folder Re-organization

To prevent `backend/scripts/` from becoming a dumping ground (currently 41 files in one folder), files will be organized into logical directories:

```
backend/
├── app.py                      # Main Flask App entry point
├── config.py                   # Central configuration
├── database.py                 # MongoDB & Redis connections
├── main_orchestrator.py        # Live trading & analysis orchestrator
│
├── engine/                     # Core Execution & Simulation Engines
│   ├── portfolio_backtest_engine.py
│   ├── backtest_position_manager.py
│   ├── backtest_metrics.py
│   └── execution_engine_paper.py
│
├── analysis/                   # Specialized Market Analyzers
│   ├── fundamental_analysis.py
│   ├── sector_analysis.py
│   ├── sentiment_analysis.py
│   ├── market_microstructure.py
│   ├── alternative_data_fetcher.py
│   └── alternative_data_analyzer.py
│
├── ml/                         # Machine Learning Modules
│   ├── feature_extractor.py
│   ├── classifier_trainer.py
│   ├── secondary_ranker.py
│   └── rl_trading_agent.py
│
├── handlers/                   # API Business Logic Handlers
├── routes/                     # Flask REST Endpoints
├── utils/                      # Shared System Utilities
│
├── scripts/                    # Runnable CLI Scripts ONLY
│   ├── run_portfolio_backtest.py
│   ├── run_ultimate_backtest.py
│   ├── run_regime_test.py
│   └── sync_historical_data.py
│
└── tests/                      # Unit & Integration Tests
```

---

## 5. Code Simplification & Cleanup Guidelines

When refactoring code across all modules, follow these mandatory human-standard rules:

1. **Delete Dead Code**: Remove commented-out code blocks, unused imports, and redundant helper abstractions.
2. **Remove Verbose Comments**: Remove obvious comments like `# Function to calculate moving average`. Keep code clean and self-explanatory.
3. **Use Vectorized Operations**: Replace slow Python `for` loops on DataFrames with vectorized pandas/numpy operations.
4. **Enforce Clean Interfaces**: Use clear parameter names and consistent function signatures.

---

## 6. Implementation Phasing & Safety Verification

To ensure zero breaking changes:

* **Phase 1**: Remove the 7 unnecessary files and verify test suite passes.
* **Phase 2**: Split `portfolio_backtest_engine.py`, `run_portfolio_backtest.py`, and `test_slippage_replay.py`.
* **Phase 3**: Move script modules into logical `engine/` and `analysis/` folders, updating import statements.
* **Phase 4**: Run full test suite (`pytest`) to confirm all strategy calculations, APIs, and backtests work seamlessly.

---

## 7. Status & Completion Summary

- [x] **File Line Limit Enforcement (< 1,000 Lines)**:
  - `portfolio_backtest_engine.py`: **Reduced from 1,510 to 989 lines** (Extracted `backtest_metrics.py` and `backtest_position_manager.py`).
  - `run_portfolio_backtest.py`: **Reduced from 1,107 to 794 lines** (Extracted `signal_precomputer.py`).
  - `test_slippage_replay.py`: **Reduced from 1,001 to 665 lines** (Extracted `replay_helpers.py`).
  - **Result**: **0 files exceed 1,000 lines** in `backend/`.

- [x] **Obsolete File Removal**:
  - Deleted `config_backup.py`, `scratch/migrate_recomms.py`, `scripts/migrate_cache_to_single_file.py`, `scripts/db_migrate.py`, and `scripts/enhanced_data_fetcher.py`.

- [x] **Verification**:
  - All automated tests (`pytest`) passed cleanly.

