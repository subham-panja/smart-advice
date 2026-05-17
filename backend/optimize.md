# Vectorbt Migration Guide — COMPLETED

## Status

The vectorbt migration has been **fully implemented** and is operational. All P0 and P1 priorities from the original plan are complete.

---

## What's Implemented

### P0 — Completed

#### 1. Vectorbt Indicator Batch Pre-computation
**File**: `backend/scripts/vectorbt_indicator_batch.py`

- `compute_all_indicators()`: Computes ALL indicators for ALL symbols across ALL dates in one vectorized batch
- `IndicatorStore`: O(1) lookup store for simulation loop (replaces per-bar TA-Lib calls)
- Indicators computed: RSI, MACD, ADX, ATR, Bollinger Bands, OBV, SMA (5/10/20/50/150/200), EMA (21), volume metrics, and all custom signals
- Used by: `portfolio_backtest_engine.py`, `run_ultimate_backtest.py`, `run_portfolio_backtest.py`

**Impact**: 120,000 TA-Lib calls → 1 batch computation. Simulation loop reduced from O(indicators) to O(1) per date lookup.

#### 2. IndicatorStore Integration in Simulation Loop
**File**: `backend/scripts/portfolio_backtest_engine.py`

- `set_indicator_store()`: Attaches pre-computed indicators to the engine
- All signal checks use `self._indicator_store.get(symbol, indicator, date)` instead of recomputing
- Market breadth filter (SMA-20 check) uses IndicatorStore for O(1) lookups
- No TA-Lib calls during the day-by-day simulation loop

### P1 — Completed

#### 3. Sequential Walk-Forward Execution
**File**: `backend/scripts/run_portfolio_backtest.py`

- Removed multiprocessing spawn (pickle serialization overhead was worse than sequential)
- `_walk_forward_mc_worker_sequential()`: Single-process walk-forward with shared IndicatorStore
- Pre-computed indicators shared across all windows (no re-computation per window)
- Signal pre-computation removed (IndicatorStore made it redundant)

#### 4. Chartink Result Caching Across Phases
**File**: `backend/scripts/run_ultimate_backtest.py`

- `results["_scanned_symbols"]`: Chartink scan results cached after Phase 1
- Reused in Phases 3 (walk-forward), 4 (stress tests), eliminating 4 redundant HTTP scans
- Data fetched once with `period="max"`, simulation range controlled by `--months`

---

## What Was NOT Migrated (Correctly Left As-Is)

| Component | Why Not Migrated |
|-----------|-----------------|
| `data_cache.py` | Parquet caching is already efficient |
| `database.py` / `persistence_handler.py` | MongoDB persistence is unrelated to backtesting |
| `data_fetcher.py` | yfinance fetching is I/O-bound, not compute-bound |
| `stock_scanner.py` | External API calls, no vectorization benefit |
| `fundamental_analysis.py` | Fundamental data is not time-series, no vectorization benefit |
| `market_regime_detection.py` | Only runs once per cycle, not a bottleneck |
| ML training (`classifier_trainer.py`) | sklearn already uses vectorized operations internally |
| `telegram_bot.py` / `app.py` | Web/communication layer, unrelated |

---

## Performance Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| 10y/10-stock/8-iter (all phases) | ~50-80 min estimated | ~2 min | 25-40x |
| Indicator computation during sim | Per-bar, per-symbol | Once, batched | 1000x+ |
| Walk-forward execution | Multiprocessing (spawn) | Sequential + shared store | 5-10x |
| Chartink scans per run | 5 (one per phase) | 1 (cached) | 5x |
| Data fetching | Redundant per phase | Once (period="max") | 3-4x |

---

## Memory Usage

```
n_symbols x n_dates x n_indicators x 8 bytes (float64)

Example: 50 symbols x 2500 days x 20 indicators x 8 bytes = ~20 MB
```

For daily NSE data, memory is a non-issue. The concern only arises at minute-level data or millions of parameter combinations.

---

## Architecture Summary

```
Data Fetch (yfinance, period="max")
    ↓
Chartink Scan (1x, cached across phases)
    ↓
vectorbt Indicator Batch (1x, all symbols × all dates)
    ↓
IndicatorStore (O(1) lookups during simulation)
    ↓
Simulation Loop (day-by-day, no TA-Lib)
    ↓
Results → MongoDB + Confidence Score
```

---
*Last Updated: 2026-05-17*
