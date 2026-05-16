# Issue: Historical Paper Test Cannot Run Full 10-Year Backtest

## Problem

`scripts/run_historical_paper_test.py` fails to run a proper 10-year backtest because of a **data availability mismatch** across the stock universe.

### Root Cause

The `PortfolioBacktestSession` engine requires a **common date intersection** across all 50 scanned stocks. When stocks like `BSE`, `MTARTECH`, or newer IPOs are included, they have far less than 10 years of data. The engine's `common-date union` logic clamps the simulation start to the **latest first-available-date** across all stocks.

**Example:**
- 48 stocks have data from 2016
- 2 stocks (BSE, MTARTECH) only have data from 2021+
- Result: sim_start clamps to 2021+, and in practice even worse → **2025-06-02** (only ~11 months)

### Current Behavior
```
Adjusting sim_start from 2016-05-16 to 2025-06-02 (earliest available data)
Running simulation: 2025-06-02 -> 2026-05-16 (120 months)
```

The 120-month request silently degrades to ~11 months.

### Error Before Fix
```
RuntimeError: Insufficient index data for regime detection: 1 days < 250 required
```
This was caused by timezone mismatch between simulation dates and cached `^NSEI` parquet data. (Fixed with date-clamping patch.)

---

## Solution: Use Each Stock's Available Data Individually

### Approach

Instead of forcing a **common date intersection** across all stocks, allow each stock to be simulated over **its own available date range** within the requested window.

### Changes Required

#### 1. Per-Symbol Date Window in `PortfolioBacktestSession`

Modify `portfolio_backtest_engine.py` to:
- Accept a `sim_start_date` and `sim_end_date` as global bounds
- For each stock, compute its **effective window**: `max(sim_start, stock_first_date)` to `min(sim_end, stock_last_date)`
- Skip stocks that have no overlap with the simulation window
- Track per-stock date ranges in results

#### 2. Relax Common-Date Union in `run_historical_paper_test.py`

Remove the clamping logic that forces all stocks to share the same start date. Instead:

```python
# Instead of:
earliest = max(df.index.min() for df in symbols_data.values())

# Do:
# Let each stock trade only when it has data
# Capital allocation handles the rest
```

#### 3. Capital Pool Handling

The shared capital pool already handles this correctly — when a stock has no data for a given date, it simply doesn't trade. The portfolio value is tracked as sum of all positions + cash.

#### 4. Minimum Data Threshold

Add a configurable minimum data threshold (e.g., 250 trading days ≈ 1 year) to exclude stocks with insufficient history for meaningful signal generation:

```python
MIN_TRADING_DAYS = 250
symbols_data = {
    sym: df for sym, df in symbols_data.items()
    if len(df) >= MIN_TRADING_DAYS
}
```

### Implementation Steps

1. **`scripts/portfolio_backtest_engine.py`**:
   - In `_simulate_day`, skip stocks that have no data for the current date
   - Track effective per-stock date ranges in results

2. **`scripts/run_historical_paper_test.py`**:
   - Remove the `max(all_dates)` clamping
   - Add minimum data filter (e.g., 500 trading days for 10y test)
   - Report actual per-symbol date ranges in output

3. **`_print_summary`**:
   - Add a row showing how many stocks were excluded due to insufficient data
   - Show the actual date range coverage (min/max across remaining stocks)

### Expected Outcome

- 10-year backtest runs properly with ~40-45 stocks that have sufficient data
- Newer IPOs are automatically excluded
- Each stock trades over its full available history within the window
- Results are more accurate and representative of strategy performance

### Alternative: Hybrid Approach

If per-symbol windows are too complex, a simpler fix is:

1. **Pre-filter stocks** by minimum history (e.g., `len(df) >= 2000` for ~8 years)
2. **Keep the common-date union** but with fewer, older stocks
3. This is less elegant but requires minimal code changes
