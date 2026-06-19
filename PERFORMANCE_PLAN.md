# Performance Optimization Plan: Portfolio Backtest Engine

## Goal
Reduce full ultimate backtest (120 months, 20 MC iterations) from **~5 hours to under 30 minutes**.

## Current Bottleneck Analysis

### Hot Path: `_simulate_day` (called 1200 times per backtest)

| Operation | Calls/Day | Cost/Call | Total/Day | % of Time |
|-----------|-----------|-----------|-----------|-----------|
| `_scan_for_signals` | 1 loop | ~1.05s | ~1.05s | **95%** |
| `analyze_swing_opportunity` | ~200 (after prefilter) | ~0.5ms | ~100ms | 4% |
| `_process_exits` | ~15 positions | ~0.1ms | ~1.5ms | 0.1% |
| `_process_pyramiding` | ~15 positions | ~0.05ms | ~0.75ms | 0.05% |
| `_record_snapshot` | 1 | ~0.05ms | ~0.05ms | 0.01% |

**Per backtest: ~1200 days x 1.1s = 22 minutes**

### Why `_scan_for_signals` dominates:
1. Iterates all ~2100 symbols every single day
2. Calls `analyze_swing_opportunity` for each (even with prefilter, ~200 pass)
3. Each call: 15+ indicator store lookups, dict creation, pandas slicing
4. Pure Python interpreter overhead on 2.5M+ function calls total

### Total work per ultimate backtest:
- Phase 1: 1 full backtest = ~22 min
- Phase 3 (Walk-forward): 59 windows x 20 MC = 1180 backtests
- Phase 4 (Stress): ~30 backtests
- **Grand total: ~1211 backtests x 22 min = 444 hours (without optimization)**

---

## Phase 1: Signal Pre-computation (Target: 5-10x speedup)
**Effort: 1-2 days | Risk: Very Low**

### What
Pre-compute ALL signals for ALL (stock, date) pairs ONCE before the simulation loop. The simulation loop then does dict lookups instead of recalculating.

### Why it works
- `analyze_swing_opportunity` is called 2.5M times but produces the SAME result for the same (stock, date) pair
- Pre-computation: 2100 stocks x 0.5ms = ~1 min (parallelizable to ~8s on 8 cores)
- Simulation drops from 22 min to ~2-3 min (dict lookups only)

### How
1. The `run_with_signals` path already exists in `portfolio_backtest_engine.py`
2. Create a `precompute_all_signals(symbols_data, strategy_config, indicator_store)` function:
   - For each stock, iterate all dates, call `analyze_swing_opportunity`
   - Store results as `Dict[symbol, Dict[date_str, {score, swing_result}]]`
   - Parallelize across 8 cores (one process per stock batch)
3. Wire this into `run_walk_forward_backtest` and `run_all_stress_tests`
4. Pre-compute signals ONCE for the full dataset, slice per window/MC iteration

### Files to modify
- `backend/scripts/portfolio_backtest_engine.py` — ensure `run_with_signals` handles all edge cases
- `backend/scripts/run_portfolio_backtest.py` — add signal pre-computation before WF loop
- `backend/scripts/stress_tests.py` — use pre-computed signals
- `backend/scripts/run_ultimate_backtest.py` — orchestrate pre-computation

### Expected result
- Per backtest: 22 min -> **3 min** (7x speedup)
- Walk-forward (1180 runs): ~5 hours -> **~45 min**

---

## Phase 2: Cython Compile Simulation Loop (Target: 10-20x additional)
**Effort: 3-5 days | Risk: Low**

### What
Compile the simulation loop (`_simulate_day`, `_process_exits`, `_scan_for_signals`, position management) with Cython to eliminate Python interpreter overhead.

### Why it works
- Python function call overhead: ~100ns per call. With 2.5M calls/day x 1200 days = 3B calls = **300 seconds of pure overhead**
- Cython with type annotations eliminates this, compiling to C-speed function calls
- Proven: pandas itself is partially Cython, [fxmastercourse.com showed 80x speedup](https://www.fxmastercourse.com/improving-your-python-backtesting-from-dataframes-to-cython-part-1/)

### How
1. Create `.pyx` files for the hot path:
   - `portfolio_engine_core.pyx` — simulation loop, exits, pyramiding, snapshots
   - `signal_scanner.pyx` — signal lookup and ranking
2. Add type annotations:
   ```cython
   cdef class PortfolioState:
       cdef double cash
       cdef double peak_value
       cdef int bar_count
       cdef dict positions  # symbol -> PositionData
       cdef list trades
   ```
3. Convert position tracking from dataclass to cdef struct
4. Build with `setup.py`:
   ```python
   from Cython.Build import cythonize
   setup(ext_modules=cythonize(["portfolio_engine_core.pyx"]))
   ```
5. Python engine delegates to Cython core for the hot loop

### Files to create/modify
- `backend/scripts/portfolio_engine_core.pyx` — NEW: Cython simulation core
- `backend/scripts/signal_scanner.pyx` — NEW: Cython signal lookup
- `backend/setup.py` — NEW: Cython build config
- `backend/scripts/portfolio_backtest_engine.py` — delegate to Cython core

### Expected result
- Per backtest: 3 min -> **10-20 seconds** (10-20x speedup)
- Walk-forward (1180 runs): 45 min -> **5-10 min**

### Reference
- [Improving Python Backtesting with Cython](https://www.fxmastercourse.com/improving-your-python-backtesting-from-dataframes-to-cython-part-1/)
- [Run Python 80x Faster with Cython](https://towardsdatascience.com/run-your-python-code-up-to-80x-faster-using-the-cython-library/)

---

## Phase 3: Rust Core Engine via PyO3 (Target: 50-100x over current)
**Effort: 2-4 weeks | Risk: Medium**

### What
Rewrite the simulation engine in Rust, expose as a Python module via PyO3. Python remains the strategy/config layer; Rust handles the hot path (day-by-day simulation, position tracking, exit logic).

### Why Rust
- **50-200x faster** than pure Python for state machine loops
- Zero-cost abstractions: no GIL, no interpreter overhead
- Memory safety without garbage collection
- Perfect for our use case: the simulation is a deterministic state machine
- This is exactly what production systems use:
  - [NautilusTrader](https://github.com/nautechsystems/nautilus_trader) — Rust core + Python API, sub-second backtests
  - [QuantConnect LEAN](https://github.com/quantconnect/lean) — C# engine + Python strategies
  - [Barter-rs](https://github.com/barter-rs/barter-rs) — pure Rust backtesting framework

### Architecture
```
Python Layer (strategy, config, signal generation)
    |
    v  (passes: numpy arrays of OHLCV, pre-computed signals, config)
    |
Rust Engine (via PyO3 + maturin)
    - Day-by-day simulation loop
    - Position state machine (open/close/pyramid)
    - Stop-loss, trailing stop, time-stop
    - Capital management, position sizing
    - Equity curve, drawdown tracking
    |
    v  (returns: trade list, daily snapshots, metrics)
    |
Python Layer (reporting, DB persistence, visualization)
```

### Data Interface (Python -> Rust)
```python
# Python side: pre-compute everything, pass as numpy arrays
prices = {symbol: df[["Open","High","Low","Close","Volume"]].values for ...}
signals = {symbol: precomputed_signal_array for ...}  # numpy bool/float arrays
config = StrategyConfig(...)  # typed config object

result = rust_engine.run_backtest(prices, signals, config)
# Returns: trades list, equity_curve np.array, metrics dict
```

### Rust Module Structure
```
backend/rust_engine/
  Cargo.toml
  src/
    lib.rs              # PyO3 module entry point
    types.rs            # Position, Trade, Config structs
    simulation.rs       # Day-by-day simulation loop
    exits.rs            # Stop-loss, trailing, time-stop logic
    entries.rs          # Position sizing, buy execution
    pyramiding.rs       # Add-to-position logic
    metrics.rs          # CAGR, Sharpe, drawdown calculation
    portfolio.rs        # Capital pool, position tracking
```

### How to build
```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin (Python-Rust build tool)
pip install maturin

# Build the extension
cd backend/rust_engine
maturin develop --release
```

### Python usage
```python
from rust_engine import PortfolioSimulator

sim = PortfolioSimulator(strategy_config)
result = sim.run(prices_dict, signals_dict, start_date, end_date)
# result.trades, result.equity_curve, result.metrics
```

### Files to create
- `backend/rust_engine/` — entire Rust crate
- `backend/scripts/rust_engine_wrapper.py` — Python wrapper that adapts existing data formats
- `backend/scripts/portfolio_backtest_engine.py` — add Rust backend as alternative engine

### Expected result
- Per backtest: 22 min -> **5-15 seconds** (100-200x speedup)
- Walk-forward (1180 runs): ~5 hours -> **2-5 min**
- Full ultimate backtest: ~5 hours -> **10-15 min**

### Reference
- [NautilusTrader Architecture](https://nautilustrader.io/docs/latest/concepts/architecture/)
- [PyO3 User Guide](https://pyo3.rs/v0.29.0/building-and-distribution)
- [Making Python 100x Faster with Rust (PyO3)](https://www.reddit.com/r/rust/comments/125pbq0/blog_post_making_python_100x_faster_with_less/)
- [Nine Rules for Python Extensions in Rust](https://medium.com/data-science/nine-rules-for-writing-python-extensions-in-rust-d35ea3a4ec29)
- [PyO3 + numpy interop](https://terencezl.github.io/blog/2023/06/06/a-week-of-pyo3-rust-numpy/)
- [Barter-rs](https://github.com/barter-rs/barter-rs)

---

## Phase 4: Numba on Remaining Numerical Loops (Bonus: 5-10x)
**Effort: 1-2 days | Risk: Very Low**

### What
Apply `@numba.njit` to any remaining numerical loops that survive Phases 1-3:
- ATR trailing stop calculation
- Position sizing math
- P&L percentage calculations
- Drawdown computation

### How
```python
from numba import njit

@njit(cache=True)
def compute_trailing_stop(prices, entry_price, atr_values, atr_multiplier):
    """Vectorized trailing stop calculation."""
    ...
```

### When to apply
After Phase 3 is done — profile the Rust engine's Python wrapper to find any remaining Python hot spots.

---

## Implementation Order

```
Phase 1 (1-2 days)
  Signal pre-computation
  Target: 22 min -> 3 min per backtest
  |
  v
Phase 2 (3-5 days)                    Phase 3 (2-4 weeks)
  Cython simulation loop     ----->    Rust core engine
  Target: 3 min -> 15 sec     OR       Target: 22 min -> 10 sec
  |                                    |
  v                                    v
Phase 4 (1-2 days)
  Numba on remaining numerical loops
  Polish, edge cases, benchmarks
```

**Phase 1 is independent and should be done immediately.**
**Phase 2 is the quick bridge** — gets us to sub-minute backtests in a week.
**Phase 3 is the endgame** — production-grade, 100x speedup, future-proof.
**Phase 2 can be skipped** if we go straight to Rust (Phase 3).

---

## Benchmark Targets

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Single backtest (5yr, 2100 stocks) | 22 min | 3 min | 15 sec | 5-10 sec |
| Walk-forward (59 win x 20 MC) | 25-49 hrs | 45 min | 5 min | 2-5 min |
| Stress tests (30 runs) | 11 hrs | 90 min | 8 min | 3-5 min |
| Full ultimate backtest | 5+ hrs | ~1.5 hrs | 15-20 min | 10-15 min |

---

## Prerequisites

### For Phase 1 (immediate)
- None — all code exists, just needs wiring

### For Phase 2 (Cython)
```bash
pip install cython
```

### For Phase 3 (Rust)
```bash
# Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Python build tool
pip install maturin

# Verify
rustc --version
maturin --version
```

---

## Risk Mitigation

1. **Correctness**: Every phase must produce identical results to the current engine. Build a comparison test:
   ```python
   # Run both engines on same data, assert results match within tolerance
   old_result = current_engine.run(data)
   new_result = optimized_engine.run(data)
   assert abs(old_result["cagr"] - new_result["cagr"]) < 0.01
   ```

2. **Fallback**: Keep the current Python engine as fallback. The optimized engine is opt-in via config flag.

3. **Incremental**: Each phase is independently useful. Phase 1 alone cuts time by 7x.

4. **Testing**: Run the full ultimate backtest after each phase and compare CAGR, Sharpe, win rate, trade count against the baseline.
