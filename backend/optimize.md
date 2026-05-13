# Vectorbt Migration Guide

## Licensing & Cost

### Is vectorbt open source?

**Yes, with a caveat.** The open-source version (`vectorbt` on PyPI) is licensed under **Apache 2.0 with Commons Clause**.

- **Source code is publicly viewable and modifiable**
- **Free to use** for personal trading, research, and internal use
- **Cannot be resold** — you cannot build a product whose primary value is vectorbt itself and sell it to third parties (e.g., you can't sell "Backtest-as-a-Service" powered by vectorbt)
- **Personal/proprietary trading is fine** — using it to run your own strategies on your own capital is not "selling the software"

### vectorbt vs vectorbt.pro

| Feature | Open Source (`vectorbt`) | Pro (`vectorbt.pro`) |
|---------|-------------------------|---------------------|
| Core backtesting engine | Yes | Yes |
| Vectorized indicators | Yes | Yes |
| Portfolio simulation | Yes | Yes |
| Walk-forward (`vbt.Splitter`) | Yes | Enhanced |
| Monte Carlo / randomization | Yes | Enhanced |
| Parameter optimization | Yes | Advanced |
| Live trading connectors | Limited | Full |
| Advanced portfolio features | — | Yes |
| Priority support | — | Yes |
| Pricing | **Free** | Paid (subscription or perpetual) |

**For our use case** (personal trading system, NSE equities), the **open-source version is sufficient**. We don't need Pro's live trading connectors or advanced portfolio features.

### Installation

```bash
pip install vectorbt
```

Or pin a stable version:
```bash
pip install vectorbt==0.25.0
```

Dependencies: `pandas`, `numpy`, `numba`, `plotly` — all standard.

---

## What to Migrate (Priority Order)

### P0 — Highest Impact (10-50x speedup)

#### 1. Pre-batch All Indicators in `portfolio_backtest_engine.py`

**Current bottleneck:** `analyze_swing_opportunity()` is called once per symbol per day inside the simulation loop. For 1200 days x 100 symbols, that's 120,000 calls, each computing 15+ TA-Lib indicators.

**vectorbt approach:** Compute ALL indicators for ALL symbols across ALL dates in one batch BEFORE the simulation loop. The day-by-day loop becomes a simple signal lookup.

```python
import vectorbt as vbt

# Current: compute indicators one symbol at a time, one date at a time
# for date in common_dates:
#     for symbol in symbols:
#         result = analyzer.analyze_swing_opportunity(symbol, df_for_date, ...)

# vectorbt: batch across all symbols + dates at once
# Stack all OHLCV data into 2D arrays: (n_dates, n_symbols)
close_matrix = pd.concat([df['Close'] for df in all_dfs], axis=1)
high_matrix = pd.concat([df['High'] for df in all_dfs], axis=1)
low_matrix = pd.concat([df['Low'] for df in all_dfs], axis=1)
volume_matrix = pd.concat([df['Volume'] for df in all_dfs], axis=1)

# Compute indicators once for all symbols simultaneously
rsi = vbt.RSI.run(close_matrix, 14).rsi
atr = vbt.ATR.run(high_matrix, low_matrix, close_matrix, 14).atr
macd = vbt.MACD.run(close_matrix)
bbands = vbt.BBands.run(close_matrix, 20, 2)
adx = vbt.talib('ADX', high_matrix, low_matrix, close_matrix, 14)

# Then the simulation loop just does signal lookups on pre-computed arrays
```

**Files affected:**
- `scripts/portfolio_backtest_engine.py` — main simulation loop
- `scripts/swing_trading_signals.py` — indicator computation logic

---

#### 2. Replace Backtrader `next()` Per-Bar Recomputation

**Current bottleneck:** In `base_strategy.py`, the `BacktraderStrategy.next()` method is called every single bar. Each call rebuilds a DataFrame and re-runs `analyze_swing_opportunity()`, which recomputes all TA-Lib indicators from scratch.

**vectorbt approach:** Use `vbt.Portfolio.from_signals()` or `vbt.Portfolio.from_orders()` with pre-computed indicator arrays. No per-bar Python loop at all.

```python
# Current: event-driven, bar-by-bar
# class MyStrategy(bt.Strategy):
#     def next(self):
#         # Recomputes ATR, RSI, etc. on every bar

# vectorbt: fully vectorized
entries = (rsi < 30) & (close > sma_200) & (adx > 25)
exits = (rsi > 70) | (close < stop_loss)

portfolio = vbt.Portfolio.from_orders(
    close=close_matrix,
    size=size_matrix,
    sl_stop=stop_loss_matrix,
    tp_stop=target_matrix,
    entries=entries,
    exits=exits,
    freq='d',
    cash=initial_capital,
    fees=0.001,
    slippage=0.001,
)

# All metrics available instantly
stats = portfolio.stats()
```

**Files affected:**
- `scripts/backtesting.py` — Backtrader engine wrapper
- `scripts/strategies/base_strategy.py` — BacktraderStrategy class
- `scripts/backtesting_runner.py` — strategy evaluation orchestration

---

### P1 — High Impact (5-10x speedup)

#### 3. Batch `feature_extractor.py` Across All Symbols

**Current bottleneck:** ML feature extraction runs per-symbol, each triggering 80+ separate TA-Lib calls. Many are duplicated (SMA 50/200 computed twice, etc.).

```python
# Current: one symbol at a time
# features = extract_features(df_for_one_symbol)

# vectorbt: all symbols at once
# talib functions in vectorbt accept 2D arrays (dates x symbols)
features = {}
features['sma_20'] = vbt.talib('SMA', close_matrix, 20)
features['sma_50'] = vbt.talib('SMA', close_matrix, 50)
features['macd'] = vbt.MACD.run(close_matrix)
features['rsi_14'] = vbt.RSI.run(close_matrix, 14).rsi
features['atr_14'] = vbt.ATR.run(high_matrix, low_matrix, close_matrix, 14).atr

# Rolling features: vectorbt handles these natively
features['volatility_20'] = vbt.IndicatorFactory(
    run_func=lambda close: close.pct_change().rolling(20).std()
).run(close_matrix)
```

**Files affected:**
- `ml/feature_extractor.py`

---

#### 4. Walk-Forward + Monte Carlo Vectorization

**Current bottleneck:** Each MC iteration spawns a subprocess that runs a complete day-by-day backtest. 20 windows x 10 iterations = 200 full simulations, each with O(days x symbols) indicator recomputation.

```python
import vectorbt as vbt
from vectorbt import Splitter

# Define rolling windows: 180-day test, 90-day step
splitter = vbt.Splitter.from_grouper(
    close.index,
    every='180D',
    window='90D',
)

# For each split, run the strategy with pre-computed indicators
# vectorbt broadcasts across all splits simultaneously
portfolios = splitter.run(
    lambda train_idx, test_idx: vbt.Portfolio.from_orders(
        close=close.iloc[test_idx],
        entries=entries.iloc[test_idx],
        exits=exits.iloc[test_idx],
        ...
    )
)

# MC: randomize the universe subsampling as an array dimension
import numpy as np

# Generate N random symbol subsets
n_iterations = 100
symbol_indices = np.arange(n_symbols)
mc_masks = np.random.random((n_iterations, n_symbols)) < sample_pct

# Apply masks to portfolio results — no re-simulation needed
mc_results = []
for mask in mc_masks:
    subset_portfolio = portfolio.select(assets=symbol_indices[mask])
    mc_results.append(subset_portfolio.stats())
```

**Files affected:**
- `scripts/run_portfolio_backtest.py` — walk-forward + MC orchestration
- `scripts/backtest_utils.py` — backtesting wrapper

---

### P2 — Medium Impact (2-5x speedup)

#### 5. Eliminate Duplicate Indicator Computations

**Current issue:** ATR is computed 3 separate times in `risk_management.py` and `trade_logic.py`. SMA 50/200 is computed twice in `feature_extractor.py`. Indicators are recomputed in 40+ individual strategy files.

**vectorbt approach:** Compute once, cache in a shared indicator store.

```python
# Centralized indicator computation
class IndicatorCache:
    def __init__(self, high, low, close, volume):
        self._close = close
        self._high = high
        self._low = low
        self._volume = volume
        self._cache = {}

    def get_atr(self, period=14):
        key = f'atr_{period}'
        if key not in self._cache:
            self._cache[key] = vbt.ATR.run(self._high, self._low, self._close, period).atr
        return self._cache[key]

    def get_rsi(self, period=14):
        key = f'rsi_{period}'
        if key not in self._cache:
            self._cache[key] = vbt.RSI.run(self._close, period).rsi
        return self._cache[key]
```

**Files affected:**
- `scripts/risk_management.py`
- `scripts/trade_logic.py`
- `scripts/swing_trading_signals.py`
- `scripts/confluence_engine.py`
- `scripts/enhanced_volume_confirmation.py`
- `scripts/strategies/*.py` (all 40+ strategy files)

---

#### 6. Vectorize OBV and Manual Loops in `swing_trading_signals.py`

**Current bottleneck:** OBV is computed with a Python `for` loop (line 74-87). ATR contraction counting uses another loop (line 330-334).

```python
# Current: Python for-loop for OBV
# for i in range(1, len(df)):
#     if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
#         obv.iloc[i] = obv.iloc[i-1] + df['Volume'].iloc[i]

# vectorbt: single line
obv = vbt.OBV.run(close, volume).obv

# Current: Python loop for ATR contractions
# vectorbt: vectorized
contractions = (atr < atr.shift(1)).rolling(min_contractions).sum()
```

**Files affected:**
- `scripts/swing_trading_signals.py`

---

#### 7. Vectorize Portfolio Metrics Calculation

**Current bottleneck:** `_calculate_metrics()` iterates through daily snapshots one-by-one to compute returns, Sharpe, drawdown.

```python
# Current: manual loop over daily_snapshots
# for i in range(1, len(self.daily_snapshots)):
#     daily_returns.append((curr - prev) / prev)

# vectorbt: computed automatically
portfolio = vbt.Portfolio.from_orders(...)
sharpe = portfolio.sharpe_ratio()
max_dd = portfolio.max_drawdown()
profit_factor = portfolio.profit_factor()
win_rate = portfolio.win_rate()
all_stats = portfolio.stats()  # comprehensive stats dict
```

**Files affected:**
- `scripts/portfolio_backtest_engine.py` — `_calculate_metrics()`

---

### P3 — Additional Optimization Areas

#### 8. Vectorize Rolling Window Calculations

All rolling operations scattered across the codebase can use vectorbt's built-in rolling utilities:

| File | Current | vectorbt |
|------|---------|----------|
| `ml/feature_extractor.py` | `returns.rolling(20).std()` | `vbt.IndicatorFactory` with rolling |
| `scripts/strategy_evaluator.py` | `Volume.rolling(50).mean()` | Pre-compute with vbt |
| `scripts/swing_trading_signals.py` | `Low.rolling(pivot_lookback).min()` | `vbt.IndicatorFactory` |
| `scripts/market_regime_detection.py` | `Close.rolling(period).mean()` | Pre-computed MA |

#### 9. Vectorize Relative Strength / Cross-Sectional Features

**Current:** Single-symbol relative strength vs index, computed one at a time.

```python
# All symbols relative strength vs index in one operation
relative_strength = (close_matrix / index_close) * 100
rs_sma = vbt.MA.run(relative_strength, rs_period)
rs_score = (relative_strength / rs_sma - 1) * 100
```

**Files affected:**
- `scripts/strategy_evaluator.py`
- `ml/feature_extractor.py` — `_add_cross_sectional_features()`

#### 10. Signal Pre-computation for Walk-Forward

**Current:** `_compute_signals_worker()` uses nested Python loops (symbol x date) even for pre-computation.

**vectorbt approach:** The entire signal matrix is computed in one vectorized pass, eliminating the need for multiprocessing workers for signal computation.

---

## Migration Strategy

### Phase 1: Indicator Pre-computation (Safest, Biggest Win)

1. Add `vectorbt` to `requirements.txt`
2. Create a new `scripts/vectorbt_indicator_batch.py` module that:
   - Takes all OHLCV DataFrames for all symbols
   - Computes ALL indicators in one batch using vectorbt
   - Returns a unified signal DataFrame indexed by (date, symbol)
3. Modify `portfolio_backtest_engine.py` to:
   - Call the batch indicator module BEFORE the simulation loop
   - Replace `analyze_swing_opportunity()` calls with signal lookups
4. **This alone should reduce the 4-hour runtime to ~minutes**

### Phase 2: Replace Backtrader Engine

1. Convert the swing trading strategy logic to vectorbt signal definitions
2. Use `vbt.Portfolio.from_signals()` or `vbt.Portfolio.from_orders()` for single-stock backtests
3. Keep the existing engine alongside during transition, compare results

### Phase 3: Full Portfolio Vectorization

1. Convert the portfolio-level simulation (position management, rebalancing, pyramiding) to vectorbt's portfolio engine
2. This is the most complex step because the current engine has custom logic for:
   - Daily active rebalancing
   - Position pyramiding
   - Multi-exit logic (SL, target, time-stop, delisted)
   - Shared capital pool with ranking-based allocation
3. vectorbt supports all of these, but the mapping requires careful testing

### Phase 4: Walk-Forward + MC

1. Replace the multiprocessing-based MC with vectorbt's array-based randomization
2. Use `vbt.Splitter` for walk-forward windows
3. Results should be identical but orders of magnitude faster

---

## What vectorbt Cannot Replace

These parts of the current system should **not** be migrated:

| Component | Why Keep |
|-----------|----------|
| `data_cache.py` | Parquet caching is already efficient |
| `database.py` / `persistence_handler.py` | MongoDB persistence is unrelated to backtesting |
| `data_fetcher.py` | yfinance fetching is I/O-bound, not compute-bound |
| `stock_scanner.py` | External API calls, no vectorization benefit |
| `fundamental_analysis.py` | Fundamental data is not time-series, no vectorization benefit |
| `market_regime_detection.py` | Only runs once per cycle, not a bottleneck |
| ML training (`classifier_trainer.py`) | sklearn already uses vectorized operations internally |
| `telegram_bot.py` / `app.py` | Web/communication layer, unrelated |

---

## RAM Considerations

vectorbt explodes data into matrices. Estimate memory usage:

```
n_symbols x n_dates x n_indicators x 8 bytes (float64)

Example: 200 symbols x 2500 days x 20 indicators x 8 bytes = ~80 MB
```

For daily NSE data, this is trivial. The concern only arises at:
- Minute-level data (millions of rows)
- Millions of parameter combinations simultaneously

For our use case (daily data, ~200 symbols, ~2500 days), memory is a **non-issue**.

---

## Quick Start Reference

```python
import vectorbt as vbt
import pandas as pd
import numpy as np

# === Single indicator ===
rsi = vbt.RSI.run(close, 14).rsi
atr = vbt.ATR.run(high, low, close, 14).atr

# === Multiple symbols at once ===
# close_matrix: DataFrame with dates as index, symbols as columns
rsi_all = vbt.RSI.run(close_matrix, 14).rsi  # same shape as close_matrix

# === Backtest from signals ===
entries = (rsi < 30) & (close > vbt.MA.run(close, 200).ma)
exits = rsi > 70

pf = vbt.Portfolio.from_orders(
    close=close_matrix,
    entries=entries,
    exits=exits,
    size=1.0,          # or a matrix of position sizes
    sl_stop=0.05,      # 5% stop loss
    tp_stop=0.10,      # 10% target
    cash=1_000_000,
    fees=0.001,
    slippage=0.001,
    freq='d',
)

# === Get all stats ===
print(pf.stats())

# === Walk-forward ===
splitter = vbt.Splitter.from_grouper(pf.index, every='180D', window='90D')

# === Parameter optimization ===
# Test multiple RSI periods and SMA periods simultaneously
rsi_params = vbt.RSI.run(close_matrix, np.arange(10, 30, 2))
sma_params = vbt.MA.run(close_matrix, np.arange(50, 250, 50))

# === Monte Carlo ===
# Randomize entry signals
n_mc = 1000
random_entries = entries.values[np.random.randint(0, 2, (n_mc, *entries.shape))]
mc_pf = vbt.Portfolio.from_orders(close=close_matrix, entries=random_entries, ...)
```

---

## Summary: Expected Speedup

| Operation | Current | With vectorbt | Factor |
|-----------|---------|---------------|--------|
| Single portfolio backtest (5yr, 100 symbols) | ~10 min | ~5 sec | 120x |
| Walk-forward (20 windows) | ~3 hours | ~1 min | 180x |
| MC (10 iterations, 70% sampling) | ~30 min | ~3 sec | 600x |
| Full WF + MC pipeline | ~4 hours | ~2-5 min | 50-120x |
| Feature extraction (all symbols) | ~5 min | ~10 sec | 30x |
| Indicator computation (all symbols) | Recomputed per-bar | Once, batched | 1000x+ |
