# Skill: Performance Debugging & Optimization

Guidelines for debugging slow backtests and optimizing performance.

## When to Use This Skill
- Backtest takes > 10 minutes for Phase 1
- High CPU usage (> 90% for extended periods)
- High memory usage (> 2GB)
- yfinance downloads happening repeatedly (cache misses)
- Walk-forward taking > 1 hour

## Performance Profiling

### Quick Check
```bash
cd backend
time python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 60
```

**Expected times:**
- Phase 1 (60 months, full NSE): 2-5 minutes
- Phase 2-6: 1-2 minutes
- Walk-forward (8 iterations): 10-30 minutes

### Detailed Profiling
```bash
cd backend
python -m cProfile -o profile.stats scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 24

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(30)"
```

**Common bottlenecks:**
1. `yf.Ticker().history()` - Data fetching (should be cached)
2. `talib.*` functions - Indicator calculation (should use vectorbt)
3. `pd.DataFrame.iterrows()` - Row iteration (slow, use vectorized)
4. `time.sleep()` - Rate limiting (reduce REQUEST_DELAY)

## Common Performance Issues

### Issue 1: Repeated yfinance Downloads

**Symptom:** Network activity during backtest, slow data fetch

**Diagnosis:**
```bash
cd backend
python3 << 'EOF'
from utils.data_cache import get_cache_stats
stats = get_cache_stats()
print(f"Cached today: {stats['today_cached']}")
print(f"Total symbols: {len(stats['symbols'])}")
print(f"Missing: {len(stats['symbols']) - stats['today_cached']}")
EOF
```

**Solutions:**
1. Pre-cache missing symbols:
   ```bash
   python3 << 'EOF'
   from utils.data_cache import fetch_historical_data_cached
   from scripts.data_fetcher import get_all_nse_symbols
   
   symbols = get_all_nse_symbols()
   for sym in symbols:
       try:
           df = fetch_historical_data_cached(sym, period='10y')
           print(f"Cached {sym}: {len(df)} rows")
       except Exception as e:
           print(f"Failed {sym}: {e}")
   EOF
   ```

2. Check cache format:
   ```bash
   ls backend/data/historical/ | head -20
   # Should be: SYMBOL.parquet (one file per symbol, e.g. RELIANCE.parquet)
   ```

3. Cache freshness is checked against NIFTY 50 (`^NSEI.parquet`) last trading day — handles weekends AND Indian holidays automatically.

### Issue 2: Non-Vectorized Operations

**Symptom:** High CPU usage, slow indicator calculation

**Diagnosis:**
```bash
grep -r "for.*in.*iterrows" backend/scripts/
grep -r "apply(" backend/scripts/ | grep -v "vectorbt"
```

**Solutions:**
1. Replace `iterrows()` with vectorized operations:
   ```python
   # Slow
   for idx, row in df.iterrows():
       df.loc[idx, 'signal'] = 1 if row['close'] > row['sma'] else 0
   
   # Fast
   df['signal'] = (df['close'] > df['sma']).astype(int)
   ```

2. Use vectorbt for indicators:
   ```python
   # Slow (TA-Lib in loop)
   for sym in symbols:
       df[sym]['rsi'] = talib.RSI(df[sym]['close'], timeperiod=14)
   
   # Fast (vectorbt batch)
   from scripts.vectorbt_indicator_batch import compute_all_indicators
   indicators = compute_all_indicators(symbols_data, strategy_config)
   ```

3. Use IndicatorStore for O(1) lookups:
   ```python
   # Slow (recalculating)
   rsi = talib.RSI(df['close'], timeperiod=14)
   
   # Fast (pre-computed lookup)
   rsi = store.get(symbol, date, 'RSI', 14)
   ```

### Issue 3: Memory Usage

**Symptom:** Memory > 2GB, system slowdown

**Diagnosis:**
```bash
# Monitor memory during backtest
python -m memory_profiler scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 24
```

**Solutions:**
1. Reduce symbol universe:
   ```python
   # In run_ultimate_backtest.py
   symbols = {s: s for s in list(all_nse)[:500]}  # Limit to 500 stocks
   ```

2. Chunk data loading:
   ```python
   # Load data in batches
   batch_size = 500
   for i in range(0, len(symbols), batch_size):
       batch = list(symbols.keys())[i:i+batch_size]
       symbols_data.update(fetch_symbols_data(batch, period='10y'))
   ```

3. Clear intermediate data:
   ```python
   import gc
   del symbols_data  # Free memory
   gc.collect()
   ```

### Issue 4: Slow Walk-Forward

**Symptom:** Walk-forward takes > 1 hour

**Diagnosis:**
Check number of windows and iterations:
```python
# In run_portfolio_backtest.py
window_days = 180
step_days = 90
# For 10 years: ~40 windows
# With 12 MC iterations: 40 * 12 = 480 runs
```

**Solutions:**
1. Reduce MC iterations:
   ```bash
   python scripts/run_ultimate_backtest.py --mc-iterations 4  # Instead of 12
   ```

2. Increase step size:
   ```python
   # In run_portfolio_backtest.py
   step_days = 180  # Instead of 90 (fewer windows)
   ```

3. Use shorter period:
   ```bash
   python scripts/run_ultimate_backtest.py --months 60  # Instead of 120
   ```

### Issue 5: MongoDB Bottleneck

**Symptom:** Slow saves to MongoDB, high I/O

**Diagnosis:**
```bash
# Check MongoDB performance
mongosh --eval "db.backtest_sessions.stats()"
```

**Solutions:**
1. Batch inserts:
   ```python
   # Slow (one by one)
   for trade in trades:
       db.portfolio_backtest_trades.insert_one(trade)
   
   # Fast (batch)
   db.portfolio_backtest_trades.insert_many(trades)
   ```

2. Add indexes:
   ```bash
   mongosh --eval "db.portfolio_backtest_trades.createIndex({session_id: 1})"
   mongosh --eval "db.portfolio_backtest_daily_snapshots.createIndex({session_id: 1, date: 1})"
   ```

3. Disable saves for testing:
   ```bash
   python scripts/run_ultimate_backtest.py --skip-db  # If flag exists
   ```

## Optimization Checklist

Before running full backtest:
- [ ] All symbols cached for today
- [ ] No `iterrows()` or `apply()` in critical paths
- [ ] Using vectorbt IndicatorStore
- [ ] Stock prefilter computed once
- [ ] MongoDB indexes in place
- [ ] Sufficient RAM (4GB+ recommended)
- [ ] No other heavy processes running

## Performance Benchmarks

**Target Performance (Apple Silicon M1/M2):**

| Scenario | Expected Time | Memory |
|----------|--------------|--------|
| Phase 1 (10y, full NSE) | 2-5 min | 1-1.5 GB |
| Phase 2-6 | 1-2 min | 1.5-2 GB |
| Walk-forward (8 iter) | 10-30 min | 2-3 GB |
| Full run (12 iter) | 20-60 min | 2-3 GB |

**If significantly slower:**
- Check for non-vectorized operations
- Verify cache is working
- Profile with cProfile
- Check for memory leaks

## Quick Fixes

### 1. Reduce Universe for Testing
```bash
# Test with 100 stocks first
python3 << 'EOF'
from scripts.data_fetcher import get_all_nse_symbols
symbols = get_all_nse_symbols()[:100]
# Modify run_ultimate_backtest.py to use this subset
EOF
```

### 2. Skip Expensive Phases
```bash
# Quick test (no walk-forward, no stress tests)
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 24 --skip-wf --skip-stress
```

### 3. Use Cached Indicators
Ensure `vectorbt_indicator_batch.py` is being used:
```python
# In run_ultimate_backtest.py
from scripts.vectorbt_indicator_batch import compute_all_indicators
indicators = compute_all_indicators(symbols_data, strategy_config)
store = IndicatorStore(indicators)
engine.set_indicator_store(store)
```

### 4. Parallel Data Fetch
```python
# In data_cache.py
with ThreadPoolExecutor(max_workers=8) as executor:  # Increase from 4
    futures = [executor.submit(fetch, sym) for sym in symbols]
```

## Monitoring During Backtest

```bash
# In another terminal
top -l 1 | grep python  # CPU usage
vm_stat | grep "Pages active"  # Memory usage
lsof -p <PID> | grep TCP  # Network connections (should be 0 if cached)
```

## When to Seek Help

Consult project owner if:
- Backtest hangs indefinitely
- Memory exceeds 4GB
- Results are incorrect (negative CAGR with good signals)
- Cache corruption (parquet files unreadable)
- MongoDB connection issues
