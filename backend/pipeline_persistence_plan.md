# Pipeline Data Persistence Plan

> **Problem:** We have 8 filtration steps but only save data at Step 7 (backtest) and Step 8 (recommendation).
> Steps 0-6 produce valuable debugging data that gets thrown away.

---

## Current State: What Gets Saved ✅ ALL WIRED

| Step | Filter | Saves to DB? | Collection |
|------|--------|:------------:|------------|
| 0. Macro Gate | Nifty 50 check | **YES** ✅ | `scan_runs` |
| 1. Symbol Fetch | Pre-filter by volume/cap | **YES** ✅ | `scan_runs` |
| 2. Data Fetch | yfinance OHLCV | **YES** ✅ | `scan_runs` |
| 3. Technical Score | 5 strategy signals | **YES** ✅ | `analysis_snapshots` |
| 4. Swing Gates | 4 gates pass/fail | **YES** ✅ | `swing_gate_results` |
| 5. Entry Patterns | 5 pattern checks | **YES** ✅ | `trade_signals` |
| 6. Exit Rules | SL/TP/Trail calc | **YES** ✅ | `trade_signals` |
| 7. Backtest | Metrics validation | **YES** ✅ | `backtest_results` |
| 8. Recommendation | Final BUY signal | **YES** ✅ | `recommended_shares` |

**All 8 steps now save data.** Implementation completed.

---

## Target State: Save at EVERY Step

| Step | Filter | Saves to DB? | Collection | What's Saved |
|------|--------|:------------:|------------|-------------|
| 0. Macro Gate | Nifty 50 check | **YES** | `scan_runs` | Nifty EMA/MACD status, safe or not |
| 1. Symbol Fetch | Pre-filter | **YES** | `scan_runs` | Total symbols fetched, how many survived |
| 2. Data Fetch | yfinance OHLCV | **YES** | `scan_runs` | Fetch success/fail counts |
| 3. Technical Score | 5 strategies | **YES** | `analysis_snapshots` | Per-stock: all 5 signals + score |
| 4. Swing Gates | 4 gates | **YES** | `swing_gate_results` | Per-stock: each gate pass/fail + values |
| 5. Entry Patterns | 5 patterns | **YES** | `trade_signals` | Per-stock: which patterns fired |
| 6. Exit Rules | SL/TP/Trail | **YES** | `trade_signals` | Per-stock: exact exit levels |
| 7. Backtest | Metrics | **YES** | `backtest_results` | Profit Factor, Win Rate, etc. |
| 8. Recommendation | Final BUY | **YES** | `recommended_shares` | Full trade plan |

---

## Implementation: Where to Add the Saves

### Save 0: Macro Gate → `scan_runs`

**File:** `run_analysis.py` → `check_macro_regime()`

```python
# BEFORE (current code, ~line 69-119):
def check_macro_regime(self) -> bool:
    # ... checks Nifty 50 ...
    return is_safe

# AFTER: Create scan_run and store macro result
def check_macro_regime(self) -> bool:
    # ... existing Nifty 50 checks ...
    
    macro_regime = {
        'nifty50_safe': is_safe,
        'nifty_close': float(current_close),
        'nifty_ema20': float(current_ema20),
        'nifty_macd': float(current_macd),
        'nifty_signal': float(current_signal),
        'price_above_ema': current_close > current_ema20,
        'macd_bullish': current_macd > current_signal,
        'reasons': reasons,
    }
    
    # Create scan run document
    config_snapshot = {
        'enabled_strategies': [k for k, v in config.STRATEGY_CONFIG.items() if v],
        'analysis_weights': dict(config.ANALYSIS_WEIGHTS),
        'min_recommendation_score': config.MIN_RECOMMENDATION_SCORE,
        'historical_data_period': config.HISTORICAL_DATA_PERIOD,
    }
    self.scan_run_id = self.persistence.create_scan_run(config_snapshot, macro_regime)
    
    return is_safe
```

### Save 1+2: Symbol Fetch + Data Fetch → `scan_runs` (update)

**File:** `run_analysis.py` → `_run_multiprocessing_pipeline()`

```python
# After Phase 1 completes (~line 270):
phase1_time = (datetime.now() - phase1_start).total_seconds()

# ADD: Update scan_run with fetch results
if hasattr(self, 'scan_run_id') and self.scan_run_id:
    self.persistence.complete_scan_run(self.scan_run_id, {
        'phase': 'data_fetch',
        'total_symbols_requested': total_stocks,
        'data_fetch_success': len(fetched_data),
        'data_fetch_failed': fetch_failed,
        'phase1_duration_seconds': round(phase1_time, 1),
    })
```

### Save 3: Technical Score → `analysis_snapshots`

**File:** `run_analysis.py` → Phase 3 save loop (~line 336-360)

```python
# Currently in Phase 3 loop:
for r in results:
    if r['success']:
        analysis_result = r['result']
        
        # ADD: Save analysis snapshot for EVERY stock (pass or fail)
        self.persistence.save_analysis_snapshot(
            analysis_result, 
            scan_run_id=getattr(self, 'scan_run_id', None)
        )
        
        # ... existing save_backtest_results() and save_recommendation() ...
```

### Save 4: Swing Gates → `swing_gate_results`

**File:** `scripts/swing_trading_signals.py` → `analyze_swing_opportunity()`

This is where the 4 gates are evaluated. The method needs to **return gate details** so they can be saved.

```python
# Currently returns:
return {
    'recommendation': 'BUY' or 'HOLD',
    'confidence': 0.72,
    'patterns': [...],
    ...
}

# CHANGE: Also return gate_results dict
return {
    'recommendation': 'BUY' or 'HOLD',
    'confidence': 0.72,
    'patterns': [...],
    'gate_results': {                          # ADD THIS
        'all_gates_passed': True/False,
        'gate_1_trend': {
            'passed': True/False,
            'adx': 24.5,
            'price_above_sma200': True,
            'sma50_above_sma200': True,
            'di_positive': True,
        },
        'gate_2_mtf': {
            'passed': True/False,
            'weekly_trend_up': True,
            'daily_rsi': 52.3,
        },
        'gate_3_volatility': {
            'passed': True/False,
            'atr_percentile': 42.0,
        },
        'gate_4_volume': {
            'passed': True/False,
            'obv_rising': True,
            'volume_zscore': 1.3,
        },
    },
}
```

Then in the save loop:

```python
# In run_analysis.py Phase 3 save loop:
gate_results = analysis_result.get('gate_results')
if gate_results:
    self.persistence.save_swing_gate_results(
        symbol=analysis_result['symbol'],
        gate_results=gate_results,
        scan_run_id=getattr(self, 'scan_run_id', None)
    )
```

### Save 5+6: Entry Patterns + Exit Rules → `trade_signals`

**File:** `scripts/swing_trading_signals.py` → pattern detection section

```python
# Currently returns patterns as part of the analysis result.
# The trade_logic.py calculates exit levels.
# Both need to be combined and returned.

# In run_analysis.py Phase 3 save loop:
trade_plan = analysis_result.get('trade_plan', {})
if analysis_result.get('is_recommended') and trade_plan:
    self.persistence.save_trade_signal(
        symbol=analysis_result['symbol'],
        signal_data={
            'entry_price': trade_plan.get('buy_price'),
            'entry_pattern': analysis_result.get('entry_pattern', 'unknown'),
            'patterns': analysis_result.get('detected_patterns', {}),
            'stop_loss': trade_plan.get('stop_loss'),
            'take_profit_1': trade_plan.get('take_profit_1'),
            'take_profit_2': trade_plan.get('take_profit_2'),
            'trailing_stop_distance': trade_plan.get('trailing_stop_distance'),
            'atr': trade_plan.get('atr'),
            'risk_per_share': trade_plan.get('risk_per_share'),
            'risk_reward_1': trade_plan.get('risk_reward_1'),
            'risk_reward_2': trade_plan.get('risk_reward_2'),
        },
        scan_run_id=getattr(self, 'scan_run_id', None)
    )
```

### Save Final: Complete scan_run with summary

**File:** `run_analysis.py` → end of Phase 3

```python
# At the end of the pipeline:
if hasattr(self, 'scan_run_id') and self.scan_run_id:
    self.persistence.complete_scan_run(self.scan_run_id, {
        'duration_seconds': round(total_time, 1),
        'total_symbols_scanned': len(results),
        'data_fetch_success': len(fetched_data),
        'data_fetch_failed': fetch_failed,
        'recommendations_generated': recommended_count,
        'failed_analysis': failed_count,
        'phase1_seconds': round(phase1_time, 1),
        'phase2_seconds': round(phase2_time, 1),
        'phase3_seconds': round(phase3_time, 1),
        'top_hold_blockers': dict(hold_reason_counter.most_common(10)),
    })
```

---

## Funnel Query Example

After implementation, you can query the funnel:

```python
from database import get_mongodb
db = get_mongodb()

# Get latest scan run
run = db.scan_runs.find_one(sort=[('started_at', -1)])
run_id = run['_id']

# Funnel counts
total_scanned = db.analysis_snapshots.count_documents({'scan_run_id': run_id})
tech_passed = db.analysis_snapshots.count_documents({'scan_run_id': run_id, 'technical_score': {'$gte': 0.40}})
gates_passed = db.swing_gate_results.count_documents({'scan_run_id': run_id, 'all_gates_passed': True})
signals_found = db.trade_signals.count_documents({'scan_run_id': run_id})
recommended = db.recommended_shares.count_documents({})  # latest

print(f"""
Scan Funnel:
  Total scanned:     {total_scanned}
  Tech score >= 0.4: {tech_passed}
  All gates passed:  {gates_passed}
  Trade signals:     {signals_found}
  Final BUY:         {recommended}
""")

# Why was RELIANCE rejected?
snap = db.analysis_snapshots.find_one({'scan_run_id': run_id, 'symbol': 'RELIANCE'})
print(f"RELIANCE score: {snap['technical_score']}")
print(f"Hold reasons: {snap['hold_reasons']}")

gates = db.swing_gate_results.find_one({'scan_run_id': run_id, 'symbol': 'RELIANCE'})
print(f"Gate 1 (Trend): {'PASS' if gates['gate_1_trend']['passed'] else 'FAIL'}")
print(f"Gate 4 (Volume): {'PASS' if gates['gate_4_volume']['passed'] else 'FAIL'}")
```

---

## Implementation Priority

| Priority | Task | File to Edit | Effort |
|----------|------|-------------|--------|
| 1 | Wire `create_scan_run()` in `check_macro_regime()` | `run_analysis.py` | 10 min |
| 2 | Wire `save_analysis_snapshot()` in Phase 3 loop | `run_analysis.py` | 10 min |
| 3 | Return `gate_results` from `analyze_swing_opportunity()` | `swing_trading_signals.py` | 30 min |
| 4 | Wire `save_swing_gate_results()` in Phase 3 loop | `run_analysis.py` | 10 min |
| 5 | Wire `save_trade_signal()` for recommended stocks | `run_analysis.py` | 10 min |
| 6 | Wire `complete_scan_run()` at end of pipeline | `run_analysis.py` | 10 min |
| 7 | Add funnel summary query to frontend API | `app.py` routes | 30 min |

**Total estimated effort: ~2 hours**
