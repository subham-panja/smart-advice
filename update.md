# Smart Advice — Backend Refactoring Plan v2

> **Goal:** Transform from indicator-soup into a focused swing trading engine with proper per-trade metrics, multi-stage database persistence, and a clear execution flow.

---

## Part 1: Config Cleanup — Disable Everything Unnecessary

### config.py — `STRATEGY_CONFIG` (keep 5 of 12)

```python
STRATEGY_CONFIG = {
    # ═══ KEEP ENABLED (5 core swing signals) ═══
    'RSI_Overbought_Oversold': True,       # Oversold bounce entry filter
    'MACD_Signal_Crossover': True,         # Momentum confirmation
    'EMA_Crossover_12_26': True,           # Fast trend confirmation
    'ADX_Trend_Strength': True,            # Core trend gate filter
    'On_Balance_Volume': True,             # Volume confirmation gate

    # ═══ DISABLE (redundant / handled by swing gates) ═══
    'MA_Crossover_50_200': False,          # Too slow for swing (200-day lag)
    'Bollinger_Band_Breakout': False,      # Squeeze logic in swing patterns
    'ATR_Volatility': False,               # Handled inside swing exit rules
    'SMA_Crossover_20_50': False,          # Redundant with EMA 12/26
    'Stochastic_Overbought_Oversold': False,  # Adds noise, not edge
    'Volume_Breakout': False,              # Handled inside swing patterns
    'Support_Resistance_Breakout': False,  # Heavy compute, marginal value
    'Multi_Timeframe_RSI': False,          # Import hang issues

    # ═══ ALL OTHERS REMAIN FALSE ═══
    # ... (no changes to already-disabled strategies)
}
```

### config.py — `ANALYSIS_CONFIG`

```python
ANALYSIS_CONFIG = {
    'technical_analysis': True,      # CORE — keep
    'fundamental_analysis': False,   # DISABLE — not needed for swing entry
    'sentiment_analysis': False,     # Already disabled
    'sector_analysis': False,        # Already disabled
    'market_regime_detection': False, # Already disabled
    'market_microstructure': False,   # Already disabled
    'alternative_data': False,        # Already disabled
    'backtesting': True,             # CORE — keep
    'risk_management': True,         # CORE — keep
    'tca_analysis': False            # Already disabled
}
```

### config.py — `ANALYSIS_WEIGHTS`

```python
ANALYSIS_WEIGHTS = {
    'technical': 1.00,     # 100% technical for swing
    'fundamental': 0.00,   # Disabled
    'sentiment': 0.00,     # Disabled
    'sector': 0.00         # Disabled
}
```

### config.py — `MIN_RECOMMENDATION_SCORE`

```python
MIN_RECOMMENDATION_SCORE = 0.40   # Require 2/5 strategies to agree (was 0.15)
```

---

## Part 2: Replace CAGR with Proper Swing Trading Metrics

### Why CAGR is Wrong

| Problem | Impact |
|---|---|
| Assumes continuous investment for 365 days | Swing capital is idle 60-80% of the time |
| Annualizing a 5-day 8% trade = 10,000%+ | Misleading, breaks decision logic |
| Ignores drawdowns completely | "Good" CAGR can hide 40% drawdowns |
| Designed for buy-and-hold | Fundamentally wrong for episodic trading |

### New Primary Metrics

| # | Metric | Formula | Target | Role |
|---|---|---|---|---|
| 1 | **Profit Factor** | `Gross Profit / Gross Loss` | > 1.5 | PRIMARY — replaces CAGR for performance rating |
| 2 | **Expectancy (R)** | `(Win% x AvgWin_R) - (Loss% x AvgLoss_R)` | > 0.2R | PRIMARY — average R earned per trade |
| 3 | **Win Rate** | `Winning / Total Trades` | 45-65% | How often you're right |
| 4 | **Avg R-Multiple** | `Mean(trade_pnl / initial_risk)` | > 1.0R | Normalized return per trade |
| 5 | **Max Drawdown** | `Max peak-to-trough %` | < 15% | Worst-case pain |
| 6 | **Recovery Factor** | `Net Profit / Max Drawdown` | > 3.0 | How fast you bounce back |
| 7 | **Avg Holding Period** | `Mean(exit_bar - entry_bar)` | 3-15 bars | Capital rotation speed |
| 8 | **Max Consecutive Losses** | Count | < 5 | Position sizing validation |

### Keep as Secondary (display only)

- **CAGR** — useful to compare vs buy-and-hold benchmark
- **Sharpe Ratio** — industry-standard risk-adjusted metric
- **Sortino Ratio** — penalizes only downside volatility

---

## Part 3: Code Changes in `backtesting_runner.py`

### Add to `_calculate_strategy_metrics()`

```python
# Profit Factor (NEW — primary metric)
won_pnl = trades.get('won', {}).get('pnl', {}).get('total', 0)
lost_pnl = abs(trades.get('lost', {}).get('pnl', {}).get('total', 0))
profit_factor = (won_pnl / lost_pnl) if lost_pnl > 0 else (float('inf') if won_pnl > 0 else 0)

# Recovery Factor (NEW)
recovery_factor = (pnl_total / max_drawdown) if max_drawdown > 0 else 0

# Add to return dict:
'profit_factor': round(float(profit_factor), 2),
'recovery_factor': round(float(recovery_factor), 2),
```

### Change `_generate_backtest_summary()` — Performance Rating

```python
# OLD (CAGR-based):
if avg_expectancy > 500 and win_rate > 60: performance_rating = 'Excellent'

# NEW (Profit Factor-based):
avg_pf = combined_metrics.get('avg_profit_factor', 0)
avg_wr = combined_metrics.get('avg_win_rate', 0)

if avg_pf > 2.0 and avg_wr > 55:
    performance_rating = 'Excellent'
elif avg_pf > 1.5 and avg_wr > 45:
    performance_rating = 'Good'
elif avg_pf > 1.0:
    performance_rating = 'Average'
else:
    performance_rating = 'Poor'

# Recommendation gate:
recommendation = ('BUY' if avg_pf > 1.5 and avg_wr > 45
                   else 'HOLD' if avg_pf > 1.0
                   else 'SELL')
```

### Add to `_calculate_combined_metrics()`

```python
avg_profit_factor = sum(r.get('profit_factor', 0) for r in successful_results) / len(successful_results)
avg_recovery_factor = sum(r.get('recovery_factor', 0) for r in successful_results) / len(successful_results)

# Return these in the combined dict
'avg_profit_factor': round(avg_profit_factor, 2),
'avg_recovery_factor': round(avg_recovery_factor, 2),
```

---

## Part 4: New MongoDB Collections (Multi-Stage Data)

### Current State (only 2 collections)

```
super_advice
  ├── recommended_shares    ← only BUY signals saved
  └── backtest_results      ← only combined metrics saved
```

### New State (6 collections)

```
super_advice
  ├── recommended_shares        ← BUY/STRONG_BUY signals (keep)
  ├── backtest_results          ← backtest metrics (keep, add profit_factor)
  ├── analysis_snapshots [NEW]  ← EVERY stock analyzed (pass or fail)
  ├── swing_gate_results [NEW]  ← 4-gate pass/fail per stock per run
  ├── trade_signals [NEW]       ← detected entry patterns + exit levels
  └── scan_runs [NEW]           ← metadata about each scan run
```

### Collection: `analysis_snapshots` (NEW)

Saves data for EVERY stock analyzed, not just recommended ones.
This lets you debug why stocks were rejected.

```python
{
    "symbol": "RELIANCE",
    "company_name": "Reliance Industries",
    "scan_run_id": ObjectId("..."),     # Links to scan_runs
    "analyzed_at": ISODate("2026-04-24T00:00:00Z"),

    # Scores
    "technical_score": 0.60,
    "combined_score": 0.45,

    # Decision
    "recommendation": "HOLD",           # or BUY, STRONG_BUY, NO_SIGNAL
    "is_recommended": false,
    "hold_reasons": ["Volume confirmation failed", "ADX below threshold"],

    # Strategy signals (which of the 5 fired)
    "strategy_signals": {
        "RSI_Overbought_Oversold": {"signal": 1, "type": "BUY"},
        "MACD_Signal_Crossover": {"signal": -1, "type": "SELL/HOLD"},
        "EMA_Crossover_12_26": {"signal": 1, "type": "BUY"},
        "ADX_Trend_Strength": {"signal": -1, "type": "SELL/HOLD"},
        "On_Balance_Volume": {"signal": 1, "type": "BUY"}
    },
    "positive_signals": 3,
    "total_signals": 5,

    # Price data snapshot
    "price_snapshot": {
        "close": 2850.50,
        "sma_200": 2720.30,
        "ema_21": 2835.10,
        "rsi_14": 52.3,
        "adx_14": 18.5,
        "atr_14": 45.2,
        "volume": 8500000,
        "volume_avg_20": 7200000
    }
}
```

### Collection: `swing_gate_results` (NEW)

Detailed gate-by-gate results for debugging.

```python
{
    "symbol": "RELIANCE",
    "scan_run_id": ObjectId("..."),
    "analyzed_at": ISODate("2026-04-24T00:00:00Z"),

    "all_gates_passed": false,

    "gate_1_trend": {
        "passed": false,
        "adx": 18.5,
        "adx_threshold": 20,
        "price_above_sma200": true,
        "sma50_above_sma200": true,
        "di_plus_above_minus": true,
        "reason": "ADX 18.5 below minimum 20"
    },

    "gate_2_mtf": {
        "passed": true,
        "weekly_trend_up": true,
        "daily_trend_up": true,
        "weekly_rsi": 55.2,
        "daily_rsi": 52.3
    },

    "gate_3_volatility": {
        "passed": true,
        "atr": 45.2,
        "atr_percentile": 42.0,
        "range": "20-80",
        "volatility_expanding": false
    },

    "gate_4_volume": {
        "passed": false,
        "obv_trending_up": false,
        "volume_zscore": 0.3,
        "zscore_threshold": 1.0,
        "reason": "Neither OBV rising nor volume spike"
    }
}
```

### Collection: `trade_signals` (NEW)

Only for stocks that PASS all gates and have entry patterns.

```python
{
    "symbol": "TCS",
    "scan_run_id": ObjectId("..."),
    "signal_date": ISODate("2026-04-24T00:00:00Z"),
    "status": "ACTIVE",                 # ACTIVE, EXPIRED, EXECUTED, STOPPED

    # Entry
    "entry_price": 3450.00,
    "entry_pattern": "pullback_to_ema",
    "pattern_strength": 0.85,
    "signal_strength": 0.72,

    # Detected patterns
    "patterns": {
        "pullback_to_ema": true,
        "bb_squeeze_breakout": false,
        "macd_zero_cross": true,
        "higher_low_structure": false,
        "volume_breakout": false
    },

    # Exit levels (ATR-based)
    "stop_loss": 3382.20,
    "take_profit_1": 3517.80,
    "take_profit_2": 3585.60,
    "trailing_stop_distance": 90.40,
    "time_stop_bars": 15,

    # Risk metrics
    "risk_per_share": 67.80,
    "risk_reward_1": 1.0,
    "risk_reward_2": 2.0,
    "atr": 45.2,

    # Tracking (updated daily if signal is ACTIVE)
    "bars_since_entry": 0,
    "current_pnl_pct": 0.0,
    "tp1_hit": false,
    "highest_since_entry": 3450.00
}
```

### Collection: `scan_runs` (NEW)

Metadata about each analysis run for audit/debugging.

```python
{
    "run_id": ObjectId("..."),
    "started_at": ISODate("2026-04-24T00:00:00Z"),
    "completed_at": ISODate("2026-04-24T00:05:30Z"),
    "duration_seconds": 330,

    "config_snapshot": {
        "enabled_strategies": ["RSI", "MACD", "EMA", "ADX", "OBV"],
        "analysis_weights": {"technical": 1.0},
        "min_recommendation_score": 0.40,
        "historical_data_period": "5y"
    },

    "macro_regime": {
        "nifty50_safe": true,
        "nifty_ema20_above": true,
        "nifty_macd_bullish": true
    },

    "results_summary": {
        "total_symbols_scanned": 500,
        "data_fetch_success": 485,
        "data_fetch_failed": 15,
        "all_gates_passed": 42,
        "entry_patterns_found": 12,
        "recommendations_generated": 8,
        "avg_time_per_stock_seconds": 0.66
    },

    "top_hold_blockers": {
        "ADX below threshold": 180,
        "Price below SMA200": 120,
        "Volume confirmation failed": 85
    }
}
```

### config.py — New Collections Config

```python
MONGODB_COLLECTIONS = {
    'recommended_shares': 'recommended_shares',
    'backtest_results': 'backtest_results',
    'analysis_snapshots': 'analysis_snapshots',     # NEW
    'swing_gate_results': 'swing_gate_results',     # NEW
    'trade_signals': 'trade_signals',               # NEW
    'scan_runs': 'scan_runs',                       # NEW
}
```

---

## Part 5: The Complete Flow (How It Works)

```
┌──────────────────────────────────────────────────────────────────────┐
│                      run_analysis.py                                 │
│                                                                      │
│  START                                                               │
│    │                                                                 │
│    ▼                                                                 │
│  ┌─────────────────────────────┐                                     │
│  │ 0. MACRO GATE               │ ◄── check_macro_regime()            │
│  │    Nifty50 > 20 EMA?        │     If NO → abort entire scan       │
│  │    MACD bullish?            │                                     │
│  └──────────┬──────────────────┘     Saves to: scan_runs             │
│             │ PASS                                                   │
│             ▼                                                        │
│  ┌─────────────────────────────┐                                     │
│  │ 1. FETCH DATA               │ ◄── Phase 1: 16 threads            │
│  │    500+ NSE symbols         │     yfinance → 5y daily OHLCV      │
│  │    Pre-filtered by volume,  │                                     │
│  │    price, market cap        │                                     │
│  └──────────┬──────────────────┘                                     │
│             │                                                        │
│             ▼                                                        │
│  ┌─────────────────────────────┐                                     │
│  │ 2. TECHNICAL SCORING        │ ◄── Phase 2: 8 processes            │
│  │    Run 5 enabled strategies │     strategy_evaluator.py           │
│  │                             │                                     │
│  │    RSI: oversold bounce? ──►│──┐                                  │
│  │    MACD: bullish cross?  ──►│──┤                                  │
│  │    EMA 12/26: crossover? ──►│──┤  tech_score = positives / 5      │
│  │    ADX: trending + DI?   ──►│──┤                                  │
│  │    OBV: rising?          ──►│──┘                                  │
│  │                             │                                     │
│  │    tech_score >= 0.40?      │     Saves to: analysis_snapshots    │
│  └──────────┬──────────────────┘     (EVERY stock, pass or fail)     │
│             │ PASS                                                   │
│             ▼                                                        │
│  ┌─────────────────────────────┐                                     │
│  │ 3. SWING TRADING GATES      │ ◄── swing_trading_signals.py        │
│  │                             │                                     │
│  │  Gate 1: TREND FILTER       │     ADX >= 20, Price > SMA200       │
│  │          ─────────────      │     SMA50 > SMA200, +DI > -DI       │
│  │                             │                                     │
│  │  Gate 2: MTF CONFIRMATION   │     Weekly SMA20 > SMA50            │
│  │          ────────────────   │     Weekly RSI > 50                  │
│  │                             │     Daily SMA20 rising, RSI > 40    │
│  │                             │                                     │
│  │  Gate 3: VOLATILITY GATE    │     ATR percentile 20-80%           │
│  │          ───────────────    │     (not too calm, not too wild)     │
│  │                             │                                     │
│  │  Gate 4: VOLUME CONFIRM     │     OBV slope > 0 (10-bar)          │
│  │          ──────────────     │     OR Volume Z-score > 1.0         │
│  │                             │                                     │
│  │    ALL 4 gates passed?      │     Saves to: swing_gate_results    │
│  └──────────┬──────────────────┘     (EVERY stock, pass or fail)     │
│             │ ALL PASS                                               │
│             ▼                                                        │
│  ┌─────────────────────────────┐                                     │
│  │ 4. ENTRY PATTERN DETECTION  │ ◄── swing_trading_signals.py        │
│  │                             │                                     │
│  │  P1: Pullback to 21 EMA    │     Low touches EMA, RSI 40-60      │
│  │  P2: BB Squeeze Breakout   │     Tight bands + price breakout    │
│  │  P3: MACD Zero Cross       │     MACD crosses signal above 0     │
│  │  P4: Higher-Low Structure  │     2+ ascending swing lows         │
│  │  P5: Volume Breakout       │     Close > 20-bar high + 1.5x vol  │
│  │                             │                                     │
│  │    At least 1 pattern?      │                                     │
│  └──────────┬──────────────────┘                                     │
│             │ YES                                                    │
│             ▼                                                        │
│  ┌─────────────────────────────┐                                     │
│  │ 5. EXIT RULES (ATR-BASED)   │ ◄── swing_trading_signals.py        │
│  │                             │                                     │
│  │  Stop Loss:  Entry - 1.5xATR│                                     │
│  │  TP1:        Entry + 1.5xATR│  ← move SL to breakeven            │
│  │  TP2:        Entry + 3.0xATR│  ← full exit                       │
│  │  Trail Stop: 2.0xATR        │  ← ratchets up                     │
│  │  Time Stop:  15 bars        │  ← exit if no progress             │
│  │                             │                                     │
│  │  R:R >= 1.5?               │                                     │
│  └──────────┬──────────────────┘                                     │
│             │ YES                                                    │
│             ▼                                                        │
│  ┌─────────────────────────────┐                                     │
│  │ 6. BACKTEST VALIDATION      │ ◄── backtesting_runner.py            │
│  │                             │                                     │
│  │  Run strategy on 5y data    │                                     │
│  │  Metrics:                   │                                     │
│  │    Profit Factor > 1.5?     │  ← PRIMARY gate                     │
│  │    Win Rate > 45%?          │                                     │
│  │    Max Drawdown < 15%?      │                                     │
│  │    Expectancy(R) > 0.2?     │                                     │
│  │                             │     Saves to: backtest_results      │
│  └──────────┬──────────────────┘                                     │
│             │ PASS                                                   │
│             ▼                                                        │
│  ┌─────────────────────────────┐                                     │
│  │ 7. SAVE RECOMMENDATION      │                                     │
│  │                             │     Saves to: recommended_shares    │
│  │  recommendation = BUY       │     Saves to: trade_signals         │
│  │  + entry price              │                                     │
│  │  + SL / TP1 / TP2 levels    │                                     │
│  │  + pattern that triggered   │                                     │
│  │  + confidence score         │                                     │
│  └─────────────────────────────┘                                     │
│                                                                      │
│  END — Results visible on frontend                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### Flow Summary (text version)

```
1. Macro Gate     → Is Nifty50 bullish? If no, abort.
2. Fetch Data     → 500 NSE stocks, 5y daily OHLCV
3. Tech Scoring   → Run 5 strategies, need score >= 0.40 (2/5)
4. Swing Gates    → 4 gates (trend, MTF, volatility, volume) ALL must pass
5. Entry Pattern  → At least 1 of 5 patterns detected
6. Exit Rules     → ATR-based SL/TP1/TP2/Trail/TimeStop, R:R >= 1.5
7. Backtest Check → Profit Factor > 1.5, Win Rate > 45%
8. Save           → BUY recommendation + trade plan to MongoDB
```

### What gets saved at each stage

| Stage | Collection | What's Saved | When |
|---|---|---|---|
| Scan start | `scan_runs` | Config, macro gate result, timing | Every run |
| Tech scoring | `analysis_snapshots` | Scores, signals, price snapshot | EVERY stock |
| Swing gates | `swing_gate_results` | 4 gate results with values | EVERY stock |
| Entry detected | `trade_signals` | Pattern, entry, SL, TP levels | Only gate-passed stocks with patterns |
| Backtest done | `backtest_results` | Profit Factor, Win Rate, DD, etc. | Only recommended stocks |
| Final BUY | `recommended_shares` | Full recommendation + trade plan | Only BUY/STRONG_BUY |

---

## Part 6: Implementation Checklist

### Phase 1 — Config Cleanup (15 min)
- [ ] Update `STRATEGY_CONFIG` — disable 7 strategies
- [ ] Update `ANALYSIS_CONFIG` — disable fundamental
- [ ] Update `ANALYSIS_WEIGHTS` — 100% technical
- [ ] Update `MIN_RECOMMENDATION_SCORE` — 0.40
- [ ] Add new collections to `MONGODB_COLLECTIONS`

### Phase 2 — Database + Persistence (1-2 hours)
- [ ] Add `analysis_snapshots` collection + indexes
- [ ] Add `swing_gate_results` collection + indexes
- [ ] Add `trade_signals` collection + indexes
- [ ] Add `scan_runs` collection + indexes
- [ ] Update `persistence_handler.py` with new save methods
- [ ] Update `database.py` with new collection init

### Phase 3 — Metrics Refactor (1 hour)
- [ ] Add `profit_factor` to `_calculate_strategy_metrics()`
- [ ] Add `recovery_factor` to `_calculate_strategy_metrics()`
- [ ] Demote `cagr` to secondary metric
- [ ] Update performance rating to use Profit Factor
- [ ] Update recommendation gate to use Profit Factor

### Phase 4 — Pipeline Integration (1-2 hours)
- [ ] Create `scan_run` document at start of `run_analysis`
- [ ] Save `analysis_snapshot` for every stock in Phase 2
- [ ] Save `swing_gate_results` during swing analysis
- [ ] Save `trade_signals` when entry patterns detected
- [ ] Update `scan_run` with summary at end

### Phase 5 — Validate (30 min)
- [ ] Run `--max-stocks 5 --verbose --single-threaded`
- [ ] Check all 6 MongoDB collections have data
- [ ] Verify rejected stocks have `hold_reasons` in `analysis_snapshots`
- [ ] Verify gate values are saved in `swing_gate_results`
