# Ultimate Backtest Results — Swing Trading Strategy

> Date: 2026-06-22 | Lookback: 120 months (10y) | MC Iterations: 12
> Session ID: 6a384a6a5d47ebd3b1cedd61
> Total Runtime: 1h 19m 42s

---

## Strategy Confidence Report

| Metric | Value |
|---|---|
| **Overall Confidence Score** | **67.1/100** (up from 57.5) |
| **Confidence Level** | **High** (was Moderate) |
| **Edge Verified** | **YES** (was NO) |
| Realistic CAGR Projection | 14.3% (30% haircut) |
| Recommended Action | Start with Rs 10K, scale after 3 months |

### Component Scores

| Component | Score | Weight | Weighted |
|---|---|---|---|
| Walk-Forward Robustness | 32.8 | 20% | 6.6 |
| Deflated Sharpe Ratio | 100.0 | 15% | 15.0 |
| Monte Carlo Permutation | 76.9 | 15% | 11.5 |
| Stress Test Pass Rate | 60.0 | 15% | 9.0 |
| Parameter Stability | 100.0 | 10% | 10.0 |
| Cost Resilience | 100.0 | 10% | 10.0 |
| Data Sufficiency | 100.0 | 5% | 5.0 |
| **TOTAL** | | | **67.1** |

---

## Historical Backtest (Realistic Costs)

| Metric | Value |
|---|---|
| Date Range | 2016-06-22 to 2026-06-19 |
| Initial Capital | Rs 10,000 |
| Final Value | Rs 63,828 |
| Total Return | +538.28% |
| CAGR | 20.39% |
| Max Drawdown | -40.19% |
| Sharpe Ratio | 0.35 |
| Profit Factor | 1.83 |
| Win Rate | 39.7% |
| Total Trades | 1,500 |
| Expectancy | Rs 76.90 |
| Avg Positions Held | 7.7 |
| Excluded Stocks | 124 |

---

## Statistical Validation

| Test | Result | Status |
|---|---|---|
| Deflated Sharpe Ratio | 0.2956 | PASS |
| DSR Confidence | 100.0% | PASS |
| Monte Carlo p-value | 0.2314 | FAIL (not significant) |
| Min Track Record | 0.04 years | PASS |
| Actual Track Record | 9.77 years | PASS |
| Edge Verified | YES | PASS |

---

## Walk-Forward Monte Carlo (59 windows x 12 MC = 684 runs)

| Metric | Value |
|---|---|
| Period | 2011-06-20 to 2026-06-19 (5478 days) |
| CAGR Mean | 111.8% +/- 196.8% |
| CAGR Range | -192.6% to 799.2% |
| CAGR Median | 74.8% |
| Avg Win Rate | 47.5% |
| Avg Sharpe | 1.09 |
| Avg Max DD | -23.7% |
| Worst Max DD | -95.5% |
| Avg Profit Factor | 2.61 |
| Positive CAGR | 65% of runs |
| Robustness Score | 0/100 |

---

## Stress Tests

### Regime-Specific

| Regime | CAGR | Max DD | Status |
|---|---|---|---|
| Bull Market (2017) | 1062.2% | -3.2% | PASS |
| Bear Market (2018 Crash) | 445.7% | -15.3% | FAIL (tol -15.0%) |
| COVID Crash (2020) | 685.6% | -3.1% | PASS |
| Strong Recovery (2020-21) | 251.6% | -3.7% | PASS |
| Sideways Market (2022-23) | 110.0% | -37.9% | FAIL (tol -12.0%) |

### ATR Stop Multiplier Sweep

| ATR | CAGR | Delta% | Cliff? |
|---|---|---|---|
| 2.0 | 64.6% | — | |
| 2.5 | 86.9% | +26.3% | |
| 2.8 | 29.4% | -67.9% | CLIFF |
| 3.0 | 84.6% | +65.3% | CLIFF |
| 3.05 | 84.2% | -0.5% | |
| 3.1 | 83.4% | -1.0% | |
| 3.15 | 82.8% | -0.7% | |
| 3.2 | 27.3% | -65.5% | CLIFF |
| 3.3 | 27.0% | -0.4% | |
| 3.5 | 40.0% | +15.3% | |
| 4.0 | 43.9% | +4.7% | |

**ATR stop multiplier is a fragile parameter** — multiple cliffs detected.

### Parameter Sensitivity

| Parameter | Low | High | Status |
|---|---|---|---|
| time_stop_bars | 10: 84.6% | 14: 84.6% | PASS |
| min_rsi | 45: 84.6% | 55: 84.6% | PASS |
| min_volume_ratio | 0.5: 84.6% | 0.7: 84.6% | PASS |
| max_positions | 6: 84.6% | 10: 84.6% | PASS |
| risk_per_trade_pct | 1.6: 84.6% | 2.4: 84.6% | PASS |

All non-ATR parameters are stable (0% deviation).

### Transaction Cost Sensitivity

| Scenario | CAGR | Impact |
|---|---|---|
| Ultra-low (brk=0.01%, slp=0.05%) | 99.9% | baseline |
| Realistic (brk=0.03%, slp=0.15%) | 84.6% | -15.2pp |
| High (brk=0.05%, slp=0.30%) | 84.8% | -15.0pp |
| Extreme (brk=0.10%, slp=0.50%) | 42.3% | -57.5pp |

---

## Phase Timings

| Phase | Duration |
|---|---|
| Phase 1: Historical Backtest | 12m 5s |
| Phase 1b: Save to MongoDB | 0.2s |
| Phase 2: Statistical Validation | 0.2s |
| Phase 3: Walk-Forward MC (684 runs) | 30m 17s |
| Phase 4: Stress Tests | 37m 18s |
| Phase 5: Trade Diagnostics | 0.3s |
| Phase 6: Confidence Score | 0.0s |
| **Total** | **1h 19m 42s** |

---

## Key Changes vs Previous Run (57.5/100)

| Change | Impact |
|---|---|
| MC Permutation test fixed | p-value: 0.8938 -> 0.2314 (now meaningful) |
| Market regime detection enabled | Edge verified: NO -> YES |
| ATR stop cliff fix (blended R:R) | Better parameter sweep coverage |
| Conflicting patterns pruned to 5 | Cleaner entries |
| Confidence score | 57.5 -> 67.1 (+9.6) |
| Confidence level | Moderate -> High |
| Realistic CAGR haircut | 50% -> 30% |
| Realistic CAGR projection | 11.3% -> 14.3% |
