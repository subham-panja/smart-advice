# Ultimate Backtesting Validation Plan — COMPLETED

## Goal
Achieve **90%+ confidence** that a strategy has a real edge (not overfit) before risking any capital. This plan combines execution realism, statistical validation, and stress testing.

**Status**: All 6 phases implemented and operational.

---

## Implementation Summary

### Phase 1: Execution Realism — COMPLETED
**Files**: `backend/scripts/portfolio_backtest_engine.py`, `backend/config.py`

All realistic costs implemented:
- Indian market transaction costs (STT, stamp duty, SEBI charges, exchange tx charge, GST)
- Slippage on entry (0.15%) and exit (0.15%)
- Gap risk modeling on stop-loss exits (30% chance, 0.5x-2.0x ATR severity)
- Brokerage charges (0.03% per trade)
- Liquidity/volume-based fill check

Total round-trip cost: ~0.50-0.65% per trade.

### Phase 2: Statistical Validation — COMPLETED
**File**: `backend/scripts/validation_tests.py`

Three tests implemented:
1. **Deflated Sharpe Ratio (DSR)** — Corrects observed Sharpe for multiple testing bias. Pass if DSR > 0.
2. **Monte Carlo Permutation Test** — Randomizes returns 5000x, computes p-value. Pass if p < 0.05.
3. **Minimum Track Record (MLRS)** — Computes required vs actual days. Pass if actual > required.

### Phase 3: Stress Testing — COMPLETED
**File**: `backend/scripts/stress_tests.py`

Three sub-tests:
1. **Regime Tests** (5 periods): Bull, Bear, Crash, Recovery, Sideways
2. **Parameter Sensitivity** (13 runs): ±20% on 6 key params (ATR stop, time stop, RSI, volume ratio, max positions, risk per trade)
3. **Cost Sensitivity** (4 scenarios): Ultra-low, Realistic, High, Extreme brokerage/slippage

### Phase 4: Trade Diagnostics — COMPLETED
**File**: `backend/scripts/trade_diagnostics.py`

Analyzes:
- Trade distribution (hold time, consecutive losses, biggest winners/losers)
- Exit reason breakdown (SL, target, time-stop, O'Neil, delisted)
- Equity curve analysis (underwater chart, recovery time, monthly returns heatmap)
- Signal quality (gate pass rates, false positive rate, pattern effectiveness)

### Phase 5: Confidence Scoring — COMPLETED
**File**: `backend/scripts/confidence_scorer.py`

Composite score (0-100) from 7 weighted components:
| Component | Weight | Threshold |
|-----------|--------|-----------|
| Walk-Forward Robustness | 20% | robustness + % positive CAGR |
| Deflated Sharpe Ratio | 15% | DSR confidence % |
| Monte Carlo Permutation | 15% | (1 - p_value) * 100 |
| Stress Test Pass Rate | 15% | % regime tests passed |
| Parameter Stability | 10% | % param variations passed |
| Cost Resilience | 10% | % scenarios with CAGR > 12% |
| Data Sufficiency | 5% | actual days / required days |

**Interpretation:**
| Score | Confidence | Haircut | Action |
|-------|-----------|---------|--------|
| 80-100 | Very High | 15% | Start with ₹10K, scale up quickly |
| 65-79 | High | 30% | Start with ₹10K, scale after 3 months |
| 50-64 | Moderate | 50% | Paper trade 3 months first |
| <50 | Low | 75% | Do not trade — refine strategy |

Realistic CAGR = base_cagr * (1 - haircut)

### Phase 6: Master Runner — COMPLETED
**File**: `backend/scripts/run_ultimate_backtest.py`

Orchestrates all 6 phases with:
- Timer and ETA estimation
- Chartink result caching across phases
- vectorbt indicator pre-computation (O(1) IndicatorStore lookups)
- Sequential walk-forward (no multiprocessing overhead)
- MongoDB persistence for all results
- Table-formatted output for all metrics
- Optional Telegram summary

---

## How to Run

```bash
cd backend
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --max-stocks 10 --mc-iterations 8
```

**Arguments:**
- `--strategy`: Strategy name (default: Swing_Trading)
- `--months`: Lookback months (default: 120 = 10y)
- `--max-stocks`: Max stocks to test (default: 50)
- `--mc-iterations`: Walk-forward MC iterations per window (default: 8)
- `--skip-wf`: Skip walk-forward analysis
- `--skip-stress`: Skip stress tests
- `--telegram`: Send summary to Telegram bot

**Data**: Fetches `period="max"` from yfinance (all available data). Simulation range controlled by `--months`.

---

## Benchmark Results

All strategies tested with 10y lookback, 10 stocks, 8 MC iterations, no skips:

| Strategy | CAGR | Final Value | Max DD | Win Rate | Profit Factor | Confidence | Realistic CAGR |
|----------|------|-------------|--------|----------|---------------|------------|----------------|
| Swing_Trading | 26.13% | ₹1,017,225 | -12.25% | 51.9% | 19.63 | 55/100 (Moderate) | 13.1% |
| Hybrid_Trading | 1.13% | ₹111,912 | -8.51% | 42.1% | 1.23 | 48/100 (Low) | 0.3% |
| Momentum_Trading | 3.51% | ₹141,084 | -17.21% | 49.3% | 1.47 | 37/100 (Low) | 0.9% |
| Nitin_Triple_Confirm | 0.14% | ₹101,379 | -12.96% | 41.4% | 0.42 | 33/100 (Low) | 0.0% |

**Swing_Trading is the best performing strategy** — 26.13% CAGR with moderate drawdown and highest profit factor.

---

## Performance Optimizations

| Optimization | Impact |
|-------------|--------|
| vectorbt indicator batch pre-computation | O(days x symbols) -> O(1) per date |
| IndicatorStore for O(1) lookups | No TA-Lib during simulation loop |
| Sequential walk-forward (no multiprocessing) | Eliminated pickle serialization overhead |
| Chartink caching across phases | 1 HTTP scan instead of 5 |
| Data fetched once with period="max" | No redundant yfinance downloads |

**Runtime**: ~2 minutes for 10y/10-stock/8-iter with all 6 phases.

---

## Sources & References

- [Lopez de Prado — Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Bailey et al. — Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [Combinatorial Purged Cross Validation](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross)
- [Walk-Forward Analysis Best Practices](https://www.reddit.com/r/algotrading/comments/1t5e9q6/a_real_professional_backtest_is_walkforward/)
- [Permuted Monte Carlo Validation](https://medium.com/@NFS303/validating-trading-strategies-with-permuted-monte-carlo-a-practical-guide-using-roc-momentum-d083180d4213)
- [SSRN — Backtest Overfitting in ML Era](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf?abstractid=4686376&mirid=1)
- [BacktestIndia — Factor Testing on NSE](https://backtestindia.com/blog/backtesting-tool-india-nse-factor-investing)
- [SSRN — Gap Risk & Slippage Modeling](https://papers.ssrn.com/sol3/Delivery.cfm/5278107.pdf?abstractid=5278107&mirid=1)
- [Bigul — Slippage in Indian Algo Trading](https://bigul.co/blog/algo-trading/navigating-slippage-in-algo-trading-your-guide-to-smoother-execution-in-india)

---
*Last Updated: 2026-05-17*
