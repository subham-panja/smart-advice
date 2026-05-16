# Ultimate Backtesting Validation Plan

## Goal
Achieve **90%+ confidence** that the Swing_Trading strategy has a real edge (not overfit) before risking any capital. This plan combines execution realism, statistical validation, and stress testing.

---

## Phase 1: Fix Execution Realism (Critical — Do First)

**Why:** The current backtest has a 3.36% max drawdown over 9 years — impossible for Indian equities. The execution model is missing real-world costs, making results 30-50% too optimistic.

### 1.1 Realistic Cost Model
**File:** `backend/scripts/portfolio_backtest_engine.py` + `backend/config.py`

Add proper Indian market transaction costs:

| Charge | Buy Rate | Sell Rate | Notes |
|--------|----------|-----------|-------|
| Brokerage | 0.03% | 0.03% | Discount broker (Zerodha/Groww) |
| STT | 0% | 0.10% | Only on sell (equity delivery) |
| Exchange tx charge | 0.00325% | 0.00325% | NSE/BSE |
| SEBI charges | 0.0001% | 0.0001% | Regulatory |
| Stamp duty | 0.015% | 0% | Only on buy (Maharashtra) |
| GST 18% | on above charges | on above charges | Tax on charges |
| **Slippage** | **0.15%** | **0.15%** | Realistic mid-cap slippage |
| **Total round-trip cost** | | | **~0.50-0.65% per trade** |

**Changes:**
- `config.py`: Add `STF_TRADES_CONFIG` dict with all rates
- `portfolio_backtest_engine.py`: Replace simple `brokerage = gross_value * self.brokerage` with `_calculate_total_costs(gross_value, is_buy)` method
- Buy slippage: increase from 0.05% → 0.15%
- **Add sell slippage**: `exit_price = close_price * (1 - self.slippage)` (you receive slightly less)

### 1.2 Gap-Down Risk Modeling
**File:** `backend/scripts/portfolio_backtest_engine.py` — `_process_exits()` and `_close_position()`

**Problem:** All exits use `df.loc[date, "Close"]` — assumes you always exit at the close price. In reality:
- Stop losses trigger intraday but fill at next available price
- Stocks gap down 5-15% on bad news (especially during crashes)
- Circuit filters can halt trading entirely

**Solution — Gap-Aware Exit Modeling:**

```python
def _apply_gap_risk(self, symbol, date, trigger_price, reason, df):
    """Apply realistic gap risk on exit fills.

    For stop-loss exits: model gap-down risk using ATR-based distribution.
    For target exits: model gap-up risk (smaller effect).
    """
    atr = self._calculate_atr(df, date)
    atr_pct = atr / trigger_price if trigger_price > 0 else 0

    if "STOP" in reason or "LOSS" in reason:
        # Stop loss fills are vulnerable to gap-down
        # Model: 30% chance of gap fill at worse price
        # Gap severity: 0.5x to 2.0x ATR beyond trigger
        gap_probability = 0.30
        if random.random() < gap_probability:
            gap_severity = random.uniform(0.5, 2.0) * atr
            fill_price = trigger_price - gap_severity
            # Floor at -20% from trigger (circuit breaker limit)
            fill_price = max(fill_price, trigger_price * 0.80)
            return fill_price

    elif "TARGET" in reason:
        # Target fills can gap up slightly in your favor
        gap_probability = 0.15
        if random.random() < gap_probability:
            gap_bonus = random.uniform(0.1, 0.5) * atr
            fill_price = trigger_price + gap_bonus
            return fill_price

    return trigger_price  # No gap, fill at trigger price
```

**Where to apply:**
- In `_process_exits()`: when checking `current_price <= pos.current_stop_loss`, instead of using `current_price`, call `_apply_gap_risk()` to get realistic fill
- In `_close_position()`: apply gap risk before calculating pnl
- Set `random.seed(42)` for reproducible results

### 1.3 Liquidity/Volume-Based Fill Check
**File:** `backend/scripts/portfolio_backtest_engine.py` — `_execute_buy()`

Before buying, check if there's enough volume to fill the order:
```python
# Skip buy if position size > 1% of average daily volume
avg_vol = df["Volume"].tail(20).mean()
if size > avg_vol * 0.01:
    return  # Can't fill this size without moving the market
```

---

## Phase 2: Statistical Validation (Prove the Edge Is Real)

**Why:** A single backtest can be lucky. These methods prove the strategy works across different conditions, not just one period.

### 2.1 Walk-Forward Analysis (Already Built — Run It)
**Command:**
```bash
cd backend
python scripts/run_portfolio_backtest.py --strategy Swing_Trading --walk-forward --mc-iterations 8 --period 10y --max-stocks 50
```

**What it does:**
- Splits 10 years into rolling 6-month windows (90-day step)
- Each window: 8 Monte Carlo iterations with random 70% stock samples
- Tests strategy across different time periods AND different stock universes

**What to look for:**
| Metric | Pass Threshold | Fail = Red Flag |
|--------|---------------|-----------------|
| Robustness Score | >60 | <40 means overfit |
| Positive CAGR % of runs | >70% | <50% means inconsistent |
| Mean CAGR | >15% | <10% means weak edge |
| CAGR std dev | <15% | >20% means unstable |
| Worst Max DD | <-25% | >30% means risky |

### 2.2 Deflated Sharpe Ratio (DSR)
**Concept:** The observed Sharpe ratio is inflated by trying many parameter combinations. The DSR corrects for this "selection bias."

**Formula (Lopez de Prado):**
```
DSR = SR_observed - sqrt((V/N) * 2*log(N_trials))
```
Where:
- `SR_observed` = your observed Sharpe (0.35)
- `N_trials` = number of backtests/variations you ran (estimate: 10-50)
- `V/N` = variance of returns / number of observations

**Implementation:** Add a `calculate_deflated_sharpe()` method that:
1. Takes the daily returns from `daily_snapshots`
2. Estimates N_trials from strategy config complexity
3. Returns DSR — if DSR > 0, the edge is statistically significant

**Pass threshold:** DSR > 0 (edge survives multiple testing correction)

### 2.3 Monte Carlo Permutation Test
**Concept:** Randomly shuffle the trade outcomes. If the shuffled results beat your strategy >5% of the time, your edge isn't real.

**Implementation:**
```python
def monte_carlo_permutation_test(trades, n_simulations=5000):
    """Test if strategy beat is statistically significant."""
    actual_cagr = calculate_cagr_from_trades(trades)

    # Shuffle entry/exit dates, keep trade count same
    better_count = 0
    for _ in range(n_simulations):
        shuffled = shuffle_trade_dates(trades)
        shuffled_cagr = calculate_cagr_from_trades(shuffled)
        if shuffled_cagr > actual_cagr:
            better_count += 1

    p_value = better_count / n_simulations
    return {
        "actual_cagr": actual_cagr,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "confidence": (1 - p_value) * 100,
    }
```

**Pass threshold:** p-value < 0.05 (95%+ confidence that edge is not random)

### 2.4 Minimum Track Record Length (MLRS)
**Concept:** How many trades do you need before the Sharpe ratio is statistically meaningful?

**Formula:**
```
MLRS = [1 + (SR * skewness/2) - (SR^2 * kurtosis/4)] * (Z_alpha / SR)^2
```

**For your strategy:**
- SR = 0.35
- 723 trades over 9 years
- If MLRS < 723, you have enough data

**Pass threshold:** Actual trades > MLRS

---

## Phase 3: Stress Testing (Break It on Purpose)

**Why:** A strategy that only works in bull markets is useless. These tests verify robustness across market conditions.

### 3.1 Regime-Specific Performance
**Command:** Run historical tests on specific market periods:
```bash
# Bull market (2017)
python scripts/run_historical_paper_test.py --end-date 2018-01-01 --months 12 --strategy Swing_Trading

# Bear market (2018 crash)
python scripts/run_historical_paper_test.py --end-date 2019-03-01 --months 15 --strategy Swing_Trading

# COVID crash (2020)
python scripts/run_historical_paper_test.py --end-date 2021-03-01 --months 13 --strategy Swing_Trading

# Strong bull (2020-2021 recovery)
python scripts/run_historical_paper_test.py --end-date 2022-01-01 --months 21 --strategy Swing_Trading

# Sideways market (2022-2023)
python scripts/run_historical_paper_test.py --end-date 2024-01-01 --months 24 --strategy Swing_Trading
```

**What to measure per period:**

| Period | Expected CAGR | Max DD Tolerance | Pass/Fail |
|--------|--------------|-------------------|-----------|
| Bull | >20% | <-10% | CAGR > 15% |
| Bear | >-5% (preserve capital) | <-15% | Loss < 10% |
| Crash | >-10% | <-20% | Better than Nifty |
| Recovery | >25% | <-15% | CAGR > 20% |
| Sideways | >10% | <-12% | Beats FD rate |

### 3.2 Parameter Sensitivity Analysis
**Concept:** Change key parameters by ±20%. If performance collapses, the strategy is overfit to specific values.

**Parameters to test:**
| Parameter | Base | Test Low | Test High |
|-----------|------|----------|-----------|
| ATR stop multiplier | 3.0 | 2.4 | 3.6 |
| Time stop bars | 12 | 10 | 14 |
| RSI filter minimum | 50 | 45 | 55 |
| Volume ratio minimum | 0.6 | 0.5 | 0.7 |
| Max positions | 8 | 6 | 10 |
| Risk per trade % | 2.0 | 1.6 | 2.4 |

**Pass threshold:** CAGR stays within ±30% of base case for all variations

### 3.3 Universe Sensitivity
**Concept:** Does the strategy only work on the current 50 stocks, or does it work on different universes too?

**Tests:**
- Run with Nifty 50 only (large caps)
- Run with Nifty Midcap 150 only
- Run with random 50 stocks from Nifty 500
- Run with only IT sector stocks
- Run with only banking sector stocks

**Pass threshold:** CAGR > 12% in at least 3 of 5 universes

### 3.4 Transaction Cost Sensitivity
**Concept:** How does performance change as costs increase?

| Cost Scenario | Total Round-Trip | Expected CAGR Impact |
|--------------|------------------|---------------------|
| Ultra-low (0.1%) | ₹10 per ₹10K | -1pp |
| Realistic (0.5%) | ₹50 per ₹10K | -3pp |
| High (1.0%) | ₹100 per ₹10K | -6pp |
| Extreme (2.0%) | ₹200 per ₹10K | -12pp |

**Pass threshold:** CAGR > 12% even at 1.0% costs

---

## Phase 4: Trade-Level Diagnostics (Understand the Engine)

**Why:** Summary metrics hide important patterns. These diagnostics reveal how the strategy actually works.

### 4.1 Trade Distribution Analysis
Query MongoDB for all trades and analyze:
- **Trades per year**: Is it consistent or clustered?
- **Holding period distribution**: Median, mean, p90
- **Consecutive losses**: Max losing streak
- **Biggest winners vs losers**: What drove them?
- **Exit reason breakdown**: % by SL, target, time-stop, O'Neil, delisted
- **Win rate by exit reason**: Which exits are most profitable?

### 4.2 Equity Curve Analysis
From `daily_snapshots`:
- **Underwater chart**: Time spent below peak equity
- **Recovery time**: Average days to recover from drawdown
- **Monthly returns**: Heatmap of returns by month/year
- **Best/worst months**: Seasonal patterns
- **Calmar ratio**: CAGR / Max Drawdown

### 4.3 Regime Performance Breakdown
- **Bull market stats**: CAGR, win rate, avg trade, max DD
- **Bear market stats**: CAGR, win rate, avg trade, max DD
- **% time in bull vs bear**: How often does regime switch?
- **Bear market behavior**: Did it stay in cash or trade actively?

### 4.4 Signal Quality Analysis
From the FilterTracker (if `--track-filters` was used):
- **Gate pass rates**: % of stocks passing each gate
- **Signal-to-trade conversion**: % of signals that became actual trades
- **False positive rate**: Signals that resulted in losing trades
- **Pattern effectiveness**: Which entry patterns have highest win rate?

---

## Phase 5: Confidence Scoring (The Final Verdict)

### 5.1 Composite Confidence Score
Combine all validation results into a single score (0-100):

| Component | Weight | How to Score |
|-----------|--------|-------------|
| Walk-Forward Robustness | 20% | (robustness_score / 100) * 20 |
| DSR Significance | 15% | (DSR > 0 ? 15 : 0) |
| Monte Carlo p-value | 15% | ((1 - p_value) * 15) |
| Stress Test Pass Rate | 15% | (% of stress tests passed * 15) |
| Parameter Stability | 10% | (% of param variations within tolerance * 10) |
| Universe Consistency | 10% | (% of universes with CAGR > 12% * 10) |
| Cost Resilience | 10% | (CAGR at 1% cost / base CAGR * 10) |
| Data Sufficiency | 5% | (trades / MLRS capped at 1.0 * 5) |

**Interpretation:**
| Score | Confidence | Action |
|-------|-----------|--------|
| 80-100 | Very High | Start with ₹10K, scale up quickly |
| 65-79 | High | Start with ₹10K, scale after 3 months |
| 50-64 | Moderate | Paper trade 3 months first |
| <50 | Low | Do not trade — refine strategy |

### 5.2 Realistic Return Projection
Given the confidence score, project realistic returns:

```
if score >= 80:
    expected_cagr = base_cagr * 0.85  # 15% hair-cut
elif score >= 65:
    expected_cagr = base_cagr * 0.70  # 30% hair-cut
elif score >= 50:
    expected_cagr = base_cagr * 0.50  # 50% hair-cut
else:
    expected_cagr = base_cagr * 0.25  # 75% hair-cut
```

---

## Phase 6: Implementation Priority

### Immediate (Do This Week)
1. **Phase 1** — Fix execution realism (slippage, STT, gap risk)
2. **Run walk-forward** — Already works, just run it
3. **Trade diagnostics** — Pull MongoDB data, analyze trades

### Short-term (Do This Month)
4. **Stress tests** — Run historical tests on specific periods
5. **Parameter sensitivity** — Test ±20% variations
6. **Monte Carlo permutation** — Implement p-value test

### Medium-term (Next 2 Months)
7. **DSR calculation** — Implement statistical significance test
8. **Universe sensitivity** — Test on different stock universes
9. **Cost sensitivity** — Test at multiple cost levels
10. **Composite score** — Build the final confidence dashboard

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `backend/scripts/portfolio_backtest_engine.py` | Modify | Add gap risk, sell slippage, realistic costs |
| `backend/config.py` | Modify | Add STT_CONFIG, realistic cost rates |
| `backend/scripts/validation_tests.py` | **NEW** | DSR, Monte Carlo permutation, MLRS |
| `backend/scripts/stress_tests.py` | **NEW** | Regime-specific, param sensitivity, universe tests |
| `backend/scripts/trade_diagnostics.py` | **NEW** | Trade distribution, equity curve, regime breakdown |
| `backend/scripts/confidence_scorer.py` | **NEW** | Composite confidence score calculator |
| `backend/scripts/run_ultimate_backtest.py` | **NEW** | Master script that runs all phases |
| `backend/scripts/run_historical_paper_test.py` | Modify | Already fixed (date clamping issue) |

---

## Expected Outcome

After completing all phases, you'll know:

1. **Does the edge exist?** — Walk-forward + DSR + p-value together give 90%+ confidence
2. **How robust is it?** — Stress tests show performance across markets, parameters, universes
3. **What's the realistic return?** — Execution fixes give honest CAGR (not inflated)
4. **What's the real risk?** — Gap modeling shows true max drawdown
5. **Should I trade it?** — Confidence score tells you go/no-go

**If all phases pass:**
- Expected realistic CAGR: 18-22%
- Realistic max drawdown: 12-20%
- Confidence level: 80-90%
- Action: Start with ₹10K, scale after 1-3 months of live confirmation

**If some phases fail:**
- You'll know exactly which part is weak (entry, exit, costs, overfitting)
- You can fix the weak part and re-test
- Better to find flaws in backtest than with real money

---

## Sources & References

- [Lopez de Prado — Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
- [Bailey et al. — Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)
- [Combinatorial Purged Cross Validation](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross)
- [Walk-Forward Analysis Best Practices](https://www.reddit.com/r/algotrading/comments/1t5e9q6/a_real_professional_backtest_is_walkforward/)
- [QuantInsti — Walk Forward Optimisation](https://www.linkedin.com/posts/quantinsti_backtesting-walkforwardoptimisation-quanttrading-activity-7321910944924598272-8Obr)
- [Permuted Monte Carlo Validation](https://medium.com/@NFS303/validating-trading-strategies-with-permuted-monte-carlo-a-practical-guide-using-roc-momentum-d083180d4213)
- [SSRN — Backtest Overfitting in ML Era](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4686376_code4361537.pdf?abstractid=4686376&mirid=1)
- [BacktestIndia — Factor Testing on NSE](https://backtestindia.com/blog/backtesting-tool-india-nse-factor-investing)
- [SSRN — Gap Risk & Slippage Modeling](https://papers.ssrn.com/sol3/Delivery.cfm/5278107.pdf?abstractid=5278107&mirid=1)
- [Arxiv — Swing Trading with STL Decomposition](https://arxiv.org/html/2401.06139v1)
- [Fintrens — Why Backtests Fail](https://blogs.fintrens.com/your-backtest-said-4-2-lakh-profit-the-market-took-80-000-heres-exactly-why/)
- [Bigul — Slippage in Indian Algo Trading](https://bigul.co/blog/algo-trading/navigating-slippage-in-algo-trading-your-guide-to-smoother-execution-in-india)
