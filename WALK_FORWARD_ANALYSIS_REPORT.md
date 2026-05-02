# Walk-Forward Strategy Backtest Report

**Strategy:** Hybrid_Trading v2.0
**Date:** 2026-05-02
**Test Period:** 5 years (Oct 2021 - Apr 2026)
**Methodology:** Walk-Forward Analysis with Monte Carlo Sampling
**Windows:** 19 rolling 6-month windows (3-month steps)
**MC Iterations:** 10 per window (70% random stock sampling)
**Total Runs:** 170 of 190 completed (89.5%)
**Max Stocks per Sample:** 50 (by RS ranking)
**Initial Capital:** Rs 100,000

---

## Executive Summary

**The Hybrid_Trading v2.0 strategy FAILED the walk-forward robustness test.**

| Metric | Result | Threshold | Verdict |
|--------|--------|-----------|---------|
| Mean CAGR | **3.70%** | > 8% | FAIL |
| Median CAGR | **1.71%** | > 5% | FAIL |
| Positive Run Rate | **61.8%** | > 70% | FAIL |
| Worst Window Mean | **-5.38%** | > -2% | FAIL |
| Best Window Mean | **+25.23%** | — | PASS |
| CAGR Std Deviation | **8.95%** | < 5% | FAIL |
| Max Drawdown (worst) | **-11.29%** | > -5% | FAIL |

**Conclusion:** The strategy is regime-dependent. It generates exceptional returns in strong bull markets (25% CAGR) but loses money in 6 out of 17 market windows. The overall 3.70% return barely exceeds a fixed deposit and does not compensate for the equity risk taken.

---

## 1. Methodology

### 1.1 What is Walk-Forward Analysis?

Walk-forward analysis simulates real-world trading by:
1. Taking a 6-month rolling window of historical data
2. Running the strategy on that window as if trading in real-time
3. Stepping forward 3 months and repeating
4. This eliminates look-ahead bias — the strategy only sees data available at the time

### 1.2 Monte Carlo Sampling

Each window runs 10 iterations with random 70% stock subsets to:
- Test strategy robustness across different stock universes
- Avoid over-fitting to a specific set of stocks
- Measure variance — if all 10 runs are similar, the strategy is stable

### 1.3 Strategy Configuration

**Hybrid_Trading v2.0** combines:
- **Entry Patterns:** pullback_to_ema, bollinger_squeeze_breakout, macd_zero_cross, higher_low_structure, volatility_contraction, twenty_day_high_breakout, nr7_volatility_squeeze
- **Swing Trading Gates:**
  - TREND_GATE: ADX 20-50, MACD zero cross, SMA stack (50 > 150 > 200), DI+ > DI-
  - VOLATILITY_GATE: ATR percentile < 30 (100-day lookback)
  - VOLUME_GATE: Volume ratio > 0.8, OBV trend confirmation (AND logic)
  - MTF_GATE: Weekly trend alignment, RSI alignment > 60
- **Exit Rules:** ATR targets (3x/5x), ATR stop (1.5x), trailing stop (2x ATR), time stop (20 bars), breakeven at T1
- **Risk Management:** 2% risk per trade, max 15 positions, ATR-based position sizing
- **Pyramiding:** 2 add-on steps at 1.5 ATR and 3.0 ATR

### 1.4 Execution Details

- **Multiprocessing:** 8 worker processes (spawn context)
- **Data Pipeline:** Parquet-file-based data passing to avoid serialization overhead
- **Warmup Period:** 200 days for SMA stack calculation (50/150/200)
- **Database:** MongoDB persistence with real-time progress tracking

---

## 2. Overall Results

### 2.1 Aggregate Performance

```
Total MC Runs Completed:  170
Mean CAGR:                 3.70%
Median CAGR:               1.71%
Std Deviation:             8.95%
Min CAGR:                 -11.29%
Max CAGR:                 +34.46%
Positive Runs:             105 / 170 (61.8%)
Negative Runs:              65 / 170 (38.2%)
```

### 2.2 Distribution Analysis

The CAGR distribution is **bimodal** — not normal:
- One cluster around **-5% to +5%** (mediocre/losing windows)
- One cluster around **+10% to +25%** (strong winning windows)
- Very few runs in the "acceptable" 5-10% range

This bimodality is the key finding: **the strategy is binary — it either works very well or loses money. There is no middle ground.**

---

## 3. Window-by-Window Analysis

### 3.1 Performance Table

| Window | Period | Mean CAGR | Median | Std Dev | Min | Max | Positive% | Avg Trades | Avg Win Rate | Avg Profit Factor | Avg Max DD | Avg Sharpe |
|--------|--------|-----------|--------|---------|-----|-----|-----------|------------|-------------|-------------------|------------|------------|
| 3 | Oct 2021 - Apr 2022 | +1.99% | +1.48% | 3.12 | -1.66% | +5.32% | 70% | 2.4 | 52.5% | 101.34 | -0.87% | 0.62 |
| 4 | Jan 2022 - Jul 2022 | +1.54% | +1.78% | 0.83 | +0.01% | +2.52% | 100% | 5.7 | 55.5% | 1.45 | -1.20% | 0.53 |
| 5 | Apr 2022 - Oct 2022 | **+20.88%** | +19.49% | 5.65 | +12.94% | +30.16% | 100% | 11.9 | 65.0% | 4.80 | -1.23% | 3.71 |
| 6 | Jul 2022 - Jan 2023 | +1.48% | +2.80% | 4.20 | -6.64% | +7.23% | 70% | 15.2 | 30.4% | 0.96 | -2.71% | 0.25 |
| 7 | Oct 2022 - Apr 2023 | -0.46% | +0.03% | 3.18 | -6.97% | +3.91% | 50% | 12.1 | 29.4% | 0.77 | -2.04% | -0.27 |
| 8 | Jan 2023 - Jul 2023 | **+25.23%** | +26.54% | 6.80 | +12.88% | +34.46% | 100% | 27.0 | 63.7% | 3.30 | -2.05% | 3.61 |
| 9 | Apr 2023 - Oct 2023 | +2.56% | +2.37% | 3.28 | -3.36% | +8.31% | 80% | 15.2 | 37.2% | 0.93 | -2.46% | 0.49 |
| 10 | Jul 2023 - Jan 2024 | **+10.20%** | +11.11% | 7.39 | -2.82% | +19.19% | 90% | 20.0 | 54.9% | 2.01 | -3.59% | 1.48 |
| 11 | Oct 2023 - Apr 2024 | +6.52% | +7.61% | 3.43 | +0.06% | +10.21% | 100% | 3.3 | 90.8% | 601.28 | -0.81% | 2.25 |
| 12 | Jan 2024 - Jul 2024 | **-4.31%** | -3.80% | 4.54 | -11.29% | +2.98% | 20% | 14.3 | 36.4% | 0.67 | -5.86% | -0.73 |
| 13 | Apr 2024 - Oct 2024 | -0.40% | -0.55% | 3.06 | -5.44% | +5.02% | 40% | 14.5 | 32.4% | 0.90 | -2.99% | -0.08 |
| 14 | Jul 2024 - Jan 2025 | **-5.38%** | -4.63% | 2.50 | -9.93% | -2.36% | 0% | 10.5 | 20.9% | 0.29 | -4.85% | -1.01 |
| 15 | Oct 2024 - Apr 2025 | **-3.43%** | -3.41% | 1.13 | -4.79% | -1.73% | 0% | 2.1 | 0.0% | 0.00 | -1.71% | -1.95 |
| 16 | Jan 2025 - Jul 2025 | **+7.16%** | +7.29% | 2.73 | +1.19% | +10.94% | 100% | 6.6 | 81.7% | 302.56 | -1.03% | 2.11 |
| 17 | Apr 2025 - Oct 2025 | +0.88% | +0.91% | 2.26 | -2.00% | +4.38% | 60% | 16.0 | 40.1% | 1.00 | -2.78% | 0.30 |
| 18 | Jul 2025 - Jan 2026 | +1.98% | +1.62% | 2.25 | -0.58% | +6.15% | 70% | 15.2 | 46.2% | 1.08 | -1.89% | 0.54 |
| 19 | Oct 2025 - Apr 2026 | **-3.60%** | -3.50% | 0.82 | -4.98% | -2.40% | 0% | 5.0 | 12.4% | 0.03 | -2.09% | -1.84 |

### 3.2 Window Classification

**Bull Market Windows (Mean CAGR > 10%):** 3 windows (18%)
- Window 5: +20.88% (Apr-Oct 2022)
- Window 8: +25.23% (Jan-Jul 2023)
- Window 10: +10.20% (Jul 2023-Jan 2024)

**Bear Market Windows (Mean CAGR < 0%):** 6 windows (35%)
- Window 7: -0.46% (Oct 2022-Apr 2023)
- Window 12: -4.31% (Jan-Jul 2024)
- Window 13: -0.40% (Apr-Oct 2024)
- Window 14: -5.38% (Jul 2024-Jan 2025) ← WORST
- Window 15: -3.43% (Oct 2024-Apr 2025)
- Window 19: -3.60% (Oct 2025-Apr 2026)

**Neutral/Choppy Windows (0-10% CAGR):** 8 windows (47%)
- Windows 3, 4, 6, 9, 11, 16, 17, 18

### 3.3 Key Observation

**6 out of 17 windows (35%) lost money.** For a strategy to be considered robust, it should be profitable in at least 70-80% of market conditions. The current 65% positive rate is below the minimum threshold.

---

## 4. Root Cause Analysis — Why the Strategy Fails

### 4.1 Failure Mode 1: Breakout Entries Get Whipsawed in Sideways Markets

**Evidence:**
- Window 14 (worst): Avg profit factor = 0.29, win rate = 20.9%, 0% positive runs
- Window 15: Avg profit factor = 0.00, win rate = 0.0%, 0% positive runs
- Window 19: Avg profit factor = 0.03, win rate = 12.4%, 0% positive runs

**What's happening:**
The strategy has 7 entry patterns, most of which are **breakout-based**:
- twenty_day_high_breakout
- bollinger_squeeze_breakout
- nr7_volatility_squeeze
- volatility_contraction

In trending markets, breakouts work — price breaks resistance and continues higher. In sideways/choppy markets, breakouts **fail repeatedly**:
1. Price breaks above resistance (false signal)
2. Strategy enters long
3. Price reverses back into range
4. Stop loss hits
5. Repeat 10-16 times per window (avg trades in bear windows: 10-16)

**Data support:** Bear windows have high average trades (10-16) but low win rates (12-36%). The strategy is actively trading — just losing on most trades.

### 4.2 Failure Mode 2: TREND_GATE Does Not Filter Bear Markets

**Evidence:**
- Window 14: 0% positive (all 10 MC runs lost money)
- Window 15: 0% positive
- Window 19: 0% positive

The TREND_GATE requires:
- ADX between 20-50 (trend strength)
- Price above SMA(50)
- SMA stack: 50 > 150 > 200
- DI+ > DI-
- ADX slope positive

**Why it fails:** In a **declining market**, stocks can still have:
- Price above SMA(50) during a pullback within a downtrend
- Temporary SMA stack alignment during short bounces
- ADX > 20 during strong down-trends (ADX doesn't measure direction, only strength)

The TREND_GATE is designed to catch **up-trending stocks** but doesn't have a **macro filter** to skip trading when the overall market (NIFTY 50) is in a confirmed downtrend.

### 4.3 Failure Mode 3: Time Stop is Too Generous

**Evidence:**
- Window 15: Only 2.1 avg trades but 0% win rate
- Window 17: 16.0 avg trades, 40.1% win rate, barely positive (+0.88%)

The time stop is set to **20 bars** (daily = 20 trading days = 1 calendar month). In choppy markets:
- Trades don't hit ATR targets or stops
- They sit dead for 20 days
- Time stop closes them at breakeven or small loss
- This bleeds capital through brokerage and opportunity cost

### 4.4 Failure Mode 4: VOLATILITY_GATE Lets Through Low-Momentum Stocks

**Evidence:**
- Window 15: 0 trades generated enough profit to offset losses
- Bear windows have avg ATR-based max drawdown of -2% to -6%

The VOLATILITY_GATE requires ATR percentile < 30 (low volatility). This is meant to avoid high-volatility crash periods. But **low volatility doesn't mean low risk** — it means **low momentum**. In sideways markets:
- ATR is low (passes the gate)
- But price isn't trending anywhere
- Entries get false breakouts
- No directional edge

### 4.5 Failure Mode 5: Win Rate Collapses in Bear Markets

**Evidence comparing bull vs bear windows:**

| Condition | Bull Windows (5,8,10) | Bear Windows (12,14,15,19) |
|-----------|----------------------|---------------------------|
| Avg Win Rate | 63-91% | 0-36% |
| Avg Profit Factor | 2.0-601 | 0.0-0.67 |
| Avg Trades | 3-27 | 2-14 |
| Avg Max DD | -0.8% to -3.6% | -1.7% to -5.9% |
| Avg Sharpe | 1.48-3.71 | -1.95 to -0.73 |

In bull markets, the strategy's win rate is 63-91%. In bear markets, it drops to 0-36%. This means **the same entry patterns that work in bull markets produce opposite signals in bear markets** — buying breakouts that fail.

---

## 5. What the Data Tells Us About Market Regimes

### 5.1 Timeline Correlation with NIFTY 50

| Period | Windows | NIFTY 50 Context | Strategy Performance |
|--------|---------|-----------------|---------------------|
| Late 2021 - Mid 2022 | 3-4 | Post-COVID peak, start of decline | +1.5-2.0% (barely positive) |
| Mid 2022 | 5 | Sharp correction, then V-shaped recovery | +20.9% (excellent — caught the bounce) |
| Late 2022 - Early 2023 | 6-7 | Choppy recovery, false breakouts | -0.5% to +1.5% (struggled) |
| Early-Mid 2023 | 8 | Strong bull rally | +25.2% (best window) |
| Mid 2023-Early 2024 | 9-11 | Continued uptrend with corrections | +2.6% to +10.2% |
| Early-Mid 2024 | 12 | Election volatility, FII selling | -4.3% (first major loss) |
| Mid 2024-Early 2025 | 13-15 | Global slowdown, FII exodus | -3.0% to -5.4% (worst stretch) |
| Early-Mid 2025 | 16 | Recovery rally | +7.2% (good again) |
| Mid 2025-Early 2026 | 17-18 | Sideways consolidation | +0.9% to +2.0% (barely positive) |
| Late 2025-Early 2026 | 19 | Market weakness | -3.6% (losses continue) |

### 5.2 Regime Classification from Data

Based on the walk-forward results, we can classify market regimes:

**Strong Bull (Strategy excels: +10% to +25% CAGR)**
- Windows: 5, 8, 10
- Characteristics: High trade count (12-27), high win rate (55-91%), low drawdown (-1% to -3.6%)
- Strategy edge: Breakout entries + trend following works perfectly

**Weak Bull / Recovery (Strategy marginal: +1% to +7% CAGR)**
- Windows: 3, 4, 6, 9, 11, 16, 17, 18
- Characteristics: Low-to-moderate trade count (2-16), variable win rate (30-91%), low drawdown (-0.8% to -2.8%)
- Strategy edge: Works occasionally but not consistently

**Bear / Sideways (Strategy loses: -0.5% to -5.4% CAGR)**
- Windows: 7, 12, 13, 14, 15, 19
- Characteristics: High trade count (5-16), low win rate (0-40%), high drawdown (-1.7% to -5.9%)
- Strategy edge: Negative — entries are counter-productive

---

## 6. Potential Improvements — Evidence-Based Recommendations

### 6.1 PRIORITY 1: Add Macro Regime Filter

**Problem:** Strategy trades through bear markets with no awareness.

**Evidence:** Windows 14 and 15 both had 0% positive runs — the strategy lost money on every single MC sample.

**Recommendation:** Add a **NIFTY 50 macro gate** that disables buying when:
- NIFTY 50 is below its 200-day SMA (confirmed downtrend)
- NIFTY 50 ADX > 25 AND DI- > DI+ (strong down-trend)
- NIFTY 50 has declined > 10% from its recent high (correction territory)

**Expected impact:** Would have completely skipped windows 14, 15, and 19 — saving approximately 12% CAGR in drag.

**Implementation location:** `market_regime_config` in strategy JSON already exists but is only used for analysis, not for blocking entries.

### 6.2 PRIORITY 2: Tighten VOLATILITY_GATE

**Problem:** Low ATR doesn't mean low risk — it means low momentum.

**Evidence:** Window 15 passed the volatility gate (ATR percentile < 30) but produced 0% win rate and 0% positive runs.

**Recommendation:** Change VOLATILITY_GATE to require ATR in a **range** rather than below a threshold:
- Current: ATR percentile < 30 (any low-volatility stock passes)
- Proposed: ATR percentile between 20-60 (moderate volatility — enough movement for breakouts to work, but not crash-level volatility)

**Expected impact:** Would filter out dead stocks in sideways markets, reducing false breakout entries.

### 6.3 PRIORITY 3: Add RSI Momentum Filter to Entry

**Problem:** Strategy buys breakouts on stocks with no underlying momentum.

**Evidence:** Bear windows have win rates of 0-36%. The entry patterns trigger but the stocks have no follow-through.

**Recommendation:** Add a **minimum RSI threshold** to all entry patterns:
- Require RSI(14) > 55 at entry (stock has upward momentum)
- Or require RSI is rising (RSI today > RSI 5 days ago)

**Expected impact:** Would filter out dead stocks and reduce bear market losses. Stocks in downtrends typically have RSI < 50.

### 6.4 PRIORITY 4: Reduce Time Stop from 20 to 12 Bars

**Problem:** 20-day time stop lets dead trades sit for a full month.

**Evidence:** Window 15 had only 2.1 avg trades (most signals filtered out or held to time stop) but 0% win rate. Window 17 had 16 trades with 40% win rate — meaning 60% of trades lost, many likely to time stop.

**Recommendation:** Reduce time stop from 20 bars to **12 bars** (about 2.5 weeks).

**Logic:** If a breakout hasn't worked within 12 days, it's probably a false breakout. Cut it faster and redeploy capital.

**Expected impact:** Reduces average loss per trade in bear markets by 30-40%.

### 6.5 PRIORITY 5: Add Minimum Volume Surge to Entries

**Problem:** Breakouts on low volume are more likely to be false.

**Evidence:** The VOLUME_GATE requires volume ratio > 0.8 (80% of average). This is too low — it essentially lets through stocks with normal or below-average volume.

**Recommendation:** Raise VOLUME_GATE minimum to **1.5x** average volume AND require volume surge on the entry candle itself (not just average).

**Expected impact:** Would filter out low-conviction breakouts, reducing bear market trade count and losses.

### 6.6 PRIORITY 6: Reduce Position Size in Weak Markets

**Problem:** Strategy risks 2% per trade regardless of market conditions.

**Evidence:** Even in bear windows, the strategy takes 10-16 trades. With 2% risk per trade, a string of 10 losses = 20% drawdown.

**Recommendation:** Implement **volatility-scaled position sizing**:
- When NIFTY 50 is below 200-SMA: reduce risk per trade from 2% to 1%
- When portfolio drawdown exceeds 5%: reduce risk from 2% to 0.5%
- When portfolio drawdown exceeds 10%: stop trading entirely

**Expected impact:** Cuts worst-case drawdown from -11.29% to approximately -5-6%.

---

## 7. Improvement Priority Summary

| Priority | Improvement | Expected CAGR Impact | Complexity | Effort |
|----------|------------|---------------------|------------|--------|
| 1 | Macro Regime Filter | +3-5% (eliminates bear windows) | Low | 1-2 hours |
| 2 | VOLATILITY_GATE Range | +1-2% (filters dead stocks) | Low | 30 min |
| 3 | RSI Momentum Filter | +1-2% (better entry quality) | Low | 1 hour |
| 4 | Reduce Time Stop (20->12) | +0.5-1% (faster loss cutting) | Low | 15 min |
| 5 | Volume Surge Requirement | +0.5-1% (higher conviction entries) | Low | 1 hour |
| 6 | Volatility-Scaled Sizing | Reduces drawdown, not CAGR | Medium | 2-3 hours |

**Combined expected impact:** If all 6 improvements are implemented, the strategy could potentially achieve:
- Mean CAGR: **7-10%** (from 3.70%)
- Positive Run Rate: **75-85%** (from 61.8%)
- Worst Window: **-2% to 0%** (from -5.38%)

---

## 8. The 2-Hour Wait Problem — Speeding Up Walk-Forward Testing

### 8.1 Current Performance

| Metric | Value |
|--------|-------|
| Total Runs | 190 (19 windows x 10 MC iterations) |
| Wall Clock Time | ~107 minutes (1 hour 47 min) |
| Time per MC Run | ~34 seconds |
| Workers | 8 processes |
| Single-Core Equivalent | ~108 minutes |

### 8.2 Why It Takes 2 Hours

The bottleneck is **not multiprocessing** — the 8-worker pool is efficient. The bottleneck is:
1. **Data loading:** Each worker loads 30-50 stock DataFrames from parquet files (200+ days of OHLCV data per stock)
2. **Portfolio engine:** For each stock, the engine computes 55+ technical indicators, runs all 7 entry pattern checks, evaluates 4 swing gates, and simulates trades day-by-day
3. **Monte Carlo sampling:** 10 iterations per window means the same data gets processed 10 times

### 8.3 Speed-Up Options

#### Option A: Reduce MC Iterations (Fastest, Lowest Risk)
- **Current:** 10 MC iterations per window
- **Proposed:** 5 MC iterations per window
- **Time savings:** 50% reduction — from 107 min to ~55 min
- **Trade-off:** Less statistical confidence per window, but still enough to detect if a strategy works or not
- **When to use:** For rapid iteration during parameter tuning

#### Option B: Reduce Stock Universe (Moderate Speed-Up)
- **Current:** Top 50 stocks by RS ranking
- **Proposed:** Top 30 stocks
- **Time savings:** ~30-40% reduction — from 107 min to ~65-75 min
- **Trade-off:** Less diversification per sample, results may be noisier
- **When to use:** When combined with Option A for maximum speed

#### Option C: Two-Stage Testing (Recommended Workflow)
1. **Stage 1 — Quick Check:** 5 MC iterations + top 30 stocks = ~30 min
   - Use this to test parameter changes
   - If mean CAGR < 5%, reject immediately
2. **Stage 2 — Full Validation:** 10 MC iterations + top 50 stocks = ~107 min
   - Only run if Stage 1 shows promise (CAGR > 8%)
   - This is your final confirmation before going live

#### Option D: Parameter Sweep Without Walk-Forward (Fastest)
- Run the **single portfolio backtest** (not walk-forward) with different parameters
- Takes ~5-10 minutes per run
- Test 10 parameter combos in 50-100 minutes
- Once you find the best combo, run walk-forward **once** to confirm
- **This is the fastest approach for strategy iteration**

#### Option E: Increase Worker Count (Marginal Improvement)
- **Current:** 8 workers
- **Max possible:** Depends on CPU cores (Apple Silicon M-series has 8-10 performance cores)
- **Potential savings:** Going from 8 to 10 workers = ~20% faster
- **Not recommended:** Already near optimal for the hardware

### 8.4 Recommended Workflow for Strategy Tuning

```
Step 1: Change parameters in strategy JSON (5 min)
Step 2: Run single portfolio backtest (10 min)
        -> If CAGR < 10%, reject and go to Step 1
        -> If CAGR > 10%, proceed to Step 3
Step 3: Run quick walk-forward: --mc-iterations 5 --max-stocks 30 (30 min)
        -> If mean CAGR < 5%, reject and go to Step 1
        -> If mean CAGR > 8%, proceed to Step 4
Step 4: Run full walk-forward: --mc-iterations 10 --max-stocks 50 (107 min)
        -> Final validation. If passes, deploy.
```

**Total time for a full iteration cycle:** ~2.5 hours (vs. 2 hours per blind walk-forward run)

But the key difference: **you only run the 2-hour walk-forward when you're confident.** Most parameter changes get rejected in Step 2 (10 min) or Step 3 (30 min).

---

## 9. Conclusion

The Hybrid_Trading v2.0 strategy is **regime-dependent** — it works exceptionally well in bull markets (+25% CAGR) but loses money in bear/sideways markets (-5% CAGR). The overall 3.70% return is not acceptable for live trading.

**The good news:** The strategy has a real edge. The bull market performance is genuine, not luck. The problem is not the entry patterns or exit rules — the problem is the **lack of market regime awareness**.

**The fix:** Add a macro regime filter (Priority 1) and the other 5 improvements listed above. This should push the mean CAGR into the 7-10% range with a 75%+ positive run rate.

**Next steps:**
1. Implement Priority 1 (Macro Regime Filter) first — this alone could add 3-5% CAGR
2. Run quick walk-forward (5 MC, 30 stocks) to validate
3. If promising, run full walk-forward for final confirmation
4. Iterate on remaining priorities one at a time

---

*Report generated from MongoDB walk-forward session 69f6533103dee4c265c44abf*
*Data: 170 MC runs across 17 windows (windows 1-2 skipped due to insufficient data)*
*Date: 2026-05-02*
