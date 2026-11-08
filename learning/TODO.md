# Quantitative Trading & Recommendation Engine: Complete Algorithmic Master Checklist

> **Target Audience**: Quantitative Traders, Financial Engineers, and Algorithmic System Architects.
> **Scope**: The algorithmic, mathematical, statistical, and financial logic powering **Smart Advice**.
> **Rule**: This document does **not** teach general programming syntax (Python, JavaScript, SQL). Instead, it teaches the **exact quantitative algorithms, microstructural signals, risk models, statistical validation pipelines, and execution mechanics** our platform uses to generate institutional-grade swing trading recommendations.

---

## 📌 Executive Summary: What Smart Advice Actually Does

**Smart Advice** is an automated, institutional-grade quantitative swing trading and recommendation platform specifically engineered for the **National Stock Exchange of India (NSE)**. 

### The Core Problem It Solves
Retail traders frequently suffer from three terminal flaws:
1. **Selection Bias & Emotional Chasing**: Buying stocks after extended parabolic runs without structural support or institutional footprint.
2. **Asymmetric Risk-Reward**: Risking 5–10% to make 2–3%, resulting in a negative expectancy that destroys capital during market drawdowns.
3. **Execution Indiscipline & Overfitting**: Trading against the prevailing macro market regime, failing to honor stops, and relying on backtested strategies that suffer from lookahead bias and p-hacking.

### How Smart Advice Solves This Algorithmic Pipeline
Our platform replaces human emotion with a strictly serialized, multi-pillar quantitative recommendation pipeline:
```
[ NSE Universe (2000+ Tickers) ]
               │
               ▼
   [ 1. Hard Liquidity & Microstructure Filter ] (Volume > 150k, MCap > ₹1000 Cr, Price ₹50-₹8000)
               │
               ▼
   [ 2. Macro Regime & Market Breadth Engine ] (Nifty 50 > 200 SMA, Universe Breadth > 35%)
               │
               ▼
   [ 3. Three-Gate Structural Qualifier ] (Trend Gate: ADX/DI/SMA Stack, Volatility Gate, Volume Gate)
               │
               ▼
   [ 4. Algorithmic Entry Pattern Engine ] (EMA Pullbacks, VCP Breakouts, Bollinger Squeezes, NR7)
               │
               ▼
   [ 5. Multi-Pillar Scoring Engine ] (65% Tech + 20% Fundamental + 15% Sector/Options OI)
               │
               ▼
   [ 6. Hard Floor & Risk-Reward Gate ] (Tech >= 0.35, Fund >= 0.10, Total >= 0.60, RR >= 1.2)
               │
               ▼
   [ 7. Volatility-Scaled Dynamic Position Sizer ] (ATR-based Stop, Capital Cap, Risk-Per-Trade)
               │
               ▼
   [ 8. Multi-Stage Staged Exit & Pyramiding Engine ] (T1 @ 3.5x ATR, T2 @ 7.0x ATR, ATR Trails)
               │
               ▼
   [ 9. 6-Phase Ultimate Backtest Validation ] (DSR, Monte Carlo Permutation, Walk-Forward, Stress Tests)
```

Completing this checklist guarantees complete operational and algorithmic mastery of every formula, filter, signal, and quantitative gate inside the Smart Advice ecosystem.

---

## 🧭 Master Roadmap & Progress Tracker

- [ ] **Module 1**: NSE Market Universe, Microstructure & Data Hygiene
- [ ] **Module 2**: Macro Regime Detection & Universe Breadth Mechanics
- [ ] **Module 3**: Quantitative Trend Architecture & Moving Average Stacks
- [ ] **Module 4**: Volatility Dynamics, Contraction & Percentile Gating
- [ ] **Module 5**: Volume Microstructure, OBV & Institutional Footprints
- [ ] **Module 6**: Momentum Oscillators & Normalized Range Filtering
- [ ] **Module 7**: Algorithmic Entry Pattern Recognition Mechanics
- [ ] **Module 8**: NSE Option Chain Open Interest (OI) & Derivatives Sentiment
- [ ] **Module 9**: Fundamental Health Scoring & Balance Sheet Solvency
- [ ] **Module 10**: Multi-Pillar Scoring Synthesis & Hard Floor Architecture
- [ ] **Module 11**: Dynamic ATR Risk Management & Volatility-Scaled Position Sizing
- [ ] **Module 12**: Multi-Target Profit Taking, Trailing Stops & Time Stops
- [ ] **Module 13**: Systematic Pyramiding & Capital Compounding Rules
- [ ] **Module 14**: Portfolio Construction, Concurrency & Circuit Breakers
- [ ] **Module 15**: 6-Phase Ultimate Backtest & Statistical Validation Pipeline
- [ ] **Module 16**: Live/Paper Execution State Machine, Replay Engine & Settlement
- [ ] **Module 17**: Intraday Quantitative Engine & VWAP Microstructure
- [ ] **Module 18**: High-Performance Vectorized Backtest & IndicatorStore Engine
- [ ] **Module 19**: Machine Learning Models & Statistical Regime Switching
- [ ] **Module 20**: Quantitative Biases, Overfitting Traps & Edge Preservation
- [ ] **Module 21**: Comprehensive Multi-Scenario Numerical Walkthroughs

---

## Module 1: NSE Market Universe, Microstructure & Data Hygiene

The quality of any quantitative model is bounded by the quality of its input universe. Smart Advice never analyzes uncurated stock tickers.

- [ ] **1.1 Understanding NSE Equities Microstructure**
  - **Description**: The National Stock Exchange of India (NSE) trades with distinct continuous auction rules (09:15 to 15:30 IST), pre-open call auctions (09:00 to 09:08 IST), and closing price calculations based on the volume-weighted average price (VWAP) of the last 30 minutes (15:00 to 15:30 IST).
  - **Algorithm**: Reject any price feed not reconciled with official NSE cash-market adjusted close data. For swing trading, all algorithmic signals are evaluated on continuous daily close bars or pre-close 15-minute intervals.
  - **Example**: If a stock trades at intraday high ₹520 at 15:28 but the 30-minute VWAP calculates at ₹512.40, the algorithmic daily close is logged as ₹512.40. All indicators (RSI, ATR, SMA) update against this settled value.

- [ ] **1.2 Hard Liquidity & Capitalization Filter**
  - **Description**: Illiquid equities suffer from wide bid-ask spreads, market impact cost, and inability to execute stop-losses.
  - **Algorithm**:
    $$\text{Average Daily Volume (20-day)} = \frac{1}{20}\sum_{t=0}^{19} \text{Volume}_t \ge 150,000 \text{ shares}$$
    $$\text{Market Capitalization} \ge ₹1,000 \text{ Crores}$$
    $$₹50.0 \le \text{Current Price} \le ₹8,000.0$$
  - **Example**: Stock ABC has price ₹25.0 and daily volume 2,000,000 shares. Even though volume is high, it is rejected by the lower price boundary (< ₹50) to prevent penny-stock manipulation. Stock XYZ with price ₹450, volume 85,000 is rejected because volume < 150,000 shares.

- [ ] **1.3 Corporate Actions & Adjusted Price Series**
  - **Description**: Stock splits, bonus shares, and dividends introduce artificial price gaps that corrupt technical indicators if unadjusted.
  - **Algorithm**:
    $$\text{Adjustment Factor } k = \frac{\text{Post-split shares}}{\text{Pre-split shares}}$$
    $$\text{Adjusted Close}_t = \text{Raw Close}_t \times \frac{1}{k}, \quad \text{Adjusted Volume}_t = \text{Raw Volume}_t \times k$$
  - **Example**: A 1:1 bonus issue cuts stock price from ₹1000 to ₹500. Without backward-adjustment, an algorithm calculates a spurious -50% crash, triggering invalid stop-loss and short signals. The data pipeline must apply split-adjusted series across the entire 10-year lookback.

- [ ] **1.4 Lookahead Bias Elimination in Replay & Backtest**
  - **Description**: An algorithm must never "know" tomorrow's open, high, low, or close when making a trade decision on bar $T$.
  - **Algorithm**: Given simulation date $T$, restrict the historical matrix strictly to:
    $$\mathcal{D}_T = \{ (\text{Open}_t, \text{High}_t, \text{Low}_t, \text{Close}_t, \text{Volume}_t) \mid t \le T \}$$
    Any rolling indicator calculation must use only indices $\le T$.
  - **Example**: In vectorbt pre-computation, slice the IndicatorStore at $T$ without including the bar $T+1$ candle.

---

## Module 2: Macro Regime Detection & Market Breadth Filtering

No technical strategy can consistently profit if it goes long while the entire macro market is collapsing. Smart Advice employs a two-tier macro gate before evaluating individual stocks.

- [ ] **2.1 Nifty 50 Macro Regime Classifier**
  - **Description**: The benchmark index `^NSEI` (Nifty 50) represents India's top 50 blue chips and dictates systemic market beta.
  - **Algorithm**:
    $$\text{Regime} = \begin{cases} \text{BULL}, & \text{if } \text{Close}_{\text{Nifty}} > \text{SMA}_{200}(\text{Nifty}) \\ \text{BEAR}, & \text{if } \text{Close}_{\text{Nifty}} \le \text{SMA}_{200}(\text{Nifty}) \end{cases}$$
    If `pause_buying_if_bearish = True` and $\text{Regime} == \text{BEAR}$, the orchestrator halts all new buy recommendations across all strategies.
  - **Example**: Nifty 50 trades at 21,800. The 200-day simple moving average is 22,150. Since $21,800 < 22,150$, the market regime is labeled **BEARISH**. The engine suspends new stock entries, preserving cash capital and avoiding false breakouts.

- [ ] **2.2 Market Breadth Filter (Advance/Decline vs Moving Average)**
  - **Description**: A market index can be artificially buoyed by 3 heavyweights (e.g., Reliance, HDFC Bank, TCS) while 80% of mid- and small-caps are in severe downtrends.
  - **Algorithm**:
    $$\text{Universe Breadth } \mathcal{B} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{Close}_{i} > \text{SMA}_{20}(\text{Close}_i))}{N} \times 100\%$$
    $$\text{Filter Condition}: \mathcal{B} \ge 35\%$$
    If $\mathcal{B} < 35\%$, block all new long swing entries unless the stock qualifies for the **Relative Strength Leader Exception**.
  - **Example**: Out of 500 scanned stocks, only 140 are trading above their 20-day SMA ($\mathcal{B} = 28\%$). System enters Market Breadth Weakness mode. All standard swing breakout candidates are blocked.

- [ ] **2.3 Relative Strength (RS) Leader Exception Under Weak Breadth**
  - **Description**: Genuine market leaders often break out 2 to 4 weeks *before* the broad market turns. The system must not miss generational leaders during early bottoming phases.
  - **Algorithm**:
    $$\text{Distance from 52-Week High} = \frac{\text{High}_{52\text{w}} - \text{Close}}{\text{High}_{52\text{w}}} \times 100\%$$
    $$\text{Leader Qualified} \iff (\text{Distance} \le 15.0\%) \land (\text{Trend Gate} == \text{PASS})$$
    If qualified as an RS Leader, bypass the market breadth block.
  - **Example**: Market breadth is weak ($\mathcal{B} = 28\%$). Stock TRENT is trading at ₹7,100 while its 52-week high is ₹7,250 (distance = 2.07%). Its Trend Gate has passed. TRENT is approved for entry as a certified Relative Strength Leader.

- [ ] **2.4 Multi-Regime Adaptive Risk Multiplier**
  - **Description**: Rather than a binary on/off switch, dynamic systems scale exposure depending on regime state.
  - **Algorithm**:
    $$\text{Max Positions Allowed} = \begin{cases} 6, & \text{Bull Regime} \\ 4, & \text{Sideways / Neutral Regime} \\ 0, & \text{Bear Regime} \end{cases}$$
    $$\text{Risk Per Trade (\%)} = \begin{cases} 1.5\%, & \text{Bull Regime} \\ 1.0\%, & \text{Sideways Regime} \\ 0.0\%, & \text{Bear Regime} \end{cases}$$
  - **Example**: Portfolio has ₹1,000,000 equity. In Bull market, risk per trade is ₹15,000 across up to 6 positions. When market slips into sideways consolidation, risk drops to ₹10,000 per trade across maximum 4 positions.

---

## Module 3: Quantitative Trend Architecture & Moving Average Stacks

A swing trading strategy requires strong momentum in the direction of the primary and secondary trends.

- [ ] **3.1 The 3-SMA Trend Stack Gating**
  - **Description**: Institutional accumulation creates smooth, ascending moving average fans without crossovers into chop.
  - **Algorithm**:
    $$\text{SMA}_k(t) = \frac{1}{k}\sum_{i=0}^{k-1} \text{Close}_{t-i}$$
    $$\text{Trend Stack Condition}: \text{Close}_t > \text{SMA}_{50}(t) > \text{SMA}_{150}(t) > \text{SMA}_{200}(t)$$
  - **Example**: Stock BEL has $\text{Close} = ₹310$, $\text{SMA}_{50} = ₹295$, $\text{SMA}_{150} = ₹270$, $\text{SMA}_{200} = ₹250$. Since $310 > 295 > 270 > 250$, the institutional trend stack condition is satisfied.

- [ ] **3.2 Moving Average Slope Quantification**
  - **Description**: A moving average that is sloping downward or flat signifies consolidation or distribution, even if price briefly crosses above it.
  - **Algorithm**:
    $$\text{Slope}(\text{SMA}_{50}) = \frac{\text{SMA}_{50}(t) - \text{SMA}_{50}(t-10)}{10}$$
    $$\text{Requirement}: \text{Slope}(\text{SMA}_{50}) > 0$$
  - **Example**: If $\text{SMA}_{50}$ was ₹290 ten bars ago and is ₹295 today, slope $= +0.50$ (positive, valid). If price is ₹300 but $\text{SMA}_{50}$ drifted from ₹305 to ₹302, slope is negative; the trend test fails.

- [ ] **3.3 Directional Movement Index (DMI) & ADX Quantification**
  - **Description**: Welles Wilder's Average Directional Index (ADX) measures trend strength regardless of direction, while $+DI$ and $-DI$ measure directional dominance.
  - **Algorithm**:
    $$+DM_t = \max(\text{High}_t - \text{High}_{t-1}, 0) \quad \text{if } \text{High}_t - \text{High}_{t-1} > \text{Low}_{t-1} - \text{Low}_t \text{ else } 0$$
    $$-DM_t = \max(\text{Low}_{t-1} - \text{Low}_t, 0) \quad \text{if } \text{Low}_{t-1} - \text{Low}_t > \text{High}_t - \text{High}_{t-1} \text{ else } 0$$
    $$\text{True Range (TR)} = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)$$
    $$+DI_{14} = 100 \times \frac{\text{EMA}_{14}(+DM)}{\text{EMA}_{14}(TR)}, \quad -DI_{14} = 100 \times \frac{\text{EMA}_{14}(-DM)}{\text{EMA}_{14}(TR)}$$
    $$DX = 100 \times \frac{|+DI_{14} - -DI_{14}|}{+DI_{14} + -DI_{14}}, \quad ADX_{14} = \text{EMA}_{14}(DX)$$
    $$\text{Trend Gate Passed} \iff (ADX_{14} \ge 20) \land (+DI_{14} > -DI_{14})$$
  - **Example**: Stock HAL shows $+DI_{14} = 28.4$, $-DI_{14} = 13.1$, and $ADX_{14} = 24.6$. Because $+DI > -DI$ and $ADX \ge 20$, the trend strength and bullish alignment checks pass.

- [ ] **3.4 Price Distance Boundary (Anti-Extension Filter)**
  - **Description**: Entering a stock that is extended 15% above its 20-day or 50-day moving average invites mean-reversion pullbacks that trigger stop losses.
  - **Algorithm**:
    $$\text{DistPct}(\text{SMA}_{20}) = \frac{|\text{Close}_t - \text{SMA}_{20}(t)|}{\text{SMA}_{20}(t)} \times 100\%$$
    $$\text{Gating Condition}: \text{DistPct}(\text{SMA}_{20}) \le 5.0\%$$
  - **Example**: Stock Dixon is ₹12,000 while its 20-day SMA is ₹11,000. $\text{DistPct} = (12000 - 11000) / 11000 = 9.09\%$. The stock is statistically overextended. The system blocks the entry until a pullback or base contraction occurs.

- [ ] **3.5 52-Week High Proximity Filter**
  - **Description**: Stocks making 52-week highs have zero overhead resistance (trapped bagholders waiting to sell at breakeven). Stocks > 25% below their 52-week high are structurally burdened by supply.
  - **Algorithm**:
    $$\text{Proximity} = \frac{\max_{t \in [T-252, T]}(\text{High}_t) - \text{Close}_T}{\max_{t \in [T-252, T]}(\text{High}_t)} \times 100\%$$
    $$\text{Requirement}: \text{Proximity} \le 25.0\%$$
  - **Example**: Stock INFY has 52-week high ₹1950 and current price ₹1400. Proximity $= (1950 - 1400)/1950 = 28.2\% > 25\%$. Filter fails. Stock LT has 52-week high ₹3800 and current price ₹3650. Proximity $= 3.94\% \le 25\%$. Filter passes.

---

## Module 4: Volatility Dynamics, Contraction & Percentile Gating

Volatility is cyclical: periods of contraction (low volatility) lead to explosive expansions (trends). Smart Advice buys during the final stage of contraction, not after volatility has peaked.

- [ ] **4.1 Average True Range (ATR) Mathematical Formulation**
  - **Description**: Measures price range volatility in absolute currency units without directional bias.
  - **Algorithm**:
    $$TR_t = \max(\text{High}_t - \text{Low}_t, |\text{High}_t - \text{Close}_{t-1}|, |\text{Low}_t - \text{Close}_{t-1}|)$$
    $$ATR_{14}(t) = \frac{ATR_{14}(t-1) \times 13 + TR_t}{14}$$
  - **Example**: Stock Tata Motors has today's High = ₹980, Low = ₹962, yesterday's Close = ₹965. $TR = \max(980 - 962, |980 - 965|, |962 - 965|) = \max(18, 15, 3) = 18$. The rolling ATR updates with this 18 True Range input.

- [ ] **4.2 ATR Historical Percentile Ranking (The Volatility Gate)**
  - **Description**: Prevents buying when daily swings are abnormally chaotic (wide ATR) or when the stock has become completely dormant (dead ATR).
  - **Algorithm**: Over a lookback window $L = 100$ days:
    $$\mathcal{H}_{\text{ATR}} = \{ ATR_{14}(t-i) \mid i \in [0, 99] \}$$
    $$\text{Percentile}(ATR_t) = \frac{\sum_{x \in \mathcal{H}_{\text{ATR}}} \mathbb{I}(x < ATR_t)}{100} \times 100\%$$
    $$\text{Volatility Gate Passed} \iff 5\% \le \text{Percentile}(ATR_t) \le 90\%$$
  - **Example**: If today's ATR is ₹45 and across the last 100 days only 4 days had a higher ATR ($\text{Percentile} = 96\%$), the stock is in a manic expansion phase. Volatility Gate = **FAIL**. If $\text{Percentile} = 42\%$, Volatility Gate = **PASS**.

- [ ] **4.3 Bollinger Band Width (BBW) Squeeze Gating**
  - **Description**: Bollinger Bands measure standard deviations around a 20-period moving average. When the bands contract tightly, volatility compression has occurred.
  - **Algorithm**:
    $$\mu_{20} = \text{SMA}_{20}(\text{Close}), \quad \sigma_{20} = \sqrt{\frac{1}{20}\sum_{i=0}^{19}(\text{Close}_{t-i} - \mu_{20})^2}$$
    $$\text{Upper Band} = \mu_{20} + 2\sigma_{20}, \quad \text{Lower Band} = \mu_{20} - 2\sigma_{20}$$
    $$\text{Bandwidth} = \frac{\text{Upper Band} - \text{Lower Band}}{\mu_{20}}$$
    $$\text{Squeeze Detected} \iff \text{Bandwidth} \le 0.10 \quad (\text{or } \le \text{Threshold specified in strategy config})$$
  - **Example**: Stock TCS trades at ₹4,000. Upper Band = ₹4,080, Lower Band = ₹3,920. Bandwidth $= (4080 - 3920)/4000 = 160 / 4000 = 0.04$ (4%). This indicates extreme compression. An expansion move is imminent.

- [ ] **4.4 NR7 Volatility Contraction Algorithm**
  - **Description**: Mark Raschke's Narrowest Range in 7 Days (NR7). Day $T$ has the smallest absolute trading range $(High - Low)$ compared to the preceding 6 sessions.
  - **Algorithm**:
    $$\text{Range}_t = \text{High}_t - \text{Low}_t$$
    $$\text{Is\_NR7} \iff \text{Range}_T < \min_{i \in [1, 6]} (\text{Range}_{T-i})$$
  - **Example**: Daily ranges over the last 7 sessions: Day 1: ₹14, Day 2: ₹12, Day 3: ₹15, Day 4: ₹11, Day 5: ₹13, Day 6: ₹10, Day 7 (Today): ₹7.5. Today's range ₹7.5 is strictly the smallest. `nr7_volatility_squeeze` activates.

---

## Module 5: Volume Microstructure, Institutional Footprints & Delivery Dynamics

Volume provides the microstructural proof of whether institutions (mutual funds, FIIs, DIIs) are accumulating or dumping shares.

- [ ] **5.1 20-Day Volume Moving Average Multiplier**
  - **Description**: A valid breakout must be backed by institutional buying volume significantly exceeding average retail noise.
  - **Algorithm**:
    $$\overline{\text{Vol}}_{20}(t) = \frac{1}{20}\sum_{i=0}^{19} \text{Volume}_{t-i}$$
    $$\text{Volume Gate Passed} \iff \text{Volume}_t \ge \overline{\text{Vol}}_{20}(t) \times \text{MinVolumeRatio}$$
    Where `min_volume_ratio` is typically 0.40 for pullback setups and $\ge 1.50$ for breakout triggers.
  - **Example**: Average 20-day volume is 500,000 shares. On a 20-day high breakout, today's volume is 920,000 shares. Volume multiplier $= 920,000 / 500,000 = 1.84 \ge 1.50$. Breakout volume confirmed.

- [ ] **5.2 On-Balance Volume (OBV) Trend Slope**
  - **Description**: Joseph Granville's OBV accumulates volume on up days and subtracts volume on down days to measure cumulative order flow.
  - **Algorithm**:
    $$OBV_t = \begin{cases} OBV_{t-1} + \text{Volume}_t, & \text{if } \text{Close}_t > \text{Close}_{t-1} \\ OBV_{t-1} - \text{Volume}_t, & \text{if } \text{Close}_t < \text{Close}_{t-1} \\ OBV_{t-1}, & \text{if } \text{Close}_t = \text{Close}_{t-1} \end{cases}$$
    Compute the linear regression slope of $OBV$ over the last 10 sessions:
    $$\text{Slope}(OBV_{10}) = \frac{10\sum_{i=1}^{10} i \cdot OBV_i - \sum_{i=1}^{10} i \sum_{i=1}^{10} OBV_i}{10\sum_{i=1}^{10} i^2 - (\sum_{i=1}^{10} i)^2} > 0$$
  - **Example**: Even if price moves sideways in a ₹10 band for 10 days, if up days have 800,000 volume and down days have 200,000 volume, OBV slope is steeply positive, revealing hidden institutional accumulation.

- [ ] **5.3 Pocket Pivot Volume Signature (Gil Morales & Chris Kacher)**
  - **Description**: A pocket pivot occurs when price breaks out of a 10-day or 50-day moving average and the day's volume is larger than the highest *down-day* volume in the last 10 sessions.
  - **Algorithm**:
    $$\mathcal{D}_{\text{down}} = \{ \text{Volume}_{t-k} \mid k \in [1, 10], \, \text{Close}_{t-k} < \text{Close}_{t-k-1} \}$$
    $$\text{Is\_Pocket\_Pivot} \iff (\text{Close}_t > \text{EMA}_{10}(t)) \land \left(\text{Volume}_t > \max(\mathcal{D}_{\text{down}})\right)$$
  - **Example**: In the last 10 days, the highest volume recorded on a negative close was 1,200,000 shares. Today price crosses above the 10 EMA with 1,350,000 shares. `Pocket_Pivot_Entry` activates as a bonus signal.

- [ ] **5.4 NSE Security-Wise Delivery Volume Confirmation**
  - **Description**: In Indian equity markets, intraday speculators inflate total volume. The true institutional signal is **Delivery Volume** (shares transferred into demat accounts).
  - **Algorithm**:
    $$\text{Delivery Percentage} = \frac{\text{Deliverable Quantity}}{\text{Traded Quantity}} \times 100\%$$
    $$\text{Condition}: \text{Delivery Percentage} > 40.0\% \implies \text{Technical Score Bonus } (+0.05)$$
  - **Example**: Stock HDFCBANK trades 10 million shares, but 5.2 million shares were marked as actual delivery ($52\% > 40\%$). The Smart Money tracker registers institutional accumulation and awards a $+0.05$ score enhancement.

- [ ] **5.5 Volume Dry-Up During Base Contraction**
  - **Description**: During pullbacks and base consolidations, lack of selling volume proves there is no institutional dumping.
  - **Algorithm**:
    $$\text{DryUp Condition} \iff \text{Volume}_t < \overline{\text{Vol}}_{20}(t) \times 0.80$$
  - **Example**: Average volume is 400,000. During day 4 of a pullback to the 21 EMA, volume shrinks to 180,000 ($45\%$ of average). Selling pressure has exhausted.

---

## Module 6: Momentum Oscillators & Normalized Range Filtering

Oscillators must be used to confirm momentum regimes, not to blindly buy oversold dips in downtrends.

- [ ] **6.1 Relative Strength Index (RSI-14) Gating**
  - **Description**: Constance Brown and Andrew Cardwell showed that in a bull market swing setup, RSI oscillates between 40 and 80, with 50 acting as support.
  - **Algorithm**:
    $$RS = \frac{\text{EMA}_{14}(\text{Up Moves})}{\text{EMA}_{14}(\text{Down Moves})}, \quad RSI_{14} = 100 - \frac{100}{1 + RS}$$
    $$\text{Swing Gating Rule}: 50.0 \le RSI_{14} \le 68.0$$
    Reject if $RSI < 50$ (bearish momentum dominance) or $RSI > 68$ (overextended exhaustion risk).
  - **Example**: Stock SBIN has $RSI_{14} = 56.5$. It falls within $[50, 68]$. The momentum gate passes. If $RSI_{14} = 74.2$, it fails due to being overbought; if $RSI_{14} = 42.0$, it fails due to lack of bullish momentum.

- [ ] **6.2 MACD Signal & Histogram Expansion**
  - **Description**: Moving Average Convergence Divergence tracks the relationship between two exponential moving averages.
  - **Algorithm**:
    $$MACD_{\text{line}} = \text{EMA}_{12}(\text{Close}) - \text{EMA}_{26}(\text{Close})$$
    $$Signal_{\text{line}} = \text{EMA}_9(MACD_{\text{line}})$$
    $$Histogram = MACD_{\text{line}} - Signal_{\text{line}}$$
    $$\text{MACD Bonus Trigger} \iff (MACD_{\text{line}} > Signal_{\text{line}}) \land (Histogram_t > Histogram_{t-1})$$
  - **Example**: $MACD_{\text{line}} = 3.2$, $Signal = 2.8$, $Histogram_t = 0.4$, $Histogram_{t-1} = 0.15$. The histogram is expanding positively above signal, conferring bonus momentum weight.

---

## Module 7: Algorithmic Entry Pattern Recognition Mechanics

Smart Advice executes trades **only** when at least one certified algorithmic entry pattern triggers simultaneously with passing all gates.

```
                  ┌─────────────────────────────────┐
                  │ Does Stock Pass All 3 Gates?    │
                  │ (Trend, Volatility, Volume)     │
                  └───────────────┬─────────────────┘
                                  │ YES
                                  ▼
      ┌─────────────────────────────────────────────────────────┐
      │          Evaluate Active Entry Patterns                 │
      ├────────────────────────┬────────────────────────────────┤
      │ 7.1 Pullback to EMA    │ 7.5 20-Day High Breakout       │
      │ 7.2 Bollinger Squeeze  │ 7.6 Minervini VCP Breakout     │
      │ 7.3 Higher Low Zigzag  │ 7.7 NR7 Squeeze Breakout       │
      │ 7.4 Vol. Contraction   │ 7.8 Pocket Pivot Entry         │
      └────────────────────────┴────────────────────────────────┘
                                  │ AT LEAST ONE TRIGGERED
                                  ▼
                  ┌─────────────────────────────────┐
                  │ Proceed to Multi-Pillar Scoring │
                  └─────────────────────────────────┘
```

- [ ] **7.1 Pattern: Pullback to EMA (21 Exponential Moving Average)**
  - **Description**: In strong trends, institutional buyers add positions when price tests the rising 21-day EMA.
  - **Algorithm**:
    $$\text{EMA}_{21}(t) = \text{Close}_t \times \frac{2}{22} + \text{EMA}_{21}(t-1) \times \left(1 - \frac{2}{22}\right)$$
    $$\text{Distance} = \frac{|\text{Low}_t - \text{EMA}_{21}(t)|}{\text{EMA}_{21}(t)} \times 100\% \le 2.0\%$$
    $$\text{RSI Condition}: 35 \le RSI_{14} \le 65$$
    $$\text{Bullish Candle Required}: \text{Close}_t > \text{Open}_t$$
  - **Example**: Stock BAJFINANCE: $\text{EMA}_{21} = ₹6,800$, today's $\text{Low} = ₹6,840$ (distance $= 0.58\% \le 2.0\%$), $\text{Close} = ₹6,890 > \text{Open} = ₹6,810$ (green candle), $RSI_{14} = 54$. All criteria met; `pullback_to_ema` entry signal fires.

- [ ] **7.2 Pattern: Bollinger Squeeze Breakout**
  - **Description**: Triggers when price explodes outside tight Bollinger Bands following prolonged compression.
  - **Algorithm**:
    $$\text{Bandwidth}_{T-1} \le \text{Squeeze Threshold } (0.10)$$
    $$\text{Breakout Condition}: \text{Close}_T > \text{Upper Band}_T \land \text{Close}_T > \text{Close}_{T-1}$$
    $$\text{Volume Multiplier}: \text{Volume}_T \ge 1.5 \times \overline{\text{Vol}}_{20}$$
  - **Example**: Bandwidth yesterday was 0.07 (tight squeeze). Today price closes at ₹415 while Upper Band is ₹408, with 2.2x the 20-day average volume. `bollinger_squeeze_breakout` triggers.

- [ ] **7.3 Pattern: Higher Low Structure (Pivot Zigzag Identification)**
  - **Description**: Identifies ascending structural support points indicating strong demand floors.
  - **Algorithm**: A swing low at bar $k$ requires:
    $$\text{Low}_k < \min(\text{Low}_{k-3}, \text{Low}_{k-2}, \text{Low}_{k-1}) \land \text{Low}_k < \min(\text{Low}_{k+1}, \text{Low}_{k+2}, \text{Low}_{k+3})$$
    Let $P_1, P_2, P_3$ be the last 3 consecutive swing lows chronologically:
    $$\text{Ascending Condition}: \text{Price}(P_3) > \text{Price}(P_2) > \text{Price}(P_1)$$
    $$\text{Volume Confirmation}: \text{Volume}(\text{Reversal Day}) > \overline{\text{Vol}}_{20}$$
  - **Example**: Low 1 on June 10 is ₹210. Low 2 on July 02 is ₹225. Low 3 on July 24 is ₹238. Since $238 > 225 > 210$, the structural higher-low sequence is intact.

- [ ] **7.4 Pattern: Volatility Contraction (ATR Sequence)**
  - **Description**: Three consecutive decreases in ATR combined with volume dry-up before an explosive move.
  - **Algorithm**:
    $$ATR_{14}(t) < ATR_{14}(t-1) < ATR_{14}(t-2) < ATR_{14}(t-3)$$
    $$\text{Volume}_t < 0.80 \times \overline{\text{Vol}}_{20}(t)$$
    $$\frac{ATR_{14}(t)}{\text{Close}_t} \le 3.0\%$$
  - **Example**: ATR values over the past 4 sessions: ₹24.50, ₹23.80, ₹22.90, ₹21.60 (3 consecutive drops). Today's volume is 70% of 20-day average, and ATR is 2.1% of stock price. `volatility_contraction` pattern activates.

- [ ] **7.5 Pattern: 20-Day High Breakout with Retest Range Validation**
  - **Description**: Richard Donchian channel breakout confirmed by institutional volume and optional retest verification.
  - **Algorithm**:
    $$\text{Highest High}_{20} = \max_{i \in [1, 20]} (\text{High}_{t-i})$$
    $$\text{Breakout Condition}: \text{Close}_t > \text{Highest High}_{20}$$
    $$\text{Volume Confirmation}: \text{Volume}_t \ge 1.50 \times \overline{\text{Vol}}_{20}(t)$$
    $$\text{Retest Rule}: \text{Low}_{\text{recent}} \in [0.97 \times \text{Highest High}_{20}, \, 1.025 \times \text{Highest High}_{20}]$$
    $$\text{Extension Cap}: \frac{\text{Close}_t - \text{EMA}_{21}(t)}{\text{EMA}_{21}(t)} \times 100\% \le 3.5\%$$
  - **Example**: The 20-day high is ₹1,500. Today price breaks to ₹1,520 with 1.8x average volume. The distance from 21 EMA is 2.1% (< 3.5%). The breakout is valid and not overextended.

- [ ] **7.6 Pattern: Minervini Volatility Contraction Pattern (VCP)**
  - **Description**: Mark Minervini's institutional setup characterized by successive contractions of price swings (e.g., -15%, then -8%, then -3%) accompanied by volume dry-up.
  - **Algorithm**:
    $$\text{Consolidation Days} \in [10, 45]$$
    $$\text{Contraction Depth Wave 1} = \frac{\text{High}_1 - \text{Low}_1}{\text{High}_1} \approx 10\% - 25\%$$
    $$\text{Contraction Depth Wave 2} = \frac{\text{High}_2 - \text{Low}_2}{\text{High}_2} \le 0.6 \times \text{Contraction Depth Wave 1}$$
    $$\text{Volume on Final Contraction} \le 0.70 \times \overline{\text{Vol}}_{50}$$
    $$\text{Breakout Trigger}: \text{Close}_t \ge \text{Consolidation High} \land \text{Volume}_t \ge 1.50 \times \overline{\text{Vol}}_{50}$$
  - **Example**: Stock consolidates for 28 days. First pullback was -16%. Second pullback was -7%. In the last 3 days of base, volume dried up to 50% of the 50-day average. Today it breaks out past the base high with 2.1x volume. `minervini_vcp_breakout` fires.

- [ ] **7.7 Pattern: NR7 Squeeze Breakout**
  - **Description**: Entering on the expansion day immediately following an NR7 session.
  - **Algorithm**:
    $$\text{Bar } T-1 \text{ was NR7}$$
    $$\text{Breakout Condition}: \text{Close}_T > \text{High}_{T-1} \land \text{Volume}_T > \text{Volume}_{T-1}$$
  - **Example**: Yesterday was NR7 with a narrow range of ₹6. Today price gaps up and crosses yesterday's high with increasing volume. `nr7_volatility_squeeze` triggers.

- [ ] **7.8 Candlestick Formations (Bullish Engulfing, Hammer, Morning Star)**
  - **Description**: Price action confirmation using classical Japanese candlestick patterns at key support levels.
  - **Algorithm**:
    - **Hammer**:
      $$\text{Lower Shadow} \ge 2 \times \text{Real Body}, \quad \text{Upper Shadow} \le 0.2 \times \text{Real Body}$$
      $$\text{Where Real Body} = |\text{Close} - \text{Open}|, \quad \text{Lower Shadow} = \min(\text{Open}, \text{Close}) - \text{Low}$$
    - **Bullish Engulfing**:
      $$\text{Close}_{t-1} < \text{Open}_{t-1} \quad (\text{Red Candle})$$
      $$\text{Open}_t \le \text{Close}_{t-1} \land \text{Close}_t > \text{Open}_{t-1} \quad (\text{Green Engulfs Red Body})$$
  - **Example**: At the 50-day SMA, stock prints a hammer: Body is ₹4, Lower shadow is ₹10 ($2.5\times$ body), Upper shadow is ₹0.5. Reversal candlestick confirmation confirmed.

---

## Module 8: NSE Option Chain Open Interest (OI) & Derivatives Sentiment

In the Indian equity market, stocks belonging to the F&O (Futures & Options) segment provide transparent derivative clues about institutional positioning.

- [ ] **8.1 Put-Call Ratio (PCR) Calculation**
  - **Description**: PCR measures the ratio of total Put open interest to Call open interest. Put writers are institutions who do not expect the market to fall below strike prices.
  - **Algorithm**:
    $$PCR_{OI} = \frac{\sum \text{Open Interest of all Puts}}{\sum \text{Open Interest of all Calls}}$$
    $$\text{PCR Bullish Threshold}: PCR_{OI} \ge 1.0 \implies \text{Score Bonus } (+0.15)$$
    $$\text{Overbought Warning}: PCR_{OI} > 1.6 \implies \text{Contrarian Reversal Risk}$$
  - **Example**: Stock RELIANCE has total Put OI of 14,500,000 contracts and total Call OI of 11,200,000 contracts. $PCR = 14.5 / 11.2 = 1.29 \ge 1.0$. Derivatives sentiment is verified bullish.

- [ ] **8.2 Max Pain Strike Identification**
  - **Description**: The strike price where option sellers (smart money) face the minimum financial loss upon monthly expiry. Price tends to be pinned near or gravitate toward this strike.
  - **Algorithm**: For every strike $K_j$:
    $$\text{Loss}(K_j) = \sum_{i} \text{CallOI}_i \times \max(0, K_j - K_i) + \sum_{i} \text{PutOI}_i \times \max(0, K_i - K_j)$$
    $$\text{Max Pain Strike} = \arg\min_{K_j} \text{Loss}(K_j)$$
    $$\text{Bullish Alignment}: \text{Current Price} \le \text{Max Pain Strike}$$
  - **Example**: Current stock price is ₹1,480. Max pain strike calculates at ₹1,540. Option market structure exerts an upward gravitational pull toward ₹1,540 before contract expiry.

- [ ] **8.3 Short-Covering vs Long-Unwinding Signatures**
  - **Description**: Analyzing the simultaneous delta in Futures Price and Open Interest reveals institutional participation intent.
  - **Algorithm**:
    | Price Delta | OI Delta | Market State | Algorithmic Signal |
    |---|---|---|---|
    | $+ \Delta P$ | $+ \Delta OI$ | Long Buildup | Strong Bullish (Aggressive Entry) |
    | $+ \Delta P$ | $- \Delta OI$ | Short Covering | Moderate Bullish (Short-lived rally) |
    | $- \Delta P$ | $+ \Delta OI$ | Short Buildup | Strong Bearish (Block Buy) |
    | $- \Delta P$ | $- \Delta OI$ | Long Unwinding | Bearish Liquidation (Block Buy) |
  - **Example**: Stock price rises $+2.4\%$ today while futures Open Interest increases $+8.5\%$. This confirms **Long Buildup** (fresh institutional capital entering long).

---

## Module 9: Fundamental Screening & Corporate Financial Health Scoring

Technical signals without fundamental backing can trap traders in "pump and dump" cycles. Smart Advice applies fundamental guardrails.

- [ ] **9.1 Return on Capital Employed (ROCE) & ROE Gating**
  - **Description**: ROCE measures how efficiently a company generates operating profits from its capital base.
  - **Algorithm**:
    $$\text{ROCE} = \frac{\text{EBIT}}{\text{Total Assets} - \text{Current Liabilities}} \times 100\%$$
    $$\text{Fundamental Rule}: \text{ROCE} \ge 18.0\%$$
  - **Example**: Company has EBIT of ₹450 Cr and Capital Employed of ₹2,000 Cr. $\text{ROCE} = (450 / 2000) \times 100 = 22.5\% \ge 18\%$. Filter passes.

- [ ] **9.2 Debt-to-Equity (D/E) Solvency Ceiling**
  - **Description**: Highly leveraged companies face severe interest rate risk and bankruptcy vulnerability during macroeconomic contractions.
  - **Algorithm**:
    $$\text{D/E Ratio} = \frac{\text{Total Debt}}{\text{Total Shareholders' Equity}}$$
    $$\text{Gating Rule}: \text{D/E Ratio} \le 1.0$$
    *(Note: Special banking/NBFC exceptions where debt is business operating liability).*
  - **Example**: Company has total debt of ₹800 Cr and equity of ₹1,200 Cr. $\text{D/E} = 800 / 1200 = 0.67 \le 1.0$. Solvency test passes.

- [ ] **9.3 Profit & Revenue Growth Trajectory (O'Neil CANSLIM Logic)**
  - **Description**: Sustainable stock outperformance requires verified quarterly and annual earnings growth.
  - **Algorithm**:
    $$\text{EPS Growth}_{3\text{y}} \ge 10.0\% \land \text{EPS Growth}_{5\text{y}} \ge 10.0\%$$
    $$\text{Quarterly EPS Growth (YoY)} \ge 10.0\%$$
  - **Example**: Q1 EPS grew $+18\%$ compared to Q1 last year, and 3-year CAGR EPS growth is $14\%$. Growth metrics pass.

- [ ] **9.4 Fundamental Composite Scoring Algorithm**
  - **Description**: Normalizes valuation, growth, and leverage into a standardized score between $0.0$ and $1.0$.
  - **Algorithm**:
    $$S_{\text{PE}} = \begin{cases} 1.0, & \text{if } PE < 15 \\ 0.5, & \text{if } 15 \le PE < 25 \\ -0.5, & \text{if } PE \ge 25 \end{cases} \times 0.30$$
    $$S_{\text{DE}} = \begin{cases} 1.0, & \text{if } DE \le 0.5 \\ 0.5, & \text{if } 0.5 < DE \le 1.0 \\ -1.0, & \text{if } DE > 1.0 \end{cases} \times 0.25$$
    $$S_{\text{EPS}} = \begin{cases} 1.0, & \text{if } \text{EPS}_g \ge 0.15 \\ 0.5, & \text{if } 0.05 \le \text{EPS}_g < 0.15 \\ 0.0, & \text{if } \text{EPS}_g < 0.05 \end{cases} \times 0.25$$
    $$S_{\text{ROE}} = \begin{cases} 1.0, & \text{if } ROE \ge 0.15 \\ 0.5, & \text{if } 0.10 \le ROE < 0.15 \\ 0.0, & \text{if } ROE < 0.10 \end{cases} \times 0.20$$
    $$\text{Fundamental Score} = \max\left(0.0, \, \frac{S_{\text{PE}} + S_{\text{DE}} + S_{\text{EPS}} + S_{\text{ROE}}}{\sum \text{Weights}}\right)$$
  - **Example**: If normalized sum yields $0.72$, then $\text{Fundamental Score} = 0.72$.

---

## Module 10: Multi-Pillar Scoring Synthesis & Hard Floor Architecture

No single indicator is reliable on its own. Smart Advice combines independent dimensions into an aggregate conviction score with strict floor vetoes.

- [ ] **10.1 Multi-Pillar Weight Distribution**
  - **Description**: Blends technical, fundamental, and sector scores using mathematically disciplined weights.
  - **Algorithm**:
    $$\text{Combined Score} = (S_{\text{Tech}} \times W_{\text{Tech}}) + (S_{\text{Fund}} \times W_{\text{Fund}}) + (S_{\text{Sector}} \times W_{\text{Sector}}) + (S_{\text{Sent}} \times W_{\text{Sent}})$$
    In standard Swing Trading configuration:
    $$W_{\text{Tech}} = 0.65, \quad W_{\text{Fund}} = 0.20, \quad W_{\text{Sector}} = 0.15, \quad W_{\text{Sent}} = 0.0$$
  - **Example**: $S_{\text{Tech}} = 0.85$, $S_{\text{Fund}} = 0.60$, $S_{\text{Sector}} = 0.80$.
    $$\text{Combined Score} = (0.85 \times 0.65) + (0.60 \times 0.20) + (0.80 \times 0.15) = 0.5525 + 0.12 + 0.12 = 0.7925$$

- [ ] **10.2 Strict Floor Enforcement (Non-Compensatory Logic)**
  - **Description**: An extraordinarily high technical score must **never** compensate for a failing fundamental floor, and vice-versa.
  - **Algorithm**:
    $$\text{Technical Floor}: S_{\text{Tech}} \ge 0.35 \quad (\text{if } < 0.35 \implies \text{REJECT})$$
    $$\text{Fundamental Floor}: S_{\text{Fund}} \ge 0.10 \quad (\text{if } < 0.10 \implies \text{REJECT})$$
    $$\text{Buy Recommendation Trigger}: \text{Combined Score} \ge 0.60 \land \text{Floors Passed}$$
  - **Example**: A penny-stock experiencing a short squeeze produces $S_{\text{Tech}} = 0.95$, but due to negative earnings $S_{\text{Fund}} = 0.02$. Since $0.02 < 0.10$, the fundamental floor vetoes the trade immediately. Recommendation = **HOLD (Rejected)**.

- [ ] **10.3 Multi-Factor Momentum Leadership Composite Score**
  - **Description**: Stocks that pass the buy threshold are sorted using a multi-factor composite score to allocate capital to the highest-momentum leaders first.
  - **Algorithm**:
    $$\text{Mom}_{3\text{m}} = \frac{\text{Close}_t}{\text{Close}_{t-63}}, \quad \text{Prox}_{52\text{w}} = \frac{\text{Close}_t}{\max_{i \in [0, 251]} \text{High}_{t-i}}$$
    $$\text{Composite Score} = (\text{Mom}_{3\text{m}} \times 2.0) + \text{Prox}_{52\text{w}} + S_{\text{Tech}}$$
  - **Example**: Stock 1 has $\text{Mom}_{3\text{m}} = 1.30$, $\text{Prox} = 0.98$, $S_{\text{Tech}} = 0.82 \implies \text{Score} = (1.30 \times 2) + 0.98 + 0.82 = 4.40$. Stock 2 calculates at $3.85$. Capital is deployed to Stock 1 first.

---

## Module 11: Dynamic ATR Risk Management & Volatility-Scaled Position Sizing

Capital preservation is the ultimate mathematical determinant of long-term trading survival. Position sizing is never fixed in shares or fixed dollar amounts; it is strictly risk-derived.

- [ ] **11.1 Stop Loss Mathematical Definition**
  - **Description**: The stop-loss is placed beyond normal market noise using the Average True Range (ATR).
  - **Algorithm**:
    $$\text{Stop Loss Price} = \text{Entry Price} - (ATR_{14} \times \text{ATR Stop Multiplier})$$
    $$\text{Risk Per Share } (\text{RPS}) = \text{Entry Price} - \text{Stop Loss Price} = ATR_{14} \times \text{Multiplier}$$
    Where `atr_stop_multiplier` is typically $2.40$ (or $2.50$).
    $$\text{Hard Stop Ceiling}: \text{Stop Loss} \ge \text{Entry Price} \times (1 - 0.08)$$
    The absolute maximum drawdown allowed on a single trade is capped at 8.0%.
  - **Example**: Entry Price = ₹1,000, $ATR_{14} = ₹25$, Multiplier = 2.4.
    $$\text{Stop Loss} = 1000 - (25 \times 2.4) = 1000 - 60 = ₹940.00$$
    $$\text{Risk per share} = ₹60.00 \quad (6.0\% \le 8.0\% \text{ ceiling})$$

- [ ] **11.2 Risk-Reward Ratio (RR) Floor Check**
  - **Description**: Never execute a trade where potential reward does not sufficiently exceed initial capital at risk.
  - **Algorithm**:
    $$\text{Target 1 Price} = \text{Entry Price} + (ATR_{14} \times \text{T1 Multiplier})$$
    $$\text{Risk-Reward Ratio } (RR) = \frac{\text{Target 1 Price} - \text{Entry Price}}{\text{Entry Price} - \text{Stop Loss Price}} = \frac{T_1 \text{ Mult}}{\text{SL Mult}}$$
    $$\text{Gating Condition}: RR \ge 1.20$$
  - **Example**: With T1 multiplier $= 3.50$ and SL multiplier $= 2.40$:
    $$RR = \frac{3.50}{2.40} = 1.458 \ge 1.20 \quad (\text{PASS})$$

- [ ] **11.3 Volatility-Scaled Risk Budgeting**
  - **Description**: When broader or stock-specific volatility is historically elevated, reduce the percentage of portfolio equity risked per trade.
  - **Algorithm**: Let $P_{\text{ATR}} \in [0, 100]$ be the 100-day ATR percentile.
    $$\text{Scaled Risk \%} = \begin{cases} R_{\max}, & \text{if } P_{\text{ATR}} \le 25 \\ R_{\max} \times \text{MinMultiplier}, & \text{if } P_{\text{ATR}} \ge 75 \\ R_{\max} \times \left[ 1.0 - \frac{P_{\text{ATR}} - 25}{50}(1.0 - \text{MinMultiplier}) \right], & \text{otherwise} \end{cases}$$
    Where $R_{\max} = 1.5\%$ of account balance, and $\text{MinMultiplier} = 0.70$.
  - **Example**: Account Balance = ₹100,000. $P_{\text{ATR}} = 75\%$ (high volatility). Scaled Risk % $= 1.5\% \times 0.70 = 1.05\%$. Capital at risk budget $= ₹1,050$ instead of ₹1,500.

- [ ] **11.4 Two-Constraint Position Sizing Formulation**
  - **Description**: Position size in shares is strictly bounded by two simultaneous limits: **Risk Budget** and **Max Capital Allocation Cap**.
  - **Algorithm**:
    $$\text{Size}_{\text{risk}} = \left\lfloor \frac{\text{Account Balance} \times \text{Scaled Risk \%}}{\text{Risk Per Share}} \right\rfloor$$
    $$\text{Size}_{\text{capital}} = \left\lfloor \frac{\text{Account Balance} \times \text{Max Position \%}}{\text{Entry Price}} \right\rfloor$$
    $$\text{Final Position Size (Shares)} = \min(\text{Size}_{\text{risk}}, \, \text{Size}_{\text{capital}})$$
    Where $\text{Max Position \%}$ is typically $18.0\%$ of total portfolio equity.
  - **Example**:
    - Account Balance: ₹100,000
    - Entry Price: ₹500
    - Stop Loss: ₹470 $\implies \text{Risk Per Share} = ₹30$
    - Risk Budget (1.5%): ₹1,500
    - $\text{Size}_{\text{risk}} = \lfloor 1500 / 30 \rfloor = 50 \text{ shares}$
    - Capital Cap (18%): $₹18,000 \implies \text{Size}_{\text{capital}} = \lfloor 18000 / 500 \rfloor = 36 \text{ shares}$
    - $\text{Final Size} = \min(50, 36) = \mathbf{36 \text{ shares}}$
    - Total Capital Deployed: $36 \times 500 = ₹18,000$
    - Actual Capital at Risk: $36 \times 30 = ₹1,080$ ($1.08\%$ of account, well within $1.5\%$ limit).

---

## Module 12: Multi-Target Profit Taking, Trailing Stops & Time Stops

A trade plan is incomplete without an asymmetric exit strategy that locks in profits while allowing winning positions to run.

- [ ] **12.1 Staged Multi-Target Profit Scaling**
  - **Description**: Taking partial profits at Target 1 reduces risk and finances the trade, while Target 2 captures large multi-week trend extensions.
  - **Algorithm**:
    $$\text{Target 1 Price} = \text{Entry Price} + (ATR_{14} \times 3.5)$$
    $$\text{Sell Quantity at T1} = \lfloor \text{Initial Position Size} \times 0.30 \rfloor$$
    $$\text{Target 2 Price} = \text{Entry Price} + (ATR_{14} \times 7.0)$$
    $$\text{Sell Quantity at T2} = \text{Remaining 70\% Position}$$
  - **Example**: Holding 100 shares entered at ₹1,000 with $ATR = ₹20$.
    - Target 1: $1000 + (20 \times 3.5) = ₹1,070$. Sell 30 shares at ₹1,070 (+7.0% gain).
    - Target 2: $1000 + (20 \times 7.0) = ₹1,140$. Remaining 70 shares held for ₹1,140 or trailing stop.

- [ ] **12.2 Breakeven Ratchet Protocol**
  - **Description**: The instant Target 1 is achieved, the stop-loss on the remaining 70% position is automatically adjusted to eliminate downside risk.
  - **Algorithm**:
    $$\text{When Price} \ge \text{Target 1} \implies \text{New Stop Loss} = \max(\text{Current Stop Loss}, \, \text{Entry Price} \times 1.002)$$
    The $+0.2\%$ buffer covers round-trip brokerage and statutory charges (STT, exchange fees).
  - **Example**: Entry at ₹1,000. Stop-loss was ₹952. Once price reaches ₹1,070 (T1), stop-loss is automatically updated to ₹1,002. The trade is now mathematically guaranteed non-losing.

- [ ] **12.3 Regime-Adaptive ATR Trailing Stops**
  - **Description**: Trailing stops lock in open profits as the trend extends. The trailing tightness adapts to the macro market regime.
  - **Algorithm**:
    $$\text{Trailing Stop} = \max_{t \ge T_{\text{entry}}} \left( \text{High}_t - (ATR_{14}(t) \times \text{Trail Multiplier}) \right)$$
    Where:
    $$\text{Trail Multiplier} = \begin{cases} 2.80, & \text{in Bull Market (Loose trail to ride mega-trends)} \\ 2.00, & \text{in Sideways/Bear Market (Tight trail to protect gains)} \end{cases}$$
  - **Example**: Stock reaches ₹1,200 in Bull market with $ATR = ₹25$. Trailing stop $= 1200 - (25 \times 2.8) = 1200 - 70 = ₹1,130$. If price pulls back to ₹1,130, remaining shares are exited.

- [ ] **12.4 Stagnant Trade Time Stop**
  - **Description**: Opportunity cost destroys returns. If a stock does not produce an expansion move within a predetermined number of bars, exit to free capital.
  - **Algorithm**:
    $$\text{Bars Held} = t_{\text{current}} - t_{\text{entry}}$$
    $$\text{Time Stop Trigger} \iff (\text{Bars Held} \ge 15 \text{ bars}) \land (\text{Price} < \text{Target 1}) \land (\text{Unrealized PnL} < 2.0\%)$$
  - **Example**: Stock entered 15 trading days ago is hovering at $+0.5\%$. The engine liquidates the position to deploy funds into active, moving setups.

- [ ] **12.5 Leader Exception Protocol (O'Neil 8-Week Rule)**
  - **Description**: If a stock surges $+20\%$ or more within the first 3 weeks of entry, it represents an exceptional market leader. Do not exit at Target 1; hold for an 8-week trend ride.
  - **Algorithm**:
    $$\text{Gain at Bar } t \le 15 = \frac{\text{Close}_t - \text{Entry}}{\text{Entry}} \times 100\% \ge 20.0\%$$
    $$\text{Action} \implies \text{Bypass T1 exit, hold full position for up to 8 weeks, trail with 20-day SMA}$$
  - **Example**: Stock gains +22% in 9 days. System cancels the 30% partial profit sell order at T1, converts trade into a "Leader Ride", and trails stops along the rising 20-day SMA.

---

## Module 13: Systematic Pyramiding & Capital Compounding Rules

True outperformance comes not from having many small winners, but from compounding winning positions as the market proves your hypothesis right.

- [ ] **13.1 Pyramiding Activation Criteria**
  - **Description**: Adding to winners is allowed only under strict risk controls; adding to losing positions is strictly banned.
  - **Algorithm**:
    $$\text{Pyramiding Allowed} \iff (\text{Regime} == \text{BULL}) \land (\text{Trade PnL} > 0) \land (\text{Target 1 not yet hit})$$
  - **Example**: Position entered at ₹1,000. Price moves up to ₹1,030. Stop-loss has moved up. Pyramiding module evaluates conditions for First Add.

- [ ] **13.2 Multi-Stage Pyramid Trigger Steps**
  - **Description**: Compounding happens at systematic ATR milestones, adding smaller fractions each step (inverted pyramid structure).
  - **Algorithm**:
    - **Step 1 (First Add)**:
      $$\text{Trigger} = \text{Entry Price} + (1.5 \times ATR_{\text{entry}})$$
      $$\text{Add Size} = 50\% \times \text{Initial Position Size}$$
    - **Step 2 (Second Add)**:
      $$\text{Trigger} = \text{Entry Price} + (3.0 \times ATR_{\text{entry}})$$
      $$\text{Add Size} = 30\% \times \text{Initial Position Size}$$
  - **Example**: Initial size = 100 shares at ₹1,000 ($ATR = ₹20$).
    - Step 1 Trigger: $1000 + (1.5 \times 20) = ₹1,030$. Add 50 shares. Total = 150 shares.
    - Step 2 Trigger: $1000 + (3.0 \times 20) = ₹1,060$. Add 30 shares. Total = 180 shares.

- [ ] **13.3 Blended Average Price & Risk Recalculation**
  - **Description**: When shares are added, the overall average bought price increases. The stop-loss must ratchet upward to ensure the net portfolio risk never increases.
  - **Algorithm**:
    $$\text{Blended Price} = \frac{\sum (\text{Shares}_k \times \text{Price}_k)}{\sum \text{Shares}_k}$$
    $$\text{Mandatory Ratchet}: \text{New Stop Loss} \ge \text{Blended Price} - (1.5 \times ATR_{\text{current}})$$
    $$\text{Total Open Risk} = \text{Total Shares} \times (\text{Blended Price} - \text{New SL}) \le \text{Initial Risk Budget}$$
  - **Example**:
    - Tranche 1: 100 shares @ ₹1000. Risk budget = ₹1,500 (SL @ ₹985, risk = ₹15/sh $\times 100 = ₹1,500$).
    - Tranche 2: 50 shares @ ₹1030. Total shares = 150. Blended price $= (100,000 + 51,500)/150 = ₹1,010$.
    - Move stop loss to ₹1,005. New total risk $= 150 \times (1,010 - 1,005) = ₹750$.
    - Total risk has decreased from ₹1,500 to ₹750 despite owning 50% more shares.

---

## Module 14: Portfolio Construction, Concurrency & Circuit Breakers

A collection of individually sound trades can still fail if portfolio-level concentration and drawdown limits are neglected.

- [ ] **14.1 Portfolio Concurrency Cap**
  - **Description**: Bounding maximum simultaneous positions to prevent over-diversification (which dilutes returns) and under-diversification (which concentrates risk).
  - **Algorithm**:
    $$\text{Active Positions} \le \text{Max Positions } (6)$$
    If $\text{Active Positions} = 6$, new buy signals are queued or discarded based on Relative Strength ranking.
  - **Example**: System has 6 active positions. Stock TITAN triggers a valid buy signal. Because the portfolio is fully allocated, TITAN is not bought unless an existing position hits an exit or stop today.

- [ ] **14.2 Sector Concentration Ceiling**
  - **Description**: Prevents holding 4 IT stocks or 4 Auto stocks simultaneously, which exposes the portfolio to sector-specific policy or tariff shocks.
  - **Algorithm**:
    $$\sum_{i \in \text{Sector } S} \text{Capital Deployed}_i \le 35.0\% \times \text{Total Portfolio Equity}$$
  - **Example**: Total equity is ₹100,000. Max allocation to Auto sector is ₹35,000. If M&M (₹18,000) and TATAMOTORS (₹15,000) are held, a new buy signal in MARUTI is blocked by the sector concentration ceiling.

- [ ] **14.3 Drawdown Pause & Circuit Breaker Logic**
  - **Description**: Protects capital during sudden black-swan crashes by automatically pausing new risk after peak-to-trough drawdowns.
  - **Algorithm**:
    $$\text{Drawdown \%} = \frac{\text{Peak Equity} - \text{Current Equity}}{\text{Peak Equity}} \times 100\%$$
    $$\text{Pause Condition}: \text{Drawdown \%} \ge 12.0\% \implies \text{Halt New Purchases for 7 days Minimum}$$
    $$\text{Resume Condition}: \text{Drawdown \%} \le 8.0\% \land \text{Cooldown Bars } (15) \text{ elapsed}$$
  - **Example**: Peak equity was ₹150,000. Due to market correction, equity drops to ₹131,000 (drawdown = 12.67%). Drawdown Circuit Breaker trips: no new buy recommendations are issued until cooldown requirements are met.

- [ ] **14.4 Maximum Daily Portfolio Loss Stop**
  - **Description**: An intraday market emergency stop that prevents cascading position wipeouts.
  - **Algorithm**:
    $$\text{Daily PnL \%} = \frac{\text{Current Equity} - \text{Equity}_{\text{EOD Yesterday}}}{\text{Equity}_{\text{EOD Yesterday}}} \times 100\%$$
    $$\text{If Daily PnL \%} \le -4.0\% \implies \text{Liquidate Intraday/Tighten All Swing Trailing Stops}$$
  - **Example**: Market crashes -3.5% intraday due to geopolitical news; portfolio falls -4.1%. Circuit breaker triggers defensive risk procedures.

---

## Module 15: The 6-Phase Ultimate Backtest & Statistical Validation Pipeline

The ultimate differentiator between a toy model and an institutional quantitative engine is **rigorous, multi-phase statistical validation**. Smart Advice validates every strategy through a 6-phase pipeline before live deployment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 6-PHASE ULTIMATE BACKTEST PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase 1: Historical Backtest with Realistic Transaction Frictions          │
│           (STT, GST, Exchange Fees, Slippage, Gap Risk)                     │
│                                  │                                          │
│  Phase 2: Statistical Validation                                            │
│           (Deflated Sharpe Ratio DSR, Monte Carlo Permutation, MLRS)        │
│                                  │                                          │
│  Phase 3: Walk-Forward Monte Carlo Simulation                               │
│           (Rolling Out-of-Sample Windows, Parameter Stability)              │
│                                  │                                          │
│  Phase 4: Stress Testing                                                    │
│           (Regime Segregation, 2x Frictions, Parameter Sensitivity +/-20%) │
│                                  │                                          │
│  Phase 5: Trade Diagnostics                                                 │
│           (MFE/MAE Distributions, Exit Reason Breakdown, Hold Time)         │
│                                  │                                          │
│  Phase 6: Composite Confidence Scoring (0-100)                              │
│           (EmpiricalCAGR Projection, Production Deployment Gating)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

- [ ] **15.1 Phase 1: Historical Simulation with Realistic Frictions**
  - **Description**: Backtesting without execution frictions produces fictional returns. Indian markets impose significant statutory friction.
  - **Algorithm**:
    $$\text{Brokerage} = \min(₹20, \, 0.05\% \times \text{Turnover})$$
    $$\text{STT (Securities Transaction Tax)} = 0.10\% \text{ on Delivery Buy & Sell}$$
    $$\text{Exchange Turnover Charges} = 0.00345\%$$
    $$\text{SEBI Turnover Charges} = ₹10 \text{ per Crore} = 0.0001\%$$
    $$\text{GST} = 18\% \text{ on (Brokerage + Exchange Fees)}$$
    $$\text{Stamp Duty} = 0.015\% \text{ on Buy Value}$$
    $$\text{Slippage Model} = 0.05\% \text{ to } 0.15\% \text{ based on market cap}$$
    $$\text{Execution Fill Price} = \text{Open}_{t+1} \times (1 + \text{Slippage})$$
  - **Example**: Buying ₹100,000 of shares does not cost ₹100,000. With STT, stamp duty, brokerage, exchange fees, and slippage, initial outlay is ₹100,285. This difference compounds dramatically over 200 trades.

- [ ] **15.2 Phase 2: Statistical Significance Testing**
  - **Description**: Proves whether strategy returns stem from genuine alpha (skill) rather than random noise (luck).
  - **Algorithm**:
    - **Deflated Sharpe Ratio (DSR)**:
      Adjusts the annualized Sharpe Ratio for non-normality (skewness, kurtosis) and multiple hypothesis testing (data mining bias):
      $$DSR = \Phi \left( \frac{(\widehat{SR} - SR_0)\sqrt{T-1}}{\sqrt{1 - \widehat{\gamma}_3 \widehat{SR} + \frac{\widehat{\gamma}_4 - 1}{4}\widehat{SR}^2}} \right)$$
      $$\text{Validation Condition}: DSR \ge 0.50$$
    - **Monte Carlo Return Permutation**:
      Shuffles trade return sequences 1,000 times to construct an empirical null distribution:
      $$p\text{-value} = \frac{1}{N}\sum_{k=1}^{1000} \mathbb{I}(SR_{\text{shuffled}}^{(k)} \ge SR_{\text{strategy}}) < 0.05$$
    - **Minimum Track Record Length (MLRS)**:
      $$\text{Required Track Record (Years)} = \frac{1 + (1 - \widehat{\gamma}_3 \widehat{SR} + \frac{\widehat{\gamma}_4 - 1}{4}\widehat{SR}^2)}{\widehat{SR}^2}$$
  - **Example**: Strategy produces Sharpe $= 0.95$. After accounting for negative skewness (-0.4) and kurtosis (4.2), DSR calculates at $0.68 \ge 0.50$ and Monte Carlo $p$-value is $0.018 < 0.05$. The returns are statistically significant.

- [ ] **15.3 Phase 3: Walk-Forward Monte Carlo Simulation**
  - **Description**: Evaluates strategy robustness across rolling out-of-sample time slices to prevent curve-fitting.
  - **Algorithm**:
    Divide a 10-year dataset into rolling 2-year in-sample training and 1-year out-of-sample testing windows:
    $$\text{Window}_k: \text{Train}[Y_k, Y_{k+2}] \longrightarrow \text{Test}[Y_{k+2}, Y_{k+3}]$$
    $$\text{Walk-Forward Efficiency (WFE)} = \frac{\text{Mean CAGR}_{\text{Out-of-Sample}}}{\text{CAGR}_{\text{In-Sample}}} \ge 0.70$$
    $$\text{Positive Window Ratio} = \frac{\sum \mathbb{I}(\text{CAGR}_{\text{OOS}} > 0)}{N_{\text{windows}}} \ge 75\%$$
  - **Example**: In-sample CAGR is 24%. Mean out-of-sample CAGR across 8 rolling windows is 19.2%. $WFE = 19.2 / 24 = 0.80 \ge 0.70$. Positive windows: 7 out of 8 (87.5%). The strategy generalizes effectively.

- [ ] **15.4 Phase 4: Stress Testing Across Regimes & Perturbations**
  - **Description**: Subjects the strategy to extreme historical regimes and parameter variations to ensure no sudden cliffs exist.
  - **Algorithm**:
    - **Crash Stress Test**: Isolate periods like March 2020 (COVID crash) and 2008 Lehman collapse:
      $$\text{Max Drawdown in Crash} \le 20.0\%$$
    - **Parameter Perturbation (+/-20%)**:
      Vary parameters (e.g., ADX threshold $20 \to [16, 24]$, SMA period $50 \to [40, 60]$, Stop ATR $2.4 \to [1.9, 2.9]$):
      $$\text{Parameter Fragility} = \frac{\sigma_{\text{CAGR}}}{\mu_{\text{CAGR}}} \le 0.25$$
    - **Cost Stress Test (2x Fees)**:
      Double the STT and slippage. CAGR must remain positive (> 10%).
  - **Example**: Varying the ADX threshold from 16 to 24 results in CAGRs between 22.1% and 25.8%. Because returns remain stable without collapsing, the strategy is parameter robust.

- [ ] **15.5 Phase 5: Trade Diagnostics (MFE/MAE & Reason Distributions)**
  - **Description**: Granular inspection of trade execution telemetry to uncover subtle leaks.
  - **Algorithm**:
    - **Maximum Adverse Excursion (MAE)**:
      $$MAE = \min_{t \in [T_{\text{entry}}, T_{\text{exit}}]} \left( \frac{\text{Low}_t - \text{Entry}}{\text{Entry}} \times 100\% \right)$$
    - **Maximum Favorable Excursion (MFE)**:
      $$MFE = \max_{t \in [T_{\text{entry}}, T_{\text{exit}}]} \left( \frac{\text{High}_t - \text{Entry}}{\text{Entry}} \times 100\% \right)$$
    - **Exit Reason Distribution**: Ensure trades are not predominantly exiting via panic stops:
      $$\text{Ratio} = \frac{N_{\text{Target 1 Hits}} + N_{\text{Target 2 Hits}}}{N_{\text{Stop Loss Hits}}} \ge 1.10$$
  - **Example**: If 75% of winning trades show an MAE of less than 2.0% before reaching Target 1 (+8.5%), the entry timing precision is mathematically sound.

- [ ] **15.6 Phase 6: Composite Confidence Score & Realistic CAGR Formulation**
  - **Description**: Synthesizes all 5 validation phases into an objective 0–100 score and projects real-world expected return.
  - **Algorithm**:
    $$\text{Confidence Score} = (0.25 \times C_{\text{Historical}}) + (0.25 \times C_{\text{Statistical}}) + (0.25 \times C_{\text{WalkForward}}) + (0.25 \times C_{\text{Stress}})$$
    $$\text{Realistic CAGR} = \text{CAGR}_{\text{Phase 1}} \times \left( \frac{\text{Confidence Score}}{100} \right) \times WFE$$
    $$\text{Production Ready} \iff \text{Confidence Score} \ge 60 \land \text{Max DD} \le 18\%$$
  - **Example**: Phase 1 CAGR is 26.13%, Confidence Score is 55/100, WFE is 0.91.
    $$\text{Realistic Expected CAGR} = 26.13\% \times 0.55 \times 0.91 \approx 13.1\%$$
    This realistic expectation protects the trader from overleveraging based on historical curves.

---

## Module 16: Live/Paper Execution State Machine, Replay Engine & Settlement

How the algorithmic decisions transition into order execution and state reconciliation.

- [ ] **16.1 Automated Order Execution State Machine**
  - **Description**: Every recommendation progresses through a deterministic lifecycle without ambiguity.
  - **Algorithm**:
    $$\text{PENDING\_ENTRY} \xrightarrow{\text{Fill}} \text{OPEN} \xrightarrow{\text{T1 Hit}} \text{HALF\_CLOSED} \xrightarrow{\text{T2 or SL}} \text{CLOSED}$$
    If order is not filled within current session, transition to `EXPIRED`.
  - **Example**: System generates Buy Signal for 40 shares of TRENT at ₹7,100 limit. Order fills at open $\to$ state changes to `OPEN`.

- [ ] **16.2 Intraday Gap Risk & Circuit Limiter Logic**
  - **Description**: Indian stocks can hit 5%, 10%, or 20% upper/lower price bands (circuits) where trading freezes.
  - **Algorithm**:
    $$\text{If } \text{Open}_{t+1} < \text{Stop Loss Price} \implies \text{Exit Price} = \text{Open}_{t+1} \quad (\text{Slippage Gap})$$
    $$\text{Circuit Check}: \text{If stock is locked in Lower Circuit} \implies \text{Exit deferred to first unfreezing bar}$$
  - **Example**: Stop loss is ₹500. Bad earnings cause the stock to gap down to open at ₹475. The simulation records the exit at ₹475, not the idealized ₹500.

- [ ] **16.3 End-of-Day (EOD) Settlement & Virtual Ledger Reconciliation**
  - **Description**: Reconciles cash, margin, unrealized PnL, realized PnL, and broker ledger balances after market close (15:30 IST).
  - **Algorithm**:
    $$\text{Total Equity}_T = \text{Cash Balance}_T + \sum_{p \in \text{Open Positions}} (\text{Quantity}_p \times \text{Close}_{p, T})$$
    $$\text{Realized PnL}_T = \sum_{c \in \text{Closed Trades Today}} (\text{Sell Proceeds} - \text{Cost Basis} - \text{Frictions})$$
    $$\text{Unrealized PnL}_T = \sum_{p \in \text{Open Positions}} \text{Quantity}_p \times (\text{Close}_{p, T} - \text{Bought Price}_p)$$
  - **Example**: Cash = ₹45,000. Open position 1: 50 shares of TCS @ ₹4,100 (bought @ ₹4,000). Total Equity $= 45,000 + (50 \times 4,100) = ₹250,000$. Unrealized PnL $= 50 \times 100 = +₹5,000$.

- [ ] **16.4 Historical Replay Simulation Engine**
  - **Description**: Plays market history day-by-day exactly as if trading live, allowing real-time debugging without forward data contamination.
  - **Algorithm**:
    $$\text{For } \text{Date } D \in [D_{\text{start}}, D_{\text{end}}]:$$
    1. Lock data to $t \le D$.
    2. Check Macro Regime on Date $D$.
    3. Update trailing stops and execute exit orders on open positions using Date $D$ OHLC.
    4. Screen universe for entry patterns on Date $D$ close.
    5. Allocate position sizes and queue orders for Date $D+1$ open.
    6. Update virtual ledger and advance clock $D \leftarrow D + 1$.
  - **Example**: In replay mode on 2024-03-15, the system sees only candles up to March 15. The subsequent rally or drop on March 18 is physically invisible to the screener.

---

## 🔬 Comprehensive Concrete Numerical Walkthrough

To cement complete understanding, here is the end-to-end execution of a single recommendation through every mathematical gate:

### 1. Scenario Inputs
- **Account Total Equity**: ₹100,000
- **Stock**: BHARAT ELECTRONICS LIMITED (`BEL.NS`)
- **Current Price ($Close_T$)**: ₹300.00
- **Date**: 2026-06-15

### 2. Macro Check
- Nifty 50 Close: 23,450 | Nifty 200 SMA: 22,100 $\implies 23,450 > 22,100 \implies$ **BULL REGIME (PASS)**.
- Universe Breadth: 48% of NSE stocks > 20 SMA $\implies 48\% \ge 35\% \implies$ **BREADTH OK (PASS)**.

### 3. Hard Liquidity Filter
- Market Cap: ₹73,000 Cr ($> ₹1,000$ Cr) $\implies$ **PASS**.
- 20-day Average Volume: 8,400,000 shares ($> 150,000$) $\implies$ **PASS**.
- Price Range: ₹300 ($₹50 \le 300 \le ₹8,000$) $\implies$ **PASS**.

### 4. Technical Gates
- **Trend Gate**:
  - $ADX_{14} = 26.2 \ge 20$ $\implies$ PASS.
  - $+DI_{14} = 29.1 > -DI_{14} = 14.2$ $\implies$ PASS.
  - $Close (₹300) > SMA_{50} (₹282) > SMA_{150} (₹260) > SMA_{200} (₹245)$ $\implies$ PASS.
- **Volatility Gate**:
  - $ATR_{14} = ₹8.50$. Historical 100-day percentile $= 44\%$ ($5\% \le 44\% \le 90\%$) $\implies$ PASS.
- **Volume Gate**:
  - Today's Volume: 11,200,000 shares vs 20-day mean 8,400,000 ($1.33\times \ge 0.40$) $\implies$ PASS.
  - OBV 10-day regression slope: $+142,000 > 0$ $\implies$ PASS.
- **52-Week High Proximity**:
  - 52-Week High = ₹312. Proximity $= (312 - 300)/312 = 3.84\% \le 25\%$ $\implies$ PASS.

### 5. Algorithmic Entry Pattern Check
- `pullback_to_ema`:
  - $EMA_{21} = ₹296.00$. Distance $= |300 - 296| / 296 = 1.35\% \le 2.0\%$.
  - $RSI_{14} = 58.2 \in [35, 65]$.
  - Bullish Candle: $Close (₹300) > Open (₹297)$.
  - **Pattern Triggered**: `pullback_to_ema` = **ACTIVE (1)**.

### 6. Scoring Synthesis
- $S_{\text{Tech}} = 0.82$ (Gates passed, pattern triggered, OBV slope positive, delivery $> 40\%$).
- $S_{\text{Fund}} = 0.70$ (ROCE = 28%, D/E = 0.0, 3-yr EPS growth = 18%).
- $S_{\text{Sector}} = 0.80$ (Defense sector momentum strong).
- **Combined Score**:
  $$\text{Score} = (0.82 \times 0.65) + (0.70 \times 0.20) + (0.80 \times 0.15) = 0.533 + 0.140 + 0.120 = \mathbf{0.793}$$
- Floor Checks: $0.82 \ge 0.35$ (Tech Pass), $0.70 \ge 0.10$ (Fund Pass), $0.793 \ge 0.60$ (Buy Pass).
- **Recommendation Strength**: **BUY**.

### 7. Risk Management & Position Sizing
- Entry Price: ₹300.00
- $ATR_{14} = ₹8.50$.
- Stop Loss Multiplier = 2.4.
  $$\text{Stop Loss Price} = 300 - (8.50 \times 2.4) = 300 - 20.40 = \mathbf{₹279.60}$$
  $$\text{Risk Per Share} = ₹300.00 - ₹279.60 = ₹20.40 \quad (6.8\% \text{ loss risk})$$
- Target 1 (3.5 ATR): $300 + (8.50 \times 3.5) = 300 + 29.75 = \mathbf{₹329.75}$ (+9.9%)
- Target 2 (7.0 ATR): $300 + (8.50 \times 7.0) = 300 + 59.50 = \mathbf{₹359.50}$ (+19.8%)
- Risk-Reward Check: $RR = 29.75 / 20.40 = 1.458 \ge 1.20 \implies$ PASS.
- Position Sizing Calculation:
  - Account Equity: ₹100,000.
  - Risk Budget: $1.5\% \times 100,000 = ₹1,500$.
  - $\text{Size}_{\text{risk}} = \lfloor 1500 / 20.40 \rfloor = 73 \text{ shares}$.
  - Capital Limit ($18\%$ max): $₹18,000 / 300 = 60 \text{ shares}$.
  - $\text{Final Size} = \min(73, 60) = \mathbf{60 \text{ shares}}$.
  - Total Capital Required: $60 \times 300 = ₹18,000$.
  - Maximum Capital Risk: $60 \times 20.40 = ₹1,224$ (1.22% of account equity).

### 8. Trade Execution Plan
- Buy 60 shares of `BEL.NS` at ₹300.00.
- Place hard Stop Loss at ₹279.60.
- When price touches ₹329.75 (T1):
  - Sell 18 shares (30% of 60 shares), pocketing ₹535.50 profit.
  - Ratchet stop loss on remaining 42 shares to breakeven (₹300.60).
- Hold remaining 42 shares with a $2.8\times$ ATR trailing stop toward Target 2 (₹359.50).

---

## Module 17: Intraday Quantitative Engine & VWAP Microstructure

Intraday recommendation algorithms must account for continuous order flow dynamics, intraday session boundaries, and mandatory broker square-off rules.

- [ ] **17.1 Anchored VWAP (Volume-Weighted Average Price) Formulation**
  - **Description**: Institutional traders benchmark execution efficiency against VWAP. Prices above daily session VWAP indicate institutional buyers are willing to pay a premium over the day's average transaction cost.
  - **Algorithm**: Resetting at 09:15 IST each session:
    $$\text{Typical Price}_i = \frac{\text{High}_i + \text{Low}_i + \text{Close}_i}{3}$$
    $$\text{VWAP}_t = \frac{\sum_{i=1}^{t} (\text{Typical Price}_i \times \text{Volume}_i)}{\sum_{i=1}^{t} \text{Volume}_i}$$
    $$\text{Variance}_t = \frac{\sum_{i=1}^{t} \text{Volume}_i \times (\text{Typical Price}_i - \text{VWAP}_t)^2}{\sum_{i=1}^{t} \text{Volume}_i}$$
    $$\text{Upper Band}_t = \text{VWAP}_t + 1.5\sqrt{\text{Variance}_t}, \quad \text{Lower Band}_t = \text{VWAP}_t - 1.5\sqrt{\text{Variance}_t}$$
    $$\text{Intraday Bullish Setup} \iff (\text{Price}_t > \text{VWAP}_t) \land (\text{VWAP}_t > \text{VWAP}_{t-1})$$
  - **Example**: Stock trades at ₹1,245. VWAP is ₹1,238 and sloping upward. Price pulls back to ₹1,239.50, tests VWAP, and prints a green reversal candle with rising volume. The VWAP support bounce triggers an intraday long signal.

- [ ] **17.2 Opening Range Breakout (ORB-15 & ORB-30) Mechanics**
  - **Description**: The first 15 to 30 minutes of the Indian trading session (09:15 to 09:45 IST) reflect overnight sentiment and institutional order imbalances.
  - **Algorithm**:
    $$\text{ORB High} = \max_{t \in [09:15, 09:30]} (\text{High}_t), \quad \text{ORB Low} = \min_{t \in [09:15, 09:30]} (\text{Low}_t)$$
    $$\text{Breakout Condition} \iff (\text{Close}_{09:31} > \text{ORB High}) \land (\text{Volume}_{09:31} \ge 1.8 \times \overline{\text{Vol}}_{15\text{m}})$$
    $$\text{Opening Range Extension Ratio} = \frac{\text{ORB High} - \text{ORB Low}}{\text{Daily } ATR_{14}} \le 0.45$$
    Reject the breakout if the opening range has already consumed more than 45% of the expected daily ATR, as risk-reward becomes unfavorable.
  - **Example**: Daily ATR is ₹40. The 15-minute opening range is High ₹810, Low ₹798 (range = ₹12, which is $30\%$ of ATR $\le 45\%$). At 09:31, candle closes at ₹812 with $2.1\times$ volume. Entry is valid with initial stop at ₹804 (VWAP).

- [ ] **17.3 Multi-Timeframe (MTF) Alignment Matrix**
  - **Description**: Intraday setups have dramatically higher win rates when aligned with higher timeframe (Daily and Weekly) momentum structures.
  - **Algorithm**:
    $$\text{Higher Timeframe (HTF) Trend} = \mathbb{I}(\text{Close}_{\text{Daily}} > \text{SMA}_{50}(\text{Daily})) \land \mathbb{I}(RSI_{14}(\text{Daily}) > 50)$$
    $$\text{Lower Timeframe (LTF) Trigger} = \text{ORB Breakout on 15m} \lor \text{Pullback to 20-EMA on 5m}$$
    $$\text{Execution Rule} = \text{HTF Trend} == \text{TRUE} \land \text{LTF Trigger} == \text{TRUE}$$
  - **Example**: On the daily chart, stock is in a confirmed stage-2 uptrend above 50 SMA. At 10:15 IST, the 5-minute chart prints an EMA-20 pullback with volume dry-up. The MTF gate confirms the trade.

- [ ] **17.4 Mandatory Intraday Auto-Square-Off State Machine**
  - **Description**: In Indian equity markets, intraday product types (MIS / Cover Orders) are forcefully squared off by brokers between 15:15 and 15:20 IST.
  - **Algorithm**:
    $$\text{If Time} \ge 15:15 \text{ IST and Position is OPEN}:$$
    $$\text{Action} \implies \text{Emit IMMEDIATE MARKET EXIT ORDER}$$
    $$\text{Transition State} \implies \text{CLOSED\_EOD\_FORCE}$$
  - **Example**: At 15:15 IST, position is in profit $+1.8\%$, but Target 2 has not been hit. The system liquidates the remaining position at market VWAP to avoid overnight margin penalties or unexpected gap risk.

- [ ] **17.5 Intraday Margin Leverage & Risk-of-Ruin Gating**
  - **Description**: Brokers offer up to $5\times$ intraday leverage on equity MIS trades. Miscalculating leverage leads to rapid account depletion.
  - **Algorithm**:
    $$\text{Intraday Risk Cap} = 0.50\% \times \text{Total Account Equity}$$
    $$\text{Max Intraday Leverage Used} \le 3.0\times \text{ Account Equity}$$
  - **Example**: Account has ₹100,000. Maximum capital risked on any intraday trade is ₹500 (0.50%). Total position value cannot exceed ₹300,000 even if the broker permits ₹500,000.

---

## Module 18: High-Performance Vectorized Backtest & IndicatorStore Engine

Running 10-year backtests across 500 stocks with 55+ technical indicators can take hours if written in slow Python loops. Smart Advice uses vectorized pre-computation and O(1) in-memory stores.

- [ ] **18.1 Vectorized Signal Generation vs Iterative Loops**
  - **Description**: Iterative row-by-row DataFrame manipulation in Python causes massive overhead. Vectorized operations compute millions of data points simultaneously using C-optimized NumPy/vectorbt arrays.
  - **Algorithm**:
    Instead of looping:
    $$\text{for } t \text{ in range}(T): \text{if } C[t] > SMA[t] \dots$$
    Use column-wise boolean matrices:
    $$\mathbf{M}_{\text{Trend}} = (\mathbf{C} > \mathbf{SMA}_{50}) \ \& \ (\mathbf{SMA}_{50} > \mathbf{SMA}_{150}) \ \& \ (\mathbf{SMA}_{150} > \mathbf{SMA}_{200})$$
    Where $\mathbf{C}$ is a 2D matrix of shape $(\text{Dates} \times \text{Symbols})$.
  - **Example**: Computing 10 years of daily indicators across 100 stocks takes 28 seconds using vectorized array slicing versus 14 minutes using row-by-row iteration.

- [ ] **18.2 IndicatorStore O(1) Matrix Lookup Architecture**
  - **Description**: Before running simulation steps, all indicator arrays are computed once and stored in contiguous memory. During trade simulation, accessing any indicator value requires exactly one pointer lookup ($O(1)$ complexity) with zero TA-Lib computation during simulation runtime.
  - **Algorithm**:
    $$\text{IndicatorStore}: (\text{Symbol}, \text{IndicatorName}, \text{Timestamp}) \longrightarrow \text{Value} \in \mathbb{R}$$
    $$\text{Access Time} = O(1)$$
  - **Example**: When the simulation evaluates whether stock SBIN passed the ADX gate on 2021-08-14, it retrieves the pre-calculated float from memory rather than calculating the 14-period ADX on the fly.

- [ ] **18.3 Pre-filtering Matrix & Fast Boolean Masking**
  - **Description**: High-throughput stock screener pre-computes an entire $(\text{Dates} \times \text{Symbols})$ boolean filter matrix $\mathbf{P}$.
  - **Algorithm**:
    $$\mathbf{P}_{t, s} = \mathbb{I}(\text{Volume}_{t, s} > 150k) \land \mathbb{I}(\text{Price}_{t, s} \ge 50) \land \mathbb{I}(RSI_{t, s} \in [50, 68]) \dots$$
    $$\text{Eligible Candidates at Day } T = \{ s \mid \mathbf{P}_{T, s} == \text{True} \}$$
  - **Example**: On any historical simulation day, candidate selection reduces to fetching the True columns in row $T$, pruning 500 stocks down to the top 8 candidates in under 2 milliseconds.

- [ ] **18.4 Memory Optimization & Float32 Serialization**
  - **Description**: Storing 10 years of multi-timeframe indicator matrices for 1,000 tickers can exhaust RAM if double-precision 64-bit floats are used indiscriminately.
  - **Algorithm**: Downcast numerical matrices to `float32` and `uint32`:
    $$\text{RAM Footprint Reduction} = 50\%$$
    Serialize pre-computed matrices into Apache Parquet with Snappy compression for instantaneous disk-to-RAM streaming.
  - **Example**: Storing 55 indicator matrices in memory drops from 4.2 GB down to 1.6 GB, allowing the entire backtest validation pipeline to run concurrently on local multi-core CPU architectures.

---

## Module 19: Machine Learning Models & Statistical Regime Switching

Beyond static indicators, advanced quantitative systems incorporate machine learning to discover hidden non-linear states and dynamic probabilities.

- [ ] **19.1 Hidden Markov Models (HMM) for Unsupervised Regime Discovery**
  - **Description**: Markets transition between latent (unobservable) regimes: Low Volatility Bull, High Volatility Bull, Low Volatility Bear, and High Volatility Crash. An HMM models these as hidden states with transition probability matrices.
  - **Algorithm**:
    Let state $S_t \in \{1, 2, 3, 4\}$. Observations $\mathbf{O}_t = [\text{Return}_t, \text{Realized Volatility}_t]$.
    $$\text{Transition Matrix } \mathbf{A} = [a_{ij}], \quad a_{ij} = P(S_{t+1} = j \mid S_t = i)$$
    $$\text{Emission Probability } B = [b_j(\mathbf{O}_t)], \quad b_j(\mathbf{O}_t) = \mathcal{N}(\mathbf{O}_t; \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)$$
    Estimate latent state sequence using the Viterbi Algorithm:
    $$\widehat{S}_{1:T} = \arg\max_{S_{1:T}} P(S_{1:T} \mid \mathbf{O}_{1:T})$$
  - **Example**: When the HMM detects a transition from State 1 (Low Vol Bull) to State 4 (High Vol Bear), the recommendation engine automatically sets all risk multipliers to zero, front-running traditional moving average lag.

- [ ] **19.2 Deep Sequence Models (LSTM / GRU) for Next-Day Direction Probability**
  - **Description**: Long Short-Term Memory networks capture long-range temporal dependencies across sequential OHLCV and volume bars.
  - **Algorithm**:
    $$\mathbf{h}_t, \mathbf{c}_t = \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1}, \mathbf{c}_{t-1})$$
    $$\widehat{y}_t = \sigma(\mathbf{W}_o \mathbf{h}_t + b_o) \in [0, 1]$$
    Where $\mathbf{x}_t$ is the normalized feature vector (returns, volume ratio, ATR normalized, distance to 50 SMA).
    $$\text{ML Confidence Bonus} \iff \widehat{y}_t \ge 0.65$$
  - **Example**: When the technical indicators and entry pattern trigger a buy signal, if the LSTM sequence model predicts probability $\widehat{y} = 0.74$, the conviction score receives an algorithmic boost.

- [ ] **19.3 Reinforcement Learning (PPO / Actor-Critic) for Exit Optimization**
  - **Description**: Fixed exit targets can exit too early during runaway bull trends. RL agents learn policy $\pi(a \mid s)$ to maximize long-term cumulative reward (Sortino Ratio).
  - **Algorithm**:
    $$\text{Action Space } a_t \in \{\text{HOLD}, \text{SELL\_T1}, \text{TIGHTEN\_TRAIL}, \text{LIQUIDATE}\}$$
    $$\text{Reward Function } R_t = \Delta \text{PnL}_t - \lambda (\text{Drawdown}_t)^2$$
    Optimize policy using Proximal Policy Optimization (PPO) clipped surrogate objective:
    $$L^{\text{CLIP}}(\theta) = \widehat{\mathbb{E}}_t \left[ \min(r_t(\theta)\widehat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\widehat{A}_t) \right]$$
  - **Example**: The RL agent observes that market breadth is expanding and historical momentum is in the 95th percentile; instead of liquidating at Target 1, it decides to `TIGHTEN_TRAIL` to let the winning trade compound.

- [ ] **19.4 Feature Stationarity & Fractional Differentiation**
  - **Description**: Raw prices are non-stationary ($I(1)$ processes), which invalidates machine learning assumptions. Simple returns are stationary ($I(0)$), but erase all long-term price memory. Fractional differentiation preserves memory while achieving stationarity.
  - **Algorithm**:
    $$(1 - B)^d = \sum_{k=0}^{\infty} (-1)^k \binom{d}{k} B^k = 1 - dB + \frac{d(d-1)}{2!} B^2 - \frac{d(d-1)(d-2)}{3!} B^3 + \dots$$
    Find minimum $d \in (0, 1)$ such that the Augmented Dickey-Fuller (ADF) test $p$-value $< 0.05$.
  - **Example**: At $d = 0.42$, the fractionally differentiated price series passes the stationarity test while retaining a correlation of $0.88$ with the original price series, providing optimal predictive input features.

---

## Module 20: Quantitative Biases, Overfitting Traps & Edge Preservation

More capital is lost to quantitative fallacies and cognitive blindspots than to coding errors.

- [ ] **20.1 Survivorship Bias in NSE Universes**
  - **Description**: Testing only current Nifty 500 constituents across 2014–2024 introduces severe positive bias because failed or bankrupt companies that dropped out (e.g., DHFL, Yes Bank, Reliance Comm) are omitted.
  - **Algorithm**:
    $$\text{Correct Testing Universe at Time } T = \mathcal{U}_T \equiv \{ \text{Constituents officially listed on Date } T \}$$
    $$\text{Bias Elimination}: \text{Never use } \mathcal{U}_{2026} \text{ retroactively for } T = 2016$$
  - **Example**: A strategy tested on today's survivors shows 32% CAGR. When re-tested with point-in-time historical constituent memberships including delisted and collapsed stocks, actual CAGR corrects to a realistic 18.5%.

- [ ] **20.2 P-Hacking, Overfitting & The Multiple Testing Fallacy**
  - **Description**: If a quant tests 200 different indicator combinations on the same historical dataset, several will show phenomenal Sharpe ratios purely by statistical coincidence ($p < 0.05 \implies 1 \text{ in } 20 \text{ tests succeeds by luck}$).
  - **Algorithm**: Apply the **Bonferroni / False Discovery Rate (FDR) Correction**:
    $$\alpha_{\text{adjusted}} = \frac{\alpha}{N_{\text{trials}}}$$
    For 100 tested variations with nominal $\alpha = 0.05$:
    $$\alpha_{\text{adjusted}} = \frac{0.05}{100} = 0.0005$$
    Any strategy variation must produce $p < 0.0005$ to be acknowledged as a true edge.
  - **Example**: Tuning the RSI period across every integer from 2 to 50 produces one "optimal" backtest at period 17. The Deflated Sharpe Ratio exposes this as data snooping; the engine rejects the overfitted parameter.

- [ ] **20.3 Execution Friction Drag (The Turnover Trap)**
  - **Description**: High-turnover strategies that trade daily generate significant trading volume, generating massive STT and exchange fees that bleed capital over time.
  - **Algorithm**:
    $$\text{Net Expectancy} = (\text{Win Rate} \times \text{Avg Win}) - (\text{Loss Rate} \times \text{Avg Loss}) - \text{Total Round-Trip Frictions}$$
    $$\text{Gating}: \text{Avg Win \%} \ge 3 \times \text{Round-Trip Friction \%}$$
    Where round-trip friction on NSE delivery is $\approx 0.35\%$.
  - **Example**: If average winning trade is $+1.5\%$ and average loss is $-1.2\%$, after subtracting $0.35\%$ friction per trade, net profitability drops to zero or turns negative over 100 trades. Smart Advice swing targets ($+7.0\%$ to $+15.0\%$) ensure friction is less than 5% of gross profit.

- [ ] **20.4 Asynchronous Corporate Actions & Suspension Safeguards**
  - **Description**: Equities facing regulatory trading suspensions (e.g., ASM/GSM stages, circuit reductions to 2%) lock capital and prevent timely stop-loss execution.
  - **Algorithm**:
    $$\text{Regulatory Filter}: \text{If Stock is in ASM/GSM Stage } \ge 2 \implies \text{EXCLUDE FROM UNIVERSE}$$
    $$\text{Circuit Ceiling}: \text{If Daily Price Band} \le 5.0\% \implies \text{EXCLUDE FROM SWING BUY}$$
  - **Example**: A mid-cap stock triggers a VCP breakout, but its daily circuit band has been reduced by SEBI to 5% due to speculative volatility. The engine rejects the recommendation because a 5% limit impedes smooth ATR-based trailing exits.

---

## Module 21: Comprehensive Multi-Scenario Numerical Walkthroughs

Four comprehensive numerical walkthroughs demonstrating exactly how the math operates across diverse market environments:

### Walkthrough 2: Bear Market Defense & Circuit Breaker Trigger

- **Market Environment**: Sudden broad market correction (similar to Jan-Feb 2022 rate hike sell-off).
- **Portfolio State**:
  - Initial Capital: ₹100,000
  - Peak Portfolio Value achieved: ₹115,000
  - Current Portfolio Value: ₹101,000
- **Step 1: Drawdown Calculation**:
  $$\text{Drawdown} = \frac{115,000 - 101,000}{115,000} \times 100\% = \frac{14,000}{115,000} \times 100\% = \mathbf{12.17\%}$$
- **Step 2: Circuit Breaker Evaluation**:
  $$\text{Pause Threshold} = 12.0\%$$
  Since $12.17\% \ge 12.0\%$, the **Drawdown Circuit Breaker trips**.
- **Step 3: Algorithmic Actions**:
  1. `pause_buying = TRUE`: All candidate scanning is suspended immediately.
  2. Active open positions have their trailing stop multipliers tightened from $2.8\times$ ATR to $2.0\times$ ATR.
  3. Minimum pause duration set to 7 calendar days.
  4. Resume condition evaluated daily: portfolio must recover below $8.0\%$ drawdown AND 15 cooldown bars must elapse.
- **Outcome**: The algorithm preserves ₹101,000 cash, preventing an emotional trader from "buying the dip" into an ongoing -25% bear market collapse.

### Walkthrough 3: Minervini VCP Breakout with Staged Execution & Pyramiding

- **Stock**: `TRENT.NS`
- **Current Close**: ₹7,200.00
- **ATR (14)**: ₹140.00
- **Account Equity**: ₹200,000
- **Step 1: Pattern Qualification**:
  - Consolidation span: 32 days.
  - Contraction 1: High ₹7,300 to Low ₹6,350 (-13.0%).
  - Contraction 2: High ₹7,100 to Low ₹6,675 (-6.0%).
  - Contraction 3: High ₹7,050 to Low ₹6,880 (-2.4%).
  - Volume on contraction 3: 42% of 50-day average (extreme dry-up).
  - Breakout Candle: Close = ₹7,200 (surpassing ₹7,100 resistance) with volume $2.4\times$ 50-day average.
  - VCP Pattern = **VALIDATED**.
- **Step 2: Risk Management & Initial Sizing**:
  - Risk Budget: $1.5\% \times 200,000 = ₹3,000$.
  - Stop Loss Price $= 7,200 - (140 \times 2.4) = 7,200 - 336 = \mathbf{₹6,864.00}$.
  - Risk Per Share $= ₹336.00$ ($4.67\%$ loss risk).
  - $\text{Size}_{\text{risk}} = \lfloor 3,000 / 336 \rfloor = 8 \text{ shares}$.
  - Capital Cap ($18\%$ of ₹200,000 = ₹36,000): $\lfloor 36,000 / 7,200 \rfloor = 5 \text{ shares}$.
  - $\text{Initial Order Size} = \min(8, 5) = \mathbf{5 \text{ shares}}$.
  - Initial Capital Deployed: $5 \times 7,200 = ₹36,000$.
  - Total Risk at Inception: $5 \times 336 = ₹1,680$ (only 0.84% of portfolio equity).
- **Step 3: First Pyramid Add (+1.5 ATR)**:
  - Add Trigger Price $= 7,200 + (1.5 \times 140) = \mathbf{₹7,410.00}$.
  - Price reaches ₹7,410 on Day 4.
  - Add Size $= 50\% \times 5 = \lfloor 2.5 \rfloor \to 2 \text{ shares}$ at ₹7,410.
  - Total Shares $= 5 + 2 = 7 \text{ shares}$.
  - Blended Cost Basis $= \frac{(5 \times 7,200) + (2 \times 7,410)}{7} = \frac{36,000 + 14,820}{7} = \mathbf{₹7,260.00}$.
  - Ratchet Stop Loss to: $\mathbf{₹7,200.00}$ (initial entry price).
  - New Net Risk: $7 \times (7,260 - 7,200) = \mathbf{₹420.00}$.
  - Open position risk has dropped from ₹1,680 down to ₹420 while total position increased from 5 to 7 shares.
- **Step 4: Target 1 Execution (+3.5 ATR)**:
  - Target 1 Price $= 7,200 + (3.5 \times 140) = \mathbf{₹7,690.00}$.
  - Price touches ₹7,690 on Day 9.
  - Sell 30% of total 7 shares $\to \lfloor 7 \times 0.30 \rfloor = 2 \text{ shares}$ sold at ₹7,690.
  - Realized Profit on 2 shares $= 2 \times (7,690 - 7,260) = +₹860.00$.
  - Remaining Position: 5 shares.
  - Stop loss ratcheted to Breakeven (+0.2% buffer): $₹7,260 \times 1.002 = \mathbf{₹7,275.00}$.
- **Step 5: Trailing to Target 2 (+7.0 ATR)**:
  - Target 2 Price $= 7,200 + (7.0 \times 140) = \mathbf{₹8,180.00}$.
  - Remaining 5 shares held with trailing stop.
  - On Day 22, stock touches ₹8,190. All remaining 5 shares exited at Target 2.
  - Realized Profit on 5 shares $= 5 \times (8,180 - 7,260) = +₹4,600.00$.
  - Total Net Trade Profit $= 860 + 4,600 = \mathbf{+₹5,460.00}$.
  - Net Gain on Account Equity: $\mathbf{+2.73\%}$ achieved with maximum downside risk capped at ₹1,680 ($0.84\%$).

### Walkthrough 4: Options Chain PCR Bullish Confirmation

- **Stock**: `INFY.NS`
- **Spot Price**: ₹1,850.00
- **Technical Signals**: 50 SMA bounce, RSI $= 54.0$, Pullback to EMA passed.
- **Options Chain Data Collection**:
  - Current Monthly Expiry:
    - Strike 1800: Call OI = 250,000 | Put OI = 1,800,000 (Massive Put Support)
    - Strike 1850: Call OI = 1,100,000 | Put OI = 1,450,000
    - Strike 1900: Call OI = 2,200,000 | Put OI = 400,000 (Major Call Resistance)
    - Sum of All Strike Put OI $= 12,800,000$ contracts.
    - Sum of All Strike Call OI $= 10,240,000$ contracts.
- **Step 1: Put-Call Ratio Calculation**:
  $$PCR_{OI} = \frac{12,800,000}{10,240,000} = \mathbf{1.25}$$
  Threshold: $PCR \ge 1.00 \implies$ **BULLISH OI ALIGNMENT (PASS)**.
- **Step 2: Strike Max Pain Calculation**:
  - Computing cumulative option writer payout across all strikes:
  $$\text{Loss}(1850) = ₹420 \text{ Lakhs}$$
  $$\text{Loss}(1880) = ₹310 \text{ Lakhs} \quad (\text{Minimum Payout})$$
  $$\text{Loss}(1900) = ₹540 \text{ Lakhs}$$
  - $\text{Max Pain Strike} = \mathbf{₹1,880.00}$.
  - Since Spot Price (₹1,850) < Max Pain (₹1,880), the gravitational pull from option writing institutions favors an upward drift into expiry.
- **Step 3: Synthesis**:
  - Technical Score boosted by $+0.15$ (Options weight).
  - Overall conviction qualifies INFY for an institutional-grade swing recommendation.

---

## 🎯 Final Operational Mastery Checklist

Review this section to ensure 100% conceptual and algorithmic fluency:

- [ ] Can you explain why we use the 200 SMA on Nifty 50 rather than predicting bottoms with indicators?
- [ ] Do you understand why the Volatility Gate rejects stocks whose ATR percentile is $> 90\%$?
- [ ] Can you differentiate between volume on an intraday basis vs Delivery Volume on the NSE?
- [ ] Do you know why a stock that meets technical criteria is vetoed if its Fundamental Score is $< 0.10$?
- [ ] Can you manually calculate the two-constraint position sizing in under 30 seconds for any stock and account balance?
- [ ] Can you explain the difference between In-Sample Deflated Sharpe Ratio (DSR) and Walk-Forward Efficiency (WFE)?
- [ ] Can you detail how pyramiding ratchets the blended stop so that total account risk remains static or shrinks?
- [ ] Do you know how the Replay Engine prevents lookahead bias during 10-year historical backtests?
- [ ] Can you describe the 4 latent market regimes identified by a Hidden Markov Model (HMM)?
- [ ] Can you explain how the IndicatorStore enables $O(1)$ indicator lookups during 10-year vectorbt simulations?
- [ ] Do you know how the Bonferroni correction prevents p-hacking when testing 100 indicator variations?
- [ ] Can you calculate Anchored VWAP and its upper/lower standard deviation bands for intraday setups?

*Congratulations! Completing and mastering every topic and algorithm in this document provides full institutional-grade proficiency in the entire Smart Advice quantitative recommendation architecture.*

