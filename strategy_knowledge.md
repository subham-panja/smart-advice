# Smart Advice — Strategy Knowledge Base

> Complete step-by-step documentation of how the swing trading system works when you scan stocks.
> **Last verified against source code: 2026-04-24**

---

## Architecture Overview

```
run_analysis.py                    ← Entry point
  ├── check_macro_regime()         ← Step 0: Nifty 50 safety check
  ├── StockScanner.get_symbols()   ← Step 1: Fetch & pre-filter NSE symbols
  ├── Phase 1: 16 threads          ← Step 2: Fetch 5y OHLCV data
  ├── Phase 2: 8 processes         ← Step 3-6: Analysis pipeline
  │     ├── strategy_evaluator.py  ← Step 3: Run 5 strategies, score
  │     ├── swing_trading_signals.py ← Step 4: 4 swing gates
  │     ├── swing_trading_signals.py ← Step 5: Entry pattern detection
  │     ├── swing_trading_signals.py ← Step 6: ATR exit levels
  │     └── backtesting_runner.py  ← Step 7: Backtest validation
  └── Phase 3: Main process        ← Step 8: Save to MongoDB
        └── persistence_handler.py
```

---

## Step 0: Macro Gate — "Is the market safe?"

**File:** `run_analysis.py` → `check_macro_regime()`

```
You run: python run_analysis.py --max-stocks 50
```

Before touching any stock, the system checks **Nifty 50 (^NSEI)**:

- Is Nifty price **above its 20-day EMA**?
- Is Nifty **MACD bullish** (MACD line > Signal line)?

**If BOTH yes** → proceed to scan
**If either fails** → abort everything. No point swing trading in a bear market.

> On failure the system logs: "MACRO BEAR MARKET DETECTED" and halts all analysis.
> On API error the gate defaults to **PASS** (fail-safe).

### Why this matters

You don't swim against the tide. If Nifty is falling, 80% of stocks fall too. This single gate protects you from taking trades in a hostile market environment.

---

## Step 1: Fetch Stock List — "Which stocks to analyze?"

**File:** `data_fetcher.py` → `get_filtered_nse_symbols()` / `filter_active_stocks()`

The system fetches all NSE symbols and **pre-filters** them before any analysis:

| Filter | Value | Why |
|---|---|---|
| Min daily volume | 100,000 shares | Skip illiquid stocks you can't actually trade |
| Min market cap | Rs 5,000 Cr (₹50 billion) | Skip penny/micro caps prone to manipulation |
| Min price | Rs 20 | Skip sub-Rs 20 junk stocks |
| Max price | Rs 50,000 | Cap on maximum stock price |
| Active trading days | 250+ | Need enough history for SMA(200) to work |
| Min delivery % | 30% | Filter out pure speculative/intraday stocks |
| Max volatility percentile | 80% | Avoid extremely volatile stocks |

**Result:** ~500 stocks survive out of 2,000+ NSE symbols.

### Concurrency

The pre-filter stage uses **1-2 worker threads** with a small random delay between API calls to avoid Yahoo Finance rate limits. Each stock requires a live `yfinance` call to fetch market cap and company info.

### Caching

Filtered symbols are cached for **72 hours** (`FILTERED_SYMBOLS_CACHE_HOURS`). Subsequent runs within that window skip the entire pre-filter step.

### Config Reference

```python
# config.py → STOCK_FILTERING
STOCK_FILTERING = {
    'min_volume': 100000,
    'min_price': 20.0,
    'max_price': 50000.0,
    'min_market_cap': 5000000000,    # 5000 Cr in raw value
    'min_historical_days': 250,
    'volume_lookback_days': 50,
    'exclude_delisted': True,
    'exclude_suspended': True,
    'min_delivery_percent': 30.0,
    'max_volatility_percentile': 80,
}
```

---

## Step 2: Fetch Historical Data — "Get 5 years of price data"

**File:** `run_analysis.py` → Phase 1 (16 threads)

For each surviving stock, fetch **5 years of daily OHLCV** data from yfinance:

- **O** = Open price
- **H** = High price
- **L** = Low price
- **C** = Close price
- **V** = Volume

Each row = 1 trading day. ~1,250 rows per stock.

**16 threads** (`DATA_FETCH_THREADS`) run simultaneously to speed up the I/O-bound data fetching. Data is cached locally as CSV files — subsequent runs read from cache if the data is fresh (≤3 days old).

### Why 5 years?

- SMA(200) needs 200 bars to stabilize
- Backtest needs enough trades (10+) to be statistically meaningful
- Covers multiple bull/bear cycles for robust validation

---

## Step 3: Technical Scoring — "Run 5 strategies, get a score"

**File:** `strategy_evaluator.py` → runs on **8 CPU cores** in parallel (Phase 2)

For each stock, the system runs the **5 enabled strategies** on the latest data:

| # | Strategy | What it checks | Signal |
|---|---|---|---|
| 1 | **RSI** | RSI(14) crossed above 30 (oversold bounce) | +1 BUY or -1 |
| 2 | **MACD** | MACD line crossed above Signal line | +1 BUY or -1 |
| 3 | **EMA 12/26** | EMA(12) crossed above EMA(26) | +1 BUY or -1 |
| 4 | **ADX** | ADX > 20 AND +DI > -DI (strong uptrend) | +1 BUY or -1 |
| 5 | **OBV** | On-Balance Volume rising (money flowing in) | +1 BUY or -1 |

Each strategy returns `+1` (bullish) or `-1` (bearish/neutral).

> **Note:** 50+ additional strategies exist in the codebase but are **disabled** in `config.py → STRATEGY_CONFIG`. Only these 5 core signals are active for swing trading.

### Score Calculation

```
Technical Score = positive signals / total signals

Example:
  RSI = +1, MACD = +1, EMA = -1, ADX = +1, OBV = -1
  Score = 3/5 = 0.60
```

### Gate

Score must be **>= 0.40** (at least 2 of 5 agree) to continue.

```python
# config.py
MIN_RECOMMENDATION_SCORE = 0.40
```

### Why this gate?

If less than 2 strategies agree, the signal is too weak and conflicted. Skip this stock — there's no consensus among indicators.

---

## Step 4: Swing Trading Gates — "4 quality checks"

**File:** `swing_trading_signals.py` → `analyze_swing_opportunity()`

Stocks that pass Step 3 now go through **4 sequential gates**. ALL 4 must pass.

---

### Gate 1: Trend Filter — "Is this stock in an UPTREND?"

| Check | Condition | What it means |
|---|---|---|
| ADX(14) | Between 20-50 | Stock is trending, but not overextended |
| Price vs SMA(200) | Price > SMA(200) | Above long-term average = bullish |
| SMA(50) vs SMA(200) | SMA(50) > SMA(200) | Golden cross territory |
| EMA(20) vs SMA(50) | EMA(20) > SMA(50) | Short-term momentum is up |
| +DI vs -DI | +DI > -DI | Buyers are stronger than sellers |

**ALL conditions must be true → otherwise REJECT stock**

### Why Gate 1?

Swing trading only works WITH the trend. Buying pullbacks in a downtrend = catching falling knives. This gate ensures you're only trading stocks that are already moving up.

### Config Reference

```python
# config.py → SWING_TRADING_GATES
'trend_filter': {
    'enabled': True,
    'adx_period': 14,
    'adx_threshold': 20,
    'sma_period': 200,
    'price_above_sma': True,
}
```

---

### Gate 2: Multi-Timeframe Confirmation — "Does the WEEKLY chart agree?"

Daily data is resampled to weekly internally if no weekly data is provided.

| Check | Condition | What it means |
|---|---|---|
| Weekly SMA(20) vs SMA(50) | Weekly SMA(20) > SMA(50) | Weekly trend is up |
| Weekly Price vs SMA(20) | Price > Weekly SMA(20) | Price above weekly average |
| Weekly RSI | > 50 | Weekly momentum is bullish |
| Daily SMA(20) | Rising (higher than 5 bars ago) | Today's trend is accelerating |
| Daily Price vs SMA(20) | Price > Daily SMA(20) | Above short-term average |
| Daily RSI | > 40 | Not deeply oversold on daily |

**Weekly AND Daily must both be bullish → otherwise REJECT**

### Why Gate 2?

If the daily chart says BUY but the weekly chart says SELL, you're fighting the bigger trend. Multi-timeframe alignment dramatically increases the probability of success.

### Config Reference

```python
# config.py → SWING_TRADING_GATES
'multi_timeframe': {
    'enabled': True,
    'weekly_trend_check': True,
    'weekly_sma_fast': 20,
    'weekly_sma_slow': 50,
}
```

---

### Gate 3: Volatility Gate — "Is volatility just right?"

```
ATR(14) percentile over last 100 bars must be between 20% and 80%
```

| ATR Percentile | Meaning | Action |
|---|---|---|
| < 20% | Stock is dead, barely moving | REJECT — no profit potential |
| 20-80% | "Goldilocks zone" — healthy movement | PASS |
| > 80% | Stock is extremely volatile | REJECT — stops get destroyed |

### Why Gate 3?

You need enough volatility to make money (can't profit from a flat stock), but not so much that your stop losses get hit by random noise before the trade can play out.

### Config Reference

```python
# config.py → SWING_TRADING_GATES
'volatility_gate': {
    'enabled': True,
    'atr_period': 14,
    'min_percentile': 20,
    'max_percentile': 80,
}
```

---

### Gate 4: Volume Confirmation — "Is money actually flowing in?"

Either of these must be true:

| Check | How it works |
|---|---|
| **OBV slope > 0** | Linear regression of On-Balance Volume over 10 bars is positive — sustained buying pressure |
| **Volume Z-score > 1.0** | Today's volume is 1 standard deviation above the 20-day average — unusual activity |

**If NEITHER → REJECT stock**

### Why Gate 4?

Price movement without volume = fake move. A stock can drift up on low volume, but it won't sustain. Volume confirmation ensures institutional/smart money is actually participating in the move.

### Config Reference

```python
# config.py → SWING_TRADING_GATES
'volume_confirmation': {
    'enabled': True,
    'obv_trend_periods': 10,
    'volume_zscore_threshold': 1.0,
    'require_either': True,  # OR logic (not AND)
}
```

---

## Step 5: Entry Pattern Detection — "When exactly to buy?"

**File:** `swing_trading_signals.py` → `detect_entry_patterns()`

Stocks that pass ALL 4 gates get checked for **specific entry setups**. At least 1 of 5 must fire:

---

### Pattern 1: Pullback to Rising EMA

```
Price dipped to the 20-EMA and bounced with a green candle
```

| Condition | Check |
|---|---|
| Low touches EMA(20) | Low <= EMA(20) x 1.02 (within 2%) |
| Close above EMA | Close > EMA(20) |
| EMA is rising | EMA(20) > EMA(20) from 5 bars ago |
| RSI neutral | RSI between 40-60 (not overbought) |
| Bullish candle | Close > Open (green candle) |

**Why it works:** You're buying at a "discount" in an uptrend. Like buying on sale during a bull run. The EMA acts as dynamic support.

---

### Pattern 2: Bollinger Squeeze Breakout

```
Bands were tight (low volatility) and price just broke above the upper band
```

| Condition | Check |
|---|---|
| BB width in squeeze | Width below 5th percentile of last 20 bars |
| Price breaks out | Close > Upper Bollinger Band |
| Retest (if configured) | Low of last 3 bars touched upper band |

**Why it works:** Tight Bollinger Bands = energy building up (like a compressed spring). When price breaks the upper band, the stored energy releases upward.

---

### Pattern 3: MACD Zero-Line Crossover

```
MACD crossed above its signal line AND is above zero
```

| Condition | Check |
|---|---|
| Bullish crossover | MACD > Signal (and wasn't yesterday) |
| Above zero line | MACD > 0 (configurable via `above_zero_only`) |

**Why it works:** Zero-line crossover = momentum shifting from bearish to bullish. Being above zero means the short-term average is above the long-term average.

---

### Pattern 4: Higher-Low Structure

```
Stock is making higher lows = healthy uptrend structure
```

| Condition | Check |
|---|---|
| Swing lows detected | Pivot lows identified in last 30 bars |
| Minimum 2 swings | At least 2 swing lows found |
| Each higher | Each swing low > previous swing low |

**Why it works:** Classic uptrend structure. Buyers are stepping in at higher and higher prices, showing increasing demand.

---

### Pattern 5: Volume-Supported Resistance Breakout

```
Price broke above 20-day high with heavy volume
```

| Condition | Check |
|---|---|
| New high | Close > highest high of last 20 bars |
| Volume surge | Volume > 1.5x the 20-day average |

**Why it works:** Breaking resistance with heavy volume = institutional buying. The breakout is backed by real money, not just retail noise.

---

## Step 6: Exit Rules — "Where to place SL, TP?"

**File:** `swing_trading_signals.py` → `calculate_exit_rules()`

For every stock with an entry signal, the system calculates **exact exit levels** based on ATR(14):

### Exit Level Calculation

```
Example: Stock at Rs 1000, ATR(14) = Rs 30

Stop Loss     = Rs 1000 - (1.5 x Rs 30) = Rs 955      ← max loss per share
Take Profit 1 = Rs 1000 + (1.5 x Rs 30) = Rs 1045     ← partial exit (config: target_1_atr)
Take Profit 2 = Rs 1000 + (3.0 x Rs 30) = Rs 1090     ← full exit (config: target_2_atr)
Trailing Stop = 2.0 x Rs 30 = Rs 60 below peak         ← ratchets up (config: trail_stop_atr)
Time Stop     = 15 bars                                ← exit if nothing happens
```

> **Note:** Config comments say TP1 = "1x ATR" and TP2 = "2.5x ATR" and trail = "3x ATR", but the actual config values are `1.5`, `3.0`, and `2.0` respectively. The code uses the numeric values, not the comments.

### Exit Rules Behavior

| Event | Action |
|---|---|
| Price hits Stop Loss | Exit immediately. Accept the loss. |
| Price hits TP1 | Move Stop Loss to entry price (breakeven). Now risk-free. |
| Price hits TP2 | Full exit. Book the profit. |
| Price rises after TP1 | Trailing stop follows: always stays 2.0 x ATR below highest price |
| 15 bars pass, no TP hit | Time stop — exit. Capital is locked with no progress. |

### Why ATR-based exits (not fixed %)?

A volatile stock (ATR Rs 50) needs wider stops than a calm stock (ATR Rs 10). Fixed percentage stops (like "always use 3%") get destroyed on volatile stocks and leave money on the table on calm ones. ATR adapts automatically to each stock's personality.

### Config Reference

```python
# config.py → SWING_PATTERNS → exit_rules
'exit_rules': {
    'initial_stop_type': 'atr_based',
    'atr_stop_multiplier': 1.5,
    'target_1_atr': 1.5,
    'target_2_atr': 3.0,
    'trail_stop_atr': 2.0,
    'time_stop_bars': 15,
    'breakeven_at_target_1': True,
}
```

---

## Step 7: Backtest Validation — "Does this actually work on this stock?"

**File:** `backtesting_runner.py`

The system runs the strategy on **5 years of history** using the Backtrader engine. It simulates every trade that would have been taken and calculates metrics.

### Which strategies are backtested?

The backtester only supports a subset of strategies that have Backtrader-compatible implementations:

| Supported for backtesting |
|---|
| MA_Crossover_50_200 |
| RSI_Overbought_Oversold |
| MACD_Signal_Crossover |
| Bollinger_Band_Breakout |
| EMA_Crossover_12_26 |
| Stochastic_Overbought_Oversold |
| ADX_Trend_Strength |

> Only strategies that are **both enabled in config AND in this supported list** get backtested. Currently that means: RSI, MACD, EMA 12/26, and ADX (4 of the 5 active strategies — OBV is not in the backtest-supported list).

### Metrics Calculated

| Metric | Formula | What it tells you | Target |
|---|---|---|---|
| **Profit Factor** | Gross Profit / Gross Loss | For every Rs 1 lost, how much Rs earned | > 1.0 |
| **Expectancy** | (Win% x AvgWin) - (Loss% x AvgLoss) | Average Rs earned per trade | > Rs 0 |
| **Win Rate** | Winning Trades / Total Trades | How often you're right | 45-65% |
| **Max Drawdown** | Largest peak-to-trough equity decline | Worst-case pain | < 20% |
| **Recovery Factor** | Net Profit / Max Drawdown | How fast you bounce back from losses | > 0 |
| **CAGR** | Annualized return | Long-term growth rate | Secondary metric |
| **Sharpe Ratio** | Return / Volatility | Risk-adjusted performance | > 0 |

### Performance Rating

```python
# backtesting_runner.py → _generate_backtest_summary()

if (avg_pf > 2.0 AND win_rate > 55%) OR (avg_expectancy > 500 AND win_rate > 60%):
    rating = 'Excellent'

elif (avg_pf > 1.5 AND win_rate > 45%) OR (avg_expectancy > 200 AND win_rate > 50%):
    rating = 'Good'

elif avg_expectancy > 0 OR avg_pf > 1.0:
    rating = 'Average'

else:
    rating = 'Poor'
```

### Backtest Recommendation (from backtesting_runner.py)

```python
if (avg_expectancy > 100 AND win_rate > 50%) OR (avg_pf > 1.5 AND win_rate > 45%):
    recommendation = 'BUY'
elif avg_expectancy >= 0 OR avg_pf > 1.0:
    recommendation = 'HOLD'
else:
    recommendation = 'SELL'
```

### Final Recommendation (from analyzer.py)

The actual BUY/HOLD decision uses `analyzer.py → _calculate_combined_recommendation()` which combines all scores:

```python
# analyzer.py — simplified logic

technical_score  = strategy score (0.0 to 1.0)
fundamental_score = 0.0  (disabled for swing)
sentiment_score  = 0.0  (disabled for swing)

combined_score = technical * 1.0 + fundamental * 0.0 + sentiment * 0.0

# STRONG_BUY if:
#   - passes_gates AND technical > 0.4 AND backtest OK
#   - OR combined_score > 0.65

# BUY if:
#   - passes_gates AND combined_score >= 0.50

# HOLD otherwise (never shows SELL — system only provides BUY recommendations)
```

> **Key:** Since fundamental and sentiment are both weighted 0.0, `combined_score` equals `technical_score` in practice.

---

## Step 8: Save to MongoDB — "Store everything"

**File:** `persistence_handler.py`

The system saves data at multiple stages, not just the final result:

### What Gets Saved

| Stage | MongoDB Collection | What's stored | Saved for |
|---|---|---|---|
| Every stock analyzed | `analysis_snapshots` | Scores, strategy signals, price data, hold reasons | ALL stocks (debug why rejected) |
| Swing gate checks | `swing_gate_results` | Gate 1-4 pass/fail with actual indicator values | ALL stocks |
| Entry signal found | `trade_signals` | Entry pattern, SL, TP1, TP2, trailing stop, ATR | Only gate-passed stocks with patterns |
| Backtest completed | `backtest_results` | Profit Factor, Win Rate, Max DD, Expectancy, CAGR | Only backtested stocks |
| Final BUY signal | `recommended_shares` | Full recommendation + trade plan + backtest metrics | Only BUY/STRONG_BUY stocks |
| Scan run metadata | `scan_runs` | Config used, timing, summary counts, macro regime | Every analysis run |

---

## Funnel: How 2,000+ Stocks Become ~5 Picks

```
2,000+ NSE symbols
    |
    v  Pre-filter (volume ≥ 100K, price ≥ Rs 20, market cap ≥ 5000 Cr, delivery ≥ 30%)
  ~500 stocks
    |
    v  Step 3: Technical Score >= 0.40 (2/5 strategies agree)
  ~150 stocks
    |
    v  Gate 1: Trending (ADX 20-50, Price > SMA200, SMA50 > SMA200, EMA20 > SMA50, +DI > -DI)
  ~80 stocks
    |
    v  Gate 2: Weekly + Daily alignment (MTF confirmation)
  ~50 stocks
    |
    v  Gate 3: Volatility in 20-80% ATR percentile range
  ~35 stocks
    |
    v  Gate 4: Volume confirms (OBV rising OR Z-score > 1.0)
  ~15 stocks
    |
    v  Step 5: Entry pattern detected (1 of 5 patterns)
  ~8 stocks
    |
    v  Step 6: Exit levels calculated (SL, TP1, TP2)
  ~8 stocks
    |
    v  Step 7: Backtest validates (Profit Factor > 1.0, Win Rate > 45%)
  ~5 stocks
    |
    v  Step 8: SAVED as BUY recommendation
  * Final picks shown on frontend
```

Each step is a **filter**. You start with 2,000+ and end with ~5 high-confidence picks. That's the precision — not indicator soup, but sequential elimination.

---

## Quick Reference: Files & What They Do

| File | Role |
|---|---|
| `config.py` | All thresholds, toggles, gate parameters, strategy on/off |
| `run_analysis.py` | Pipeline orchestrator (macro gate → fetch → analyze → save) |
| `scripts/strategy_evaluator.py` | Runs 5 strategies, calculates technical score |
| `scripts/swing_trading_signals.py` | 4 gates + 5 entry patterns + exit rules |
| `scripts/trade_logic.py` | Trade plan generation (SL/TP/trailing) |
| `scripts/backtesting_runner.py` | Backtest engine + metrics (PF, Expectancy, CAGR) |
| `scripts/analyzer.py` | Combines scores → final BUY/HOLD decision |
| `utils/persistence_handler.py` | Saves to 6 MongoDB collections |
| `database.py` | MongoDB connection + collection indexes |
| `utils/stock_scanner.py` | Pre-filter and symbol fetching |
| `scripts/data_fetcher.py` | yfinance data fetching, caching, stock info |

---

## Config Quick Reference

```python
# 5 Enabled Strategies
RSI_Overbought_Oversold      # Oversold bounce filter
MACD_Signal_Crossover         # Momentum confirmation
EMA_Crossover_12_26           # Fast trend confirmation
ADX_Trend_Strength            # Trend gate filter
On_Balance_Volume             # Volume confirmation

# Weights
technical: 1.00               # 100% technical
fundamental: 0.00             # Disabled for swing
sentiment: 0.00               # Disabled for swing

# Key Thresholds
MIN_RECOMMENDATION_SCORE = 0.40    # 2/5 strategies must agree
buy_combined threshold = 0.50       # Combined score for BUY
strong_buy_combined threshold = 0.65 # Combined score for STRONG_BUY
ADX threshold = 20                  # Minimum trend strength
ADX max = 50                        # Maximum (overextended)
ATR percentile range = 20-80%       # Volatility sweet spot
Volume Z-score = 1.0               # Volume spike threshold
SL multiplier = 1.5x ATR           # Stop loss distance
TP1 = 1.5x ATR                     # First target
TP2 = 3.0x ATR                     # Second target
Trailing stop = 2.0x ATR           # Trail distance
Time stop = 15 bars                 # Max holding period

# Risk Management
Risk per trade = 1%                 # Position sizing
Max position = 20% of portfolio     # Concentration limit
Max concurrent positions = 5        # Portfolio cap
Min risk-reward = 1.5:1             # Floor ratio
```
