# Swing Trading Playbook
## Strategic Guide for High-Precision Trading

### Version: 2.0
### Last Updated: August 15, 2024
### Target Audience: Traders, Portfolio Managers, Risk Managers

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Market Philosophy](#market-philosophy)
3. [Entry Strategy](#entry-strategy)
4. [Exit Strategy](#exit-strategy)
5. [Position Sizing](#position-sizing)
6. [Risk Management](#risk-management)
7. [Monitoring & Execution](#monitoring--execution)
8. [Performance Metrics](#performance-metrics)
9. [Troubleshooting](#troubleshooting)
10. [Appendices](#appendices)

---

## Executive Summary

### Core Objectives
- **Target Precision**: 65-75% hit rate at 10-20 day horizons
- **Risk-Reward**: Minimum 2.5:1 reward-to-risk ratio
- **Portfolio Risk**: Maximum 1% risk per trade, 3% daily loss limit
- **Time Horizon**: 5-25 trading days per position

### Key Success Factors
1. **All gates must pass** before entry consideration
2. **Volume confirmation** is mandatory for all signals
3. **Multi-timeframe alignment** prevents counter-trend trades
4. **Strict position sizing** maintains consistent risk
5. **Disciplined exits** preserve capital and lock in profits

---

## Market Philosophy

### Precision Over Volume
- Generate fewer, higher-quality signals
- Avoid market noise and false breakouts
- Focus on stocks with strong institutional backing
- Trade with the prevailing trend, not against it

### Risk-First Approach
- Capital preservation is paramount
- Every trade has a predefined stop-loss
- Position sizes are calculated based on risk, not conviction
- Portfolio-level risk controls override individual trade signals

### Evidence-Based Decisions
- All entries must pass quantitative gate checks
- Performance is measured against specific metrics
- Continuous monitoring and adjustment based on data
- Regular backtesting validation of strategy parameters

---

## Entry Strategy

### Gate System Overview

All four gates must pass before considering an entry:

#### 1. Trend Filter Gate
**Purpose**: Ensure we're trading with the primary trend
**Requirements**:
- ADX(14) > 20 (trending market, not sideways)
- Price above 200-day SMA (long-term uptrend)
- Price above 20-day SMA (short-term momentum)

**Implementation**:
```python
def trend_filter_check(df):
    adx = calculate_adx(df['High'], df['Low'], df['Close'], period=14)
    sma_200 = df['Close'].rolling(200).mean()
    sma_20 = df['Close'].rolling(20).mean()
    current_price = df['Close'].iloc[-1]
    
    return (
        adx.iloc[-1] > 20 and
        current_price > sma_200.iloc[-1] and
        current_price > sma_20.iloc[-1]
    )
```

#### 2. Volatility Gate
**Purpose**: Avoid extreme volatility that leads to whipsaws
**Requirements**:
- ATR(14) percentile between 20th-80th percentile of last 100 days
- Avoid both extremely low volatility (compression) and high volatility (chaos)

**Rationale**: 
- Low volatility often leads to explosive moves but with poor timing
- High volatility creates excessive noise and wider stop-losses

#### 3. Volume Confirmation Gate
**Purpose**: Ensure institutional participation and liquidity
**Requirements** (any one of):
- OBV trending upward over last 20 days
- Volume Z-score > 1.0 on breakout day
- Volume above 20-day average with price strength

**Critical Importance**: Volume validates price moves and reduces false signals

#### 4. Multi-Timeframe Confirmation Gate
**Purpose**: Prevent counter-trend trades
**Requirements**:
- Weekly 20/50 SMA alignment bullish
- Daily trend aligned with weekly trend
- Weekly RSI > 50 (weekly momentum positive)

### Entry Patterns

Once all gates pass, look for specific entry patterns:

#### Pattern 1: Pullback to Rising 20 EMA
**Setup**:
- 20 EMA trending upward
- Price pulls back to within 2% of 20 EMA
- RSI(14) between 40-60 (not oversold, but reset)
- Bullish reversal candle (close > open)

**Entry**: Break above pullback high
**Stop**: Recent swing low or 1.5x ATR below entry

#### Pattern 2: Bollinger Band Squeeze Breakout
**Setup**:
- Bollinger Band width in bottom 25% of 20-day range (squeeze)
- Volume declining during squeeze
- Price breaks above upper Bollinger Band
- Volume expansion on breakout

**Entry**: Close above upper band with volume confirmation
**Stop**: Lower Bollinger Band or 2x ATR, whichever is closer

#### Pattern 3: MACD Zero-Line Cross
**Setup**:
- MACD line crosses above zero line
- MACD line above signal line
- No significant overhead resistance within 5%
- Volume above average

**Entry**: Break above resistance level
**Stop**: Recent swing low

#### Pattern 4: Higher-Low Structure
**Setup**:
- Clear series of higher lows (at least 2)
- Each low held above previous low
- Price approaching previous high
- Volume pattern showing accumulation

**Entry**: Break above previous high
**Stop**: Below most recent higher low

### Entry Checklist

Before entering any trade, verify:

- [ ] All 4 gates passed (trend, volatility, volume, MTF)
- [ ] At least one entry pattern present
- [ ] Risk-reward ratio ≥ 2.5:1
- [ ] Position size calculated (max 1% account risk)
- [ ] Stop-loss level determined
- [ ] Target levels identified
- [ ] No conflicting sector/market signals
- [ ] Adequate liquidity for position size
- [ ] No major news/events due before target date

---

## Exit Strategy

### Stop-Loss Rules

#### Initial Stop-Loss Placement
1. **Swing Low Method** (preferred): 2% below recent swing low
2. **ATR Method**: 1.5x ATR below entry price
3. **Support Level Method**: 1% below identified support level

**Selection Criteria**: Use the method that provides the tightest stop while respecting market structure

#### Stop-Loss Management
- **Never move stops against you** (wider stops)
- **Move to breakeven** when position shows 1x ATR profit
- **Trail stops** using 3x ATR chandelier exit method
- **Time-based stops**: Exit after 15 bars with no progress

### Take-Profit Strategy

#### Multi-Target Approach
1. **Target 1 (TP1)**: 1x ATR from entry
   - Close 33% of position
   - Move stop-loss to breakeven
   
2. **Target 2 (TP2)**: 2.5x ATR from entry
   - Close 33% of position
   - Trail stop to 1.5x ATR from current price
   
3. **Target 3 (TP3)**: Trail remaining 34% until stopped out
   - Use 3x ATR chandelier exit
   - Or major resistance level

#### Dynamic Target Adjustment
Adjust targets based on:
- **Volatility**: Higher ATR = wider targets
- **Market Regime**: Bull market = higher targets
- **Sector Strength**: Strong sectors = extended targets
- **Time Decay**: Longer holding = tighter exits

### Exit Decision Tree

```
Position P&L Check (Daily)
├── Loss > 0.5% account value
│   └── Exit immediately (risk management)
├── Time in trade > 15 days
│   └── Evaluate: Exit if no clear catalyst
├── Profit >= TP1 (1x ATR)
│   ├── Take partial profit
│   └── Move stop to breakeven
├── Profit >= TP2 (2.5x ATR)
│   ├── Take second partial profit
│   └── Trail stop aggressively
└── Monitor for pattern breakdown
    └── Exit if entry pattern invalidated
```

---

## Position Sizing

### Core Principle
**Risk-based position sizing**: Each trade risks exactly 1% of account value, regardless of stock price or conviction level.

### Calculation Formula

```
Position Size = (Account Value × Risk %) / (Entry Price - Stop Price)

Example:
- Account Value: $100,000
- Risk per trade: 1% = $1,000
- Entry Price: $50
- Stop Price: $47
- Risk per share: $3
- Position Size: $1,000 / $3 = 333 shares
- Total Position Value: $16,650 (16.65% of account)
```

### Position Sizing Rules

1. **Maximum position size**: 20% of account value
2. **Minimum position size**: 2% of account value
3. **Maximum sector exposure**: 40% of account
4. **Maximum concurrent positions**: 5 active trades
5. **Cash reserve**: Minimum 20% in cash/equivalents

### Volatility-Based Adjustments

**High Volatility Stocks** (ATR > 4% of price):
- Reduce position size by 25%
- Use wider stops (2x ATR minimum)
- Set more conservative targets

**Low Volatility Stocks** (ATR < 1.5% of price):
- Standard position sizing
- Tighter stops acceptable
- Extended time horizons may be needed

### Portfolio-Level Controls

#### Daily Risk Limit
- Maximum daily loss: 3% of account value
- If hit, cease all new positions until next day
- Review existing positions for early exits

#### Correlation Management
- No more than 2 positions in same sector
- Monitor correlation between holdings
- Reduce position sizes if correlation > 0.7

---

## Risk Management

### Individual Trade Risk

#### Pre-Trade Risk Assessment
1. **News Risk**: Earnings, FDA approvals, lawsuits
2. **Technical Risk**: Major resistance levels, gap fills
3. **Market Risk**: Fed meetings, economic data
4. **Sector Risk**: Industry-specific events
5. **Liquidity Risk**: Average daily volume vs. position size

#### During-Trade Risk Management
- **Daily P&L monitoring**: Never let single trade exceed 2% loss
- **Position correlation**: Avoid clustered sector exposure
- **Market regime changes**: Exit if market character shifts
- **News flow monitoring**: Exit before major catalyst events

### Portfolio Risk

#### Risk Metrics Dashboard
Track daily:
- Total portfolio beta
- Sector concentration
- Average correlation between positions
- Maximum drawdown (rolling 30-day)
- Days since last 1%+ down day

#### Risk Escalation Triggers
- **Level 1** (3% daily loss): Review all positions, no new entries
- **Level 2** (5% weekly loss): Close worst performers, reduce size
- **Level 3** (10% monthly loss): Full strategy review, position reduction

### Black Swan Protection

#### Tail Risk Hedging
- Maintain 5% allocation to VIX calls or put spreads
- Use wide stop-losses during earnings season
- Reduce position sizes during high-event periods

#### Stress Testing
Monthly scenarios:
- 20% market decline over 5 days
- Sector rotation (tech to value)
- Interest rate spike
- Individual position gap down

---

## Monitoring & Execution

### Daily Routine

#### Market Open (9:30 AM - 10:30 AM)
1. **Pre-market Review**:
   - Check overnight news on holdings
   - Review futures and global markets
   - Identify potential gap trades
   
2. **Opening Hour Execution**:
   - Monitor existing positions for stops
   - Place new orders based on overnight analysis
   - Avoid trades in first 15 minutes (volatility)

#### Mid-Session (10:30 AM - 3:00 PM)
1. **Position Monitoring**:
   - Track profit targets
   - Adjust stop-losses if appropriate
   - Monitor volume patterns
   
2. **New Opportunity Scanning**:
   - Run gate filters on watchlist
   - Identify emerging patterns
   - Prepare orders for potential entries

#### Market Close (3:00 PM - 4:00 PM)
1. **Daily P&L Review**:
   - Calculate daily performance
   - Update risk metrics dashboard
   - Plan for next day

2. **After-Hours Analysis**:
   - Review earnings announcements
   - Update technical analysis
   - Prepare watchlist for tomorrow

### Weekly Review Process

#### Sunday Evening Preparation
1. **Market Analysis**:
   - Review major indices and sectors
   - Identify potential themes for the week
   - Check economic calendar
   
2. **Portfolio Review**:
   - Assess position health
   - Rebalance if necessary
   - Update risk parameters

3. **Backtesting Update**:
   - Run weekly performance analysis
   - Update strategy parameters if needed
   - Review gate effectiveness

### Technology Setup

#### Required Tools
- **Trading Platform**: Professional platform with advanced charting
- **Data Feed**: Real-time Level 2 data
- **Scanning Software**: Custom alerts for gate conditions
- **Risk Management**: Position sizing calculator
- **Backup Systems**: Secondary internet connection

#### Alert Configuration
Set alerts for:
- Gate condition changes on watchlist stocks
- Stop-loss triggers
- Target level approaches
- Volume spikes
- News events on holdings

---

## Performance Metrics

### Key Performance Indicators (KPIs)

#### Precision Metrics
- **Hit Rate**: % of trades profitable at 10-day horizon
- **Target**: 65-75%
- **Measurement**: Daily rolling calculation

#### Risk-Adjusted Returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Target**: >1.5 annually
- **Sortino Ratio**: Downside risk-adjusted returns
- **Target**: >2.0 annually

#### Risk Metrics
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Target**: <15% annually
- **Average R-Multiple**: Average win/loss ratio
- **Target**: >2.5

### Performance Tracking

#### Daily Metrics
```
Daily Performance Scorecard:
├── Total P&L: $XXX (X.XX%)
├── Winning Trades: X/X (XX%)
├── Average R-Multiple: X.X
├── Largest Winner: $XXX
├── Largest Loser: $XXX
└── Risk Exposure: XX% of account
```

#### Weekly Analysis
- Compare performance to benchmarks (SPY, sector ETFs)
- Analyze gate effectiveness
- Review trade execution quality
- Update rolling performance statistics

#### Monthly Deep Dive
- Full strategy review
- Parameter optimization
- Risk system stress testing
- Forward-looking adjustments

### Benchmark Comparisons

#### Primary Benchmarks
1. **SPY (S&P 500)**: Overall market performance
2. **Sector ETFs**: Sector-specific performance
3. **Buy-and-Hold**: Simple alternative strategy

#### Performance Attribution
- **Alpha Generation**: Return above benchmark
- **Beta Management**: Market exposure control
- **Risk Contribution**: Source of portfolio risk
- **Timing Effect**: Impact of entry/exit timing

---

## Troubleshooting

### Common Issues and Solutions

#### Low Hit Rate (<60%)
**Symptoms**: More losing trades than expected
**Diagnosis**:
- Check gate calibration (too loose?)
- Verify entry pattern quality
- Review stop-loss placement

**Solutions**:
1. Tighten gate thresholds
2. Add additional confirmation signals
3. Review backtesting parameters
4. Reduce position sizes temporarily

#### High Hit Rate (>80%) but Low Returns
**Symptoms**: Many small wins, few big wins
**Diagnosis**:
- Taking profits too early
- Stops too tight
- Not letting winners run

**Solutions**:
1. Extend target levels
2. Improve trailing stop methodology
3. Review market regime (trending vs. ranging)
4. Increase position sizes slightly

#### Excessive Risk Exposure
**Symptoms**: Daily losses > 2%, correlated positions
**Diagnosis**:
- Position sizing errors
- Sector concentration
- Market regime change

**Solutions**:
1. Recalibrate position sizing calculator
2. Implement correlation filters
3. Reduce overall exposure
4. Review gate effectiveness in current regime

#### Technical System Issues
**Symptoms**: Missed signals, delayed executions
**Solutions**:
1. Backup trading systems activated
2. Manual monitoring procedures
3. Reduced position sizes during outages
4. Post-incident analysis and improvements

### Decision Framework for Strategy Modifications

#### When to Adjust Parameters
1. **Performance degradation** > 30 days
2. **Market regime change** clearly identified
3. **Backtesting validation** supports changes
4. **Risk metrics** exceed acceptable levels

#### When NOT to Adjust
- After single bad trade or day
- During high-volatility periods
- Without statistical significance
- Under emotional stress

---

## Appendices

### Appendix A: Gate Calculation Details

#### ADX Calculation
```python
def calculate_adx(high, low, close, period=14):
    # True Range calculation
    tr = np.maximum(high - low, 
         np.maximum(abs(high - close.shift(1)), 
                   abs(low - close.shift(1))))
    
    # Directional Movement
    dm_plus = np.where((high - high.shift(1)) > (low.shift(1) - low), 
                       np.maximum(high - high.shift(1), 0), 0)
    dm_minus = np.where((low.shift(1) - low) > (high - high.shift(1)), 
                        np.maximum(low.shift(1) - low, 0), 0)
    
    # Smoothed calculations
    tr_smooth = tr.rolling(period).mean()
    dm_plus_smooth = dm_plus.rolling(period).mean()
    dm_minus_smooth = dm_minus.rolling(period).mean()
    
    # Directional Indicators
    di_plus = (dm_plus_smooth / tr_smooth) * 100
    di_minus = (dm_minus_smooth / tr_smooth) * 100
    
    # ADX
    dx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100
    adx = dx.rolling(period).mean()
    
    return adx
```

### Appendix B: Position Sizing Calculator

#### Excel Formula
```
=MIN(MAX_POSITION_SIZE, 
     (ACCOUNT_VALUE * RISK_PCT) / ABS(ENTRY_PRICE - STOP_PRICE))
```

#### Python Implementation
```python
def calculate_position_size(account_value, risk_pct, entry_price, stop_price, max_position_pct=0.20):
    risk_amount = account_value * risk_pct
    risk_per_share = abs(entry_price - stop_price)
    
    shares = risk_amount / risk_per_share
    position_value = shares * entry_price
    position_pct = position_value / account_value
    
    if position_pct > max_position_pct:
        shares = (account_value * max_position_pct) / entry_price
        position_value = shares * entry_price
    
    return int(shares), position_value
```

### Appendix C: Watchlist Criteria

#### Stock Selection Filters
- Market cap > $500M
- Average daily volume > 100,000 shares
- Price > $20 (avoid penny stocks)
- Historical data availability > 250 days
- Not in delisting process
- Delivery percentage > 30% (for Indian markets)

#### Sector Allocation Targets
- Technology: 25%
- Healthcare: 20%
- Financial: 15%
- Consumer: 15%
- Industrial: 10%
- Materials: 10%
- Other: 5%

### Appendix D: Emergency Procedures

#### Market Crash Response (>5% S&P decline)
1. Immediately close all positions showing losses
2. Reduce position sizes by 50% for new trades
3. Raise cash allocation to 40%
4. Monitor for stabilization signals
5. Resume normal operations only after 3-day stability

#### System Failure Response
1. Switch to backup trading platform
2. Convert electronic orders to phone orders
3. Reduce position sizes by 75%
4. No new positions until systems restored
5. Document all manual actions for post-incident review

---

### Document Control

**Version History**:
- v1.0 (Jan 2024): Initial playbook creation
- v1.5 (May 2024): Added risk management enhancements
- v2.0 (Aug 2024): Comprehensive update with gate system

**Review Schedule**: Monthly review, quarterly updates

**Approvals**: 
- Strategy Team: ✓
- Risk Management: ✓
- Compliance: ✓

**Next Review Date**: November 15, 2024

---

*This playbook is a living document and should be updated regularly based on market conditions, performance analysis, and regulatory changes.*
