# Skill: Entry Pattern Optimization

Guidelines for optimizing entry patterns and swing trading gates.

## When to Use This Skill
- Strategy has low win rate (< 40%)
- Too few entry signals (< 100 trades in 10 years)
- Too many entry signals (> 5000 trades in 10 years)
- Need to add new entry patterns
- Need to tune existing gate thresholds

## Entry Patterns Overview

Current patterns in `backend/scripts/swing_trading_signals.py`:

| Pattern | Description | Best For |
|---------|-------------|----------|
| pullback_to_ema | Price near EMA with RSI in range | Trending markets |
| bollinger_squeeze_breakout | Bandwidth squeeze + upper band break | Volatility expansion |
| macd_zero_cross | MACD crosses from below to above zero | Momentum shifts |
| higher_low_structure | Rising swing lows | Uptrend confirmation |
| volatility_contraction | Decreasing ATR over last 5 days | Breakout setup |
| nr7_volatility_squeeze | Narrowest range in 7 days + volume dry-up | Consolidation break |
| twenty_day_high_breakout | Breakout of 20-day high with volume | Momentum entries |

## Optimization Workflow

### 1. Analyze Current Pattern Performance
```bash
cd backend
python3 << 'EOF'
from database import get_mongodb
db = get_mongodb()

# Get trades from latest session
session = db.backtest_sessions.find_one(
    {'status': 'completed'},
    sort=[('completed_at', -1)]
)
if session:
    session_id = session['_id']
    trades = list(db.portfolio_backtest_trades.find(
        {'session_id': session_id}
    ))
    
    # Group by entry pattern
    from collections import defaultdict
    pattern_stats = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_pnl': 0})
    
    for t in trades:
        pattern = t.get('entry_pattern', 'unknown')
        pnl = t.get('pnl_pct', 0)
        if pnl > 0:
            pattern_stats[pattern]['wins'] += 1
        else:
            pattern_stats[pattern]['losses'] += 1
        pattern_stats[pattern]['total_pnl'] += pnl
    
    print("Pattern Performance:")
    for pattern, stats in sorted(pattern_stats.items()):
        total = stats['wins'] + stats['losses']
        win_rate = (stats['wins'] / total * 100) if total > 0 else 0
        avg_pnl = stats['total_pnl'] / total if total > 0 else 0
        print(f"  {pattern}: {total} trades, {win_rate:.1f}% win, {avg_pnl:.2f}% avg PnL")
EOF
```

### 2. Identify Weak Patterns
**Disable patterns with:**
- Win rate < 30%
- Avg PnL < -2%
- Total trades < 20 (insufficient data)

**How to disable:**
In strategy JSON (`backend/strategies/swing_trading.json`):
```json
{
  "entry_patterns": {
    "pullback_to_ema": {
      "enabled": false,  // Disable weak pattern
      ...
    }
  }
}
```

### 3. Tune Gate Thresholds

**TREND_GATE:**
```json
{
  "TREND_GATE": {
    "adx_threshold": 25,  // Increase for stronger trends (25-35)
    "di_alignment": true,  // Require +DI > -DI
    "price_above_sma_stack": true  // Price > SMA50 > SMA150 > SMA200
  }
}
```
- If too few signals: Decrease `adx_threshold` to 20
- If low win rate: Increase `adx_threshold` to 30

**VOLATILITY_GATE:**
```json
{
  "VOLATILITY_GATE": {
    "atr_percentile_max": 30  // ATR in bottom 30% of 100-day lookback
  }
}
```
- If too few signals: Increase to 40-50
- If entries during high volatility: Decrease to 20-25

**VOLUME_GATE:**
```json
{
  "VOLUME_GATE": {
    "volume_sma_ratio": 0.8,  // Volume >= 80% of 20-day avg
    "obv_trend_positive": true  // Positive OBV slope
  }
}
```
- If missing accumulation: Increase `volume_sma_ratio` to 1.0
- If too strict: Decrease to 0.6-0.7

**MTF_GATE:**
```json
{
  "MTF_GATE": {
    "weekly_trend_confirmation": true  // Weekly trend must align
  }
}
```
- If too few signals: Disable (set to false)
- If entries against weekly trend: Keep enabled

### 4. Add New Entry Patterns

**Step 1:** Add pattern definition to strategy JSON:
```json
{
  "entry_patterns": {
    "rsi_divergence": {
      "enabled": true,
      "rsi_oversold": 35,
      "price_lower_low": true,
      "rsi_higher_low": true,
      "volume_confirmation": true
    }
  }
}
```

**Step 2:** Implement pattern logic in `backend/scripts/swing_trading_signals.py`:
```python
def _check_rsi_divergence(self, df, config):
    """Check for bullish RSI divergence."""
    rsi = self._get_indicator(df, 'RSI', config.get('rsi_period', 14))
    
    # Find price lower low
    price_lower_low = df['Low'].iloc[-1] < df['Low'].iloc[-5:-1].min()
    
    # Find RSI higher low (divergence)
    rsi_higher_low = rsi.iloc[-1] > rsi.iloc[-5:-1].min()
    
    # Check oversold condition
    rsi_oversold = rsi.iloc[-1] < config.get('rsi_oversold', 35)
    
    return price_lower_low and rsi_higher_low and rsi_oversold
```

**Step 3:** Register pattern in `analyze_swing_opportunity` method:
```python
if 'rsi_divergence' in entry_patterns and entry_patterns['rsi_divergence'].get('enabled'):
    if self._check_rsi_divergence(df, entry_patterns['rsi_divergence']):
        signals.append('rsi_divergence')
```

**Step 4:** Test with backtest:
```bash
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 60
```

### 5. Pattern Combination Strategy

**High-Probability Setups (use 2-3 patterns together):**
1. **Trend Continuation:**
   - pullback_to_ema + higher_low_structure + volatility_contraction
   
2. **Breakout Setup:**
   - bollinger_squeeze_breakout + nr7_volatility_squeeze + twenty_day_high_breakout

3. **Momentum Reversal:**
   - macd_zero_cross + rsi_divergence + volume_confirmation

**In strategy JSON:**
```json
{
  "entry_patterns": {
    "required_patterns": 2,  // Require at least 2 patterns to trigger
    "pullback_to_ema": {"enabled": true},
    "higher_low_structure": {"enabled": true},
    "volatility_contraction": {"enabled": true}
  }
}
```

## Parameter Tuning Guidelines

### ATR-Based Parameters
- **Entry zone**: 1-2x ATR from EMA (closer = better entries)
- **Stop loss**: 2-3x ATR (tighter = more stops, looser = larger losses)
- **Target 1**: 3-4x ATR (achievable profit)
- **Target 2**: 5-6x ATR (let winners run)
- **Trailing stop**: 1.5-2x ATR (lock in profits)

### RSI Parameters
- **Oversold**: 30-40 (for entries)
- **Overbought**: 60-70 (for exits)
- **Period**: 14 (standard), 7 (faster), 21 (slower)

### Moving Average Parameters
- **Fast**: 20-50 EMA (trend direction)
- **Medium**: 100-150 SMA (trend confirmation)
- **Slow**: 200 SMA (long-term trend)

## Backtesting Parameter Changes

Always test parameter changes with:
1. **Quick test** (60 months, no walk-forward):
   ```bash
   python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 60
   ```

2. **Compare metrics** before/after:
   - CAGR should improve or stay similar
   - Max DD should not increase significantly
   - Win rate should be in realistic range (35-55%)

3. **Full validation** if quick test looks good:
   ```bash
   python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --mc-iterations 8
   ```

## Common Mistakes

**1. Over-optimization (Curve Fitting)**
- Symptom: Works perfectly on historical data, fails walk-forward
- Solution: Use simple, robust parameters; test on multiple time periods

**2. Too Many Patterns**
- Symptom: Conflicting signals, analysis paralysis
- Solution: Limit to 3-5 high-quality patterns

**3. Ignoring Market Regime**
- Symptom: Pattern works in bull markets, fails in bear
- Solution: Add regime-specific logic or disable pattern in unfavorable regimes

**4. Unrealistic Entry Assumptions**
- Symptom: Backtest assumes perfect entry at signal price
- Solution: Add slippage (0.1-0.2%), use next-day open for entries

## Success Metrics

After optimization, strategy should have:
- Win rate: 40-55%
- Profit factor: > 2.0
- Avg win / Avg loss: > 1.5
- Total trades: > 500 (10 years)
- Max consecutive losses: < 10
