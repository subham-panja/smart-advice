# Skill: Risk Management Configuration

Guidelines for configuring and validating risk management rules.

## When to Use This Skill
- Setting up risk parameters for new strategies
- Max drawdown exceeds acceptable levels
- Position sizing needs adjustment
- Need to validate risk controls

## Risk Parameters in Strategy JSON

### Position Sizing
```json
{
  "risk_management": {
    "max_position_pct": 10.0,  // Max 10% of portfolio per position
    "risk_per_trade_pct": 2.0,  // Risk 2% of capital per trade
    "max_positions": 8,  // Maximum concurrent positions
    "min_position_size": 5000  // Minimum position value (Rs)
  }
}
```

**Guidelines:**
- **Conservative**: max_position_pct=5%, risk_per_trade=1%, max_positions=5
- **Moderate**: max_position_pct=10%, risk_per_trade=2%, max_positions=8
- **Aggressive**: max_position_pct=15%, risk_per_trade=3%, max_positions=10

### Stop Loss Configuration
```json
{
  "exit_rules": {
    "stop_loss": {
      "atr_multiple": 2.5,  // Stop at 2.5x ATR below entry
      "max_loss_pct": 8.0  // Maximum loss per trade (hard stop)
    }
  }
}
```

**Guidelines:**
- **Tight stops**: atr_multiple=2.0, max_loss_pct=5% (more stops, smaller losses)
- **Loose stops**: atr_multiple=3.0, max_loss_pct=10% (fewer stops, larger losses)

### Exit Targets
```json
{
  "exit_rules": {
    "targets": {
      "T1": {"atr_multiple": 3.0, "sell_pct": 50},  // Sell 50% at 3x ATR
      "T2": {"atr_multiple": 5.0, "sell_pct": 50}  // Sell remaining at 5x ATR
    },
    "trailing_stop": {
      "atr_multiple": 2.0,  // Trail at 2x ATR
      "activate_after_t1": true  // Activate after T1 hit
    },
    "breakeven": {
      "activate_after_t1": true,  // Move stop to breakeven after T1
      "buffer_pct": 0.5  // Small buffer above breakeven
    },
    "time_stop": {
      "days": 20  // Exit if no profit after 20 days
    }
  }
}
```

## Risk Validation Checklist

After running backtest, verify:
- [ ] Max drawdown < 20%
- [ ] Largest single loss < max_loss_pct
- [ ] Average position size within max_position_pct
- [ ] Max concurrent positions <= max_positions
- [ ] Time stops trigger for stagnant trades
- [ ] Trailing stops activate after T1

## Common Risk Scenarios

### Scenario 1: High Drawdown (> 25%)
**Causes:**
- Too many correlated positions
- Loose stops
- Large position sizes
- No market regime filter

**Solutions:**
1. Reduce max_positions from 8 to 5
2. Tighten stop_loss.atr_multiple from 2.5 to 2.0
3. Reduce max_position_pct from 10% to 7%
4. Add market regime detection (disable in bear markets)

### Scenario 2: Too Many Small Losses
**Causes:**
- Tight stops
- Poor entry timing
- High volatility stocks

**Solutions:**
1. Loosen stop_loss.atr_multiple from 2.0 to 2.5
2. Add volatility filter (avoid high ATR stocks)
3. Improve entry patterns (see entry_pattern_optimization skill)

### Scenario 3: Large Single Loss
**Causes:**
- Gap risk (overnight jumps)
- Circuit breaker hits
- Stop not executing

**Solutions:**
1. Reduce max_position_pct
2. Add gap risk protection (skip stocks with high gap frequency)
3. Use limit orders instead of market orders

### Scenario 4: Overconcentration
**Causes:**
- Too many positions in same sector
- Correlated stocks

**Solutions:**
1. Add sector diversification rule
2. Limit positions per sector to 2-3
3. Use correlation matrix to avoid correlated stocks

## Advanced Risk Controls

### Market Regime Filter
```json
{
  "analysis_config": {
    "market_regime_detection": true
  },
  "market_regime_config": {
    "index": "^NSEI",  // NIFTY 50
    "regime_lookback": 250,  // Days for regime classification
    "trade_in_regimes": ["BULL", "SIDEWAYS"],  // Skip BEAR/CRASH
    "position_scale": {
      "BULL": 1.0,  // Full position in bull
      "SIDEWAYS": 0.7,  // 70% position in sideways
      "BEAR": 0.3,  // 30% position in bear
      "CRASH": 0.0  // No new positions in crash
    }
  }
}
```

### Volatility-Based Position Sizing
```json
{
  "risk_management": {
    "volatility_adjustment": {
      "enabled": true,
      "atr_percentile_threshold": 70,  // Reduce size for high volatility
      "scale_factor": 0.5  // Halve position for high volatility stocks
    }
  }
}
```

### Correlation Filter
```json
{
  "risk_management": {
    "correlation_filter": {
      "enabled": true,
      "max_correlation": 0.7,  // Skip if correlation > 0.7 with existing positions
      "lookback_days": 60  // Correlation calculation period
    }
  }
}
```

## Risk Metrics to Monitor

### Portfolio Level
- **Max Drawdown**: Should be < 20%
- **Sharpe Ratio**: Should be > 0.5
- **Sortino Ratio**: Should be > 0.7 (focuses on downside risk)
- **Calmar Ratio**: CAGR / Max DD (should be > 1.0)

### Trade Level
- **Avg Win / Avg Loss**: Should be > 1.5
- **Max Consecutive Losses**: Should be < 10
- **Recovery Factor**: Time to recover from max DD (should be < 6 months)
- **Expectancy**: (Win% × Avg Win) - (Loss% × Avg Loss) (should be positive)

## Testing Risk Parameters

### Step 1: Conservative Baseline
```json
{
  "risk_management": {
    "max_position_pct": 5.0,
    "risk_per_trade_pct": 1.0,
    "max_positions": 5
  }
}
```
Run backtest and note metrics.

### Step 2: Gradual Increase
Increase one parameter at a time:
- max_position_pct: 5% → 7% → 10% → 12%
- Run backtest after each change
- Stop when Max DD > 20% or Sharpe decreases

### Step 3: Stress Test
Run stress tests to validate:
```bash
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --mc-iterations 8
```

Check:
- Regime tests: Max DD in bear/crash should be < 25%
- Cost sensitivity: Should remain profitable with 2x costs
- Parameter sensitivity: Risk params should be robust to +/-20% changes

## Emergency Controls

### Circuit Breaker
In `backend/config.py`:
```python
TRADING_OPTIONS = {
    "circuit_breaker": False,  # Set to True to halt all trading
    "max_daily_loss_pct": 3.0,  # Halt if daily loss > 3%
    "max_weekly_loss_pct": 7.0  # Halt if weekly loss > 7%
}
```

**When to activate:**
- Unexpected market crash
- Strategy bug causing excessive losses
- Broker API issues
- Data feed problems

### Manual Override
```bash
# Disable all strategies
cd backend/strategies
for f in *.json; do
    jq '.enabled = false' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done
```

## Risk Management Best Practices

1. **Never risk more than 2% per trade** (industry standard)
2. **Max 8 concurrent positions** (manageable number)
3. **Use ATR-based stops** (adapts to volatility)
4. **Scale position by regime** (reduce in unfavorable markets)
5. **Monitor correlation** (avoid overconcentration)
6. **Set hard stops** (max_loss_pct as backstop)
7. **Use time stops** (exit stagnant trades)
8. **Trail winners** (let profits run, cut losses)
9. **Diversify across sectors** (reduce sector risk)
10. **Review risk monthly** (adjust based on performance)
