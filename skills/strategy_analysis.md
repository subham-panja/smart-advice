# Skill: Strategy Analysis & Improvement

Guidelines for analyzing trading strategy performance and making data-driven improvements.

## When to Use This Skill
- Strategy has low confidence score (< 50/100)
- Win rate is below 40% or above 70% (suspicious)
- Profit factor is below 1.5
- Max drawdown exceeds 25%
- Need to compare multiple strategies

## Analysis Workflow

### 1. Check Confidence Score Components
```bash
cd backend
python3 << 'EOF'
from database import get_mongodb
db = get_mongodb()
session = db.backtest_sessions.find_one(
    {'status': 'completed'},
    sort=[('completed_at', -1)]
)
if session:
    phases = session.get('ultimate_phases', {})
    conf = phases.get('confidence_score', {})
    print(f"Total Score: {conf.get('total_score')}/100")
    print(f"Level: {conf.get('confidence_level')}")
    print(f"Realistic CAGR: {conf.get('realistic_cagr')}%")
    
    # Component breakdown
    components = conf.get('components', {})
    for k, v in components.items():
        print(f"{k}: {v.get('score')}/{v.get('max')}")
EOF
```

### 2. Identify Weak Components
**Low Walk-Forward Score (< 10/20):**
- Strategy overfits to historical data
- Solution: Simplify gates, reduce parameter count, increase walk-forward windows

**Low DSR Score (< 10/15):**
- Sharpe ratio not statistically significant
- Solution: Increase sample size (more months), check for data snooping bias

**Low Stress Test Score (< 10/15):**
- Strategy fails in certain market regimes
- Solution: Add regime-specific filters, adjust entry patterns for bear markets

**Low Parameter Stability (< 6/10):**
- Strategy sensitive to small parameter changes
- Solution: Use more robust parameters (e.g., percentile-based vs absolute values)

### 3. Analyze Trade Diagnostics
```bash
cd backend
python scripts/trade_diagnostics.py --strategy Swing_Trading
```

**Key Metrics to Check:**
- **Avg Hold Time**: If < 5 days, exits too early. If > 30 days, consider time-stop.
- **Win/Loss Distribution**: Check if losses are clustered (regime issue) or random.
- **Exit Reasons**: If most exits are "time_stop", entry timing is poor.
- **Max Concurrent Positions**: If always at max (8), filters too loose.

### 4. Strategy JSON Optimization
**Common Adjustments:**

| Problem | Solution | JSON Field |
|---------|----------|------------|
| Low win rate | Tighten entry gates | `swing_trading_gates.TREND_GATE.adx_threshold` (increase) |
| Too few trades | Loosen filters | `stock_filters.min_volume` (decrease) |
| High drawdown | Reduce position size | `risk_management.max_position_pct` (decrease) |
| Early exits | Increase targets | `exit_rules.targets.T1` (increase from 3x to 4x ATR) |
| Late exits | Tighten trailing stop | `exit_rules.trailing_stop.atr_multiple` (decrease from 2x to 1.5x) |

### 5. Validate Changes
After modifying strategy JSON:
1. Run quick backtest (no walk-forward):
   ```bash
   python scripts/run_ultimate_backtest.py --strategy YourStrategy --months 60
   ```
2. Compare metrics before/after
3. If improved, run full validation with walk-forward
4. Commit changes with clear message

## Red Flags (Don't Ignore)
- **Sharpe < 0.3**: Strategy has poor risk-adjusted returns
- **Win Rate > 75%**: Likely overfitting or lookahead bias
- **Profit Factor > 10**: Suspicious, check for data errors
- **Max DD > 30%**: Unacceptable for swing trading, reduce risk
- **Total Trades < 100**: Insufficient sample size, extend backtest period

## Success Criteria
- Confidence Score: >= 55 (Moderate)
- CAGR: >= 15% (after realistic costs)
- Sharpe: >= 0.5
- Max DD: <= 20%
- Win Rate: 35-55% (realistic range)
- Profit Factor: >= 1.8
- Total Trades: >= 500 (for statistical significance)
