# Skill: Backtest Validation & Interpretation

Guidelines for validating backtest results and interpreting confidence scores correctly.

## When to Use This Skill
- After running ultimate backtest
- Before deploying strategy to live trading
- When comparing multiple strategy versions
- When backtest results seem "too good to be true"

## Backtest Execution

### Quick Validation (5-10 minutes)
```bash
cd backend
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --telegram
```
Use for: Initial strategy screening, parameter exploration

### Full Validation (30-60 minutes)
```bash
cd backend
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --mc-iterations 12 --telegram
```
Use for: Final strategy validation before deployment

## Interpreting Results

### Phase 1: Historical Backtest
**Key Metrics:**
- **CAGR**: Annualized return (should be > 15% for swing trading)
- **Max Drawdown**: Worst peak-to-trough decline (should be < 20%)
- **Sharpe Ratio**: Risk-adjusted return (should be > 0.5)
- **Win Rate**: % of profitable trades (realistic: 35-55%)
- **Profit Factor**: Gross profit / Gross loss (should be > 1.8)

**Red Flags:**
- CAGR > 50%: Likely overfitting or unrealistic assumptions
- Win Rate > 70%: Suspicious, check for lookahead bias
- Max DD < 5%: Too conservative, missing opportunities
- Total Trades < 200: Insufficient sample size

### Phase 2: Statistical Validation
**Deflated Sharpe Ratio (DSR):**
- Adjusts Sharpe for multiple testing bias
- DSR > 0.5: Good (statistically significant)
- DSR < 0.3: Poor (could be luck)

**Monte Carlo Permutation:**
- Randomizes returns to test if Sharpe is from skill vs luck
- p-value < 0.05: Significant (skill, not luck)
- p-value > 0.10: Not significant (could be random)

**Minimum Track Record (MLRS):**
- Checks if sample size is sufficient
- If actual_years < required_years: Need more data

### Phase 3: Walk-Forward Analysis
**Purpose:** Tests strategy across different time periods (out-of-sample)

**Interpretation:**
- Mean CAGR across windows: Should be within 20% of Phase 1 CAGR
- Std Dev of CAGR: Lower is better (consistency)
- % Positive Windows: Should be > 70%

**Red Flags:**
- Walk-forward CAGR much lower than Phase 1: Overfitting
- High variance across windows: Unstable strategy
- Many negative windows: Strategy doesn't generalize

### Phase 4: Stress Tests
**Regime Tests:**
- Bull Market: CAGR should be high (> 20%)
- Bear Market: CAGR can be negative but DD should be controlled (< 15%)
- Crash: Max DD should be < 25%
- Sideways: CAGR can be low but positive

**Parameter Sensitivity:**
- Tests +/-20% variation in key parameters
- All variations should remain profitable
- If 50%+ variations fail: Strategy is parameter-sensitive (bad)

**Cost Sensitivity:**
- Tests different slippage/brokerage scenarios
- Strategy should remain profitable with 2x current costs

### Phase 5: Trade Diagnostics
**Avg Hold Time:**
- Swing trading: 5-20 days is typical
- < 3 days: Day trading behavior (wrong strategy type)
- > 30 days: Position trading (consider time-stop)

**Exit Reasons:**
- Target hit: Good (strategy working as designed)
- Stop loss: Acceptable if < 40% of exits
- Time stop: If > 30% of exits, entry timing is poor
- Trailing stop: Good (letting winners run)

**Max Concurrent Positions:**
- If always at max (8): Filters too loose, too many signals
- If rarely > 3: Filters too tight, missing opportunities
- Ideal: 4-6 average concurrent positions

### Phase 6: Confidence Score
**Scoring Breakdown:**
- Walk-Forward Robustness: 20%
- Deflated Sharpe Ratio: 15%
- Monte Carlo Permutation: 15%
- Stress Tests: 15%
- Parameter Stability: 10%
- Cost Resilience: 10%
- Data Sufficiency: 5%

**Confidence Levels:**
- 75-100: Very High (deploy with full allocation)
- 65-74: High (deploy with 75% allocation)
- 55-64: Moderate (deploy with 50% allocation)
- 45-54: Low (paper trade only, needs improvement)
- < 45: Very Low (do not use, redesign strategy)

**Realistic CAGR:**
- Applies haircut based on confidence level
- Moderate (55-64): 50% haircut (22% CAGR → 11% realistic)
- High (65-74): 30% haircut (22% CAGR → 15.4% realistic)
- Very High (75+): 15% haircut (22% CAGR → 18.7% realistic)

## Validation Checklist

Before deploying strategy:
- [ ] Confidence Score >= 55 (Moderate)
- [ ] Edge Verified = YES
- [ ] Walk-forward CAGR within 20% of Phase 1
- [ ] Max DD < 20% in all regimes
- [ ] Sharpe > 0.5 (Phase 1) and DSR > 0.5 (Phase 2)
- [ ] Monte Carlo p-value < 0.05
- [ ] Parameter sensitivity: > 70% variations profitable
- [ ] Total trades > 500 (statistical significance)
- [ ] Realistic CAGR meets your return expectations

## Common Pitfalls

**1. Overfitting to Historical Data**
- Symptom: Phase 1 CAGR much higher than walk-forward
- Solution: Simplify strategy, reduce parameters, increase walk-forward iterations

**2. Data Snooping Bias**
- Symptom: High Sharpe but low DSR
- Solution: Test on out-of-sample data, use multiple time periods

**3. Unrealistic Cost Assumptions**
- Symptom: Strategy profitable in backtest but not live
- Solution: Increase slippage assumptions, add market impact costs

**4. Regime Dependence**
- Symptom: Works in bull markets, fails in bear/sideways
- Solution: Add market regime filters, adjust gates by regime

**5. Insufficient Sample Size**
- Symptom: High variance in results, wide confidence intervals
- Solution: Extend backtest period, test on more stocks

## Querying Past Results

```bash
cd backend
python3 << 'EOF'
from database import get_mongodb
db = get_mongodb()

# Get all completed sessions
sessions = list(db.backtest_sessions.find(
    {'status': 'completed'},
    sort=[('completed_at', -1)],
    limit=10
))

for s in sessions:
    print(f"\n{s.get('session_name')}")
    m = s.get('summary_metrics', {})
    print(f"  CAGR: {m.get('cagr'):.2f}%")
    print(f"  Sharpe: {m.get('sharpe_ratio'):.2f}")
    print(f"  Max DD: {m.get('max_drawdown_pct'):.2f}%")
    
    phases = s.get('ultimate_phases', {})
    if phases:
        conf = phases.get('confidence_score', {})
        print(f"  Confidence: {conf.get('total_score')}/100 ({conf.get('confidence_level')})")
EOF
```
