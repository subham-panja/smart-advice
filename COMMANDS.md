# Smart Advice - Analysis Commands

## Quick Start Commands

### 1. Analyze ALL Stocks with Quality Recommendations
```bash
backend/venv/bin/python backend/run_analysis.py
```
- Uses all available NSE stocks
- Shows progress bar with count
- Quality thresholds: CAGR ≥ 3%, positive technical/fundamental
- Saves only strong recommendations to database

### 2. Analyze Limited Number of Stocks (Testing)
```bash
# Test with 50 stocks
backend/venv/bin/python backend/run_analysis.py --max-stocks 50

# Test with 100 stocks
backend/venv/bin/python backend/run_analysis.py --max-stocks 100

# Full 500 stocks
backend/venv/bin/python backend/run_analysis.py --max-stocks 500
```

### 3. Verbose Mode (See Detailed Logs)
```bash
backend/venv/bin/python backend/run_analysis.py --max-stocks 50 --verbose
```
Shows detailed logs for debugging

### 4. Check Saved Recommendations
```bash
backend/venv/bin/python backend/check_recommendations.py
```

## Current Threshold Settings

**Quality Recommendations** (backend/config.py):
```python
'buy_combined': 0.10,              # Positive combined score required
'min_backtest_return': 3.0,        # Minimum 3% CAGR
'technical_minimum': 0.05,         # Slightly positive technical
'fundamental_minimum': 0.05,       # Slightly positive fundamental
'volume_confirmation_required': True,
'strong_buy_combined': 0.50,       # Strong signals only
```

**What Gets Saved**:
- ✅ Stocks with positive momentum (technical + fundamental)
- ✅ Backtested strategies with ≥3% annual returns
- ✅ Volume-confirmed signals (not just price action)
- ❌ HOLD recommendations (filtered out)
- ❌ Weak signals with 0% CAGR

## Progress Display

### Normal Mode (Non-Verbose)
```
Analyzing 55 actively traded stocks...
Progress: 45.5% (25/55) - 3 recommendations
```
- Clean, single-line progress
- Shows: percentage, count, recommendations found
- Updates in real-time

### Verbose Mode
```
2025-11-26 15:10:00 - Analyzing SBIN (1/55)
2025-11-26 15:10:02 - Technical Score: -0.30
2025-11-26 15:10:02 - Fundamental Score: 0.56
...detailed logs...
```
- Full analysis logs
- Useful for debugging

## Understanding the Results

### Empty Fields in Database
Some fields will be empty if advanced analysis is disabled:
- `alternative_data`: Requires external data sources
- `market_regime`: Requires regime detection (disabled for speed)
- `sector_analysis`: Disabled for speed
- `rl_action`: Reinforcement learning (disabled)

These can be enabled in `backend/config.py` line 147:
```python
ANALYSIS_CONFIG = {
    'sector_analysis': True,        # Enable sector analysis
    'market_regime_detection': True, # Enable market regime
    ...
}
```
**Warning**: Enabling these significantly increases analysis time.

### Key Fields to Check
- `combined_score`: Overall recommendation score
- `backtest_metrics.cagr`: Historical performance (target: ≥3%)
- `backtest_metrics.win_rate`: Percentage of winning trades
- `technical_score`: Short-term momentum
- `fundamental_score`: Company fundamentals
- `expected_return_percent`: Target profit %
- `est_time_to_target`: Expected holding period

## Troubleshooting

### No Recommendations Found
If you run analysis and get 0 recommendations:
1. **Normal market conditions**: Sometimes no stocks meet criteria
2. **Too strict thresholds**: Lower `buy_combined` to 0.05 or `min_backtest_return` to 2.0
3. **Check logs**: Use `--verbose` to see why stocks are rejected

### Analysis Takes Too Long
- Use `--max-stocks 50` for quick tests
- Fundamental and sentiment analysis are slow
- Each stock takes ~5-10 seconds with full analysis

### MongoDB Connection Issues
```bash
# Check MongoDB is running
cd backend && ../backend/venv/bin/python -c "from database import get_mongodb; print('Connected:', get_mongodb().name)"
```

## Recommended Workflow

1. **Quick Test** (2-3 minutes):
   ```bash
   backend/venv/bin/python backend/run_analysis.py --max-stocks 25
   ```

2. **Check Results**:
   ```bash
   backend/venv/bin/python backend/check_recommendations.py
   ```

3. **Full Analysis** (30-60 minutes for all stocks):
   ```bash
   backend/venv/bin/python backend/run_analysis.py
   ```

4. **Review on Dashboard**:
   - Open http://localhost:3000
   - View recommendations with charts

## Advanced Options

### Clear Old Data First
```bash
# Run analysis with data purge (keeps last 7 days)
backend/venv/bin/python backend/run_analysis.py --purge-days 7

# Clear ALL data and start fresh
backend/venv/bin/python backend/run_analysis.py --purge-days 0
```

### Use All Symbols (Not Just Actively Traded)
```bash
backend/venv/bin/python backend/run_analysis.py --all
```
Note: May include illiquid stocks

## Expected Output

With current quality settings, expect:
- **50 stocks**: 1-3 recommendations
- **500 stocks**: 5-15 recommendations  
- **All stocks (~2000)**: 20-50 recommendations

Quality over quantity! Better to have 10 strong signals than 100 weak ones.
