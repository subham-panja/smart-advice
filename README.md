# Smart Advice - AI-Powered Stock Analysis Platform

Smart Advice is a stock market analysis platform combining technical, fundamental, and ML-based analysis with paper trading and multi-strategy backtesting.

## Features

- **Technical Analysis**: 55+ indicators, chart patterns, and swing trading gates (trend, volatility, volume, multi-timeframe)
- **Strategy System**: JSON-configured strategies in `backend/strategies/` with entry patterns, multi-target exits, and pyramiding
- **Paper Trading**: Full simulation with portfolio tracking, ATR-based stops, and circuit breakers
- **Ultimate Backtesting**: 6-phase validation pipeline — historical backtest with realistic costs, statistical validation (DSR, MC permutation, MLRS), walk-forward Monte Carlo, stress tests, trade diagnostics, and composite confidence scoring
- **Options OI Analysis**: NSE Option Chain with PCR and unwinding detection
- **Telegram Bot**: Remote control for analysis and portfolio monitoring
- **ML/RL Models**: LSTMs, HMM regime detection, and reinforcement learning agents

## Architecture

- **Frontend**: Next.js 15 + React 19 + TypeScript + Tailwind CSS + Chart.js
- **Backend**: Flask + Pandas/NumPy + TA-Lib + PyTorch + scikit-learn + vectorbt
- **Database**: MongoDB (documents) + Redis (cache)

## Quick Start

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py  # http://localhost:5001
```

### Frontend
```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:5001" > .env.local
npm run dev  # http://localhost:3000
```

## Available Strategies

| Strategy | File | Status | Description |
|----------|------|--------|-------------|
| Swing Trading | `swing_trading.json` | Enabled | Classic swing with gates, multi-target exits, pyramiding |
| Momentum Trading | `momentum_trading.json` | Disabled | 52-week high breakouts, RS leaders |
| Hybrid Trading | `hybrid_trading.json` | Disabled | Multi-factor combined strategy |
| Nitin Triple Confirm | `triple_confirm.json` | Disabled | Retracement-based strategy |

## Ultimate Backtest

Run full 6-phase strategy validation:

```bash
cd backend
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --max-stocks 10 --mc-iterations 8
```

**Phases:**
1. Historical backtest with realistic costs (gap risk, STT, slippage, brokerage)
2. Statistical validation (DSR, Monte Carlo permutation, MLRS)
3. Walk-Forward Monte Carlo (rolling windows, parallel execution)
4. Stress tests (regime, parameter sensitivity, cost sensitivity)
5. Trade diagnostics (hold times, exit reasons, concurrent positions)
6. Composite confidence score (0-100 with realistic CAGR projection)

**Performance:** ~2 minutes for 10y/10-stock/8-iter with all phases (vectorbt indicator pre-computation, O(1) IndicatorStore lookups, sequential walk-forward).

**Benchmark Results (Swing_Trading, 10y, 10 stocks, 8 iter):**
| Metric | Value |
|---|---|
| Initial Capital | ₹100,000 |
| Final Value | ₹1,017,225 |
| CAGR | 26.13% |
| Max Drawdown | -12.25% |
| Sharpe Ratio | 0.32 |
| Profit Factor | 19.63 |
| Win Rate | 51.9% |
| Total Trades | 206 |
| Confidence | 55/100 (Moderate) |
| Realistic CAGR | 13.1% |

## Key Commands

```bash
# Run analysis
python run_analysis.py

# Full trading cycle (analysis + execution + backtest)
python main_orchestrator.py

# Paper trading monitor
python scripts/portfolio_monitor_paper.py

# Telegram bot
python telegram_bot.py

# Ultimate backtest (full validation)
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 120 --max-stocks 10 --mc-iterations 8

# Quick backtest (historical only, skip validation)
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --skip-wf --skip-stress

# Tests
npm run test              # Frontend
python tests/test_complete_system.py  # Backend integration
```

## Configuration

Strategy configs live in `backend/strategies/*.json`. Main settings in `backend/config.py` (MongoDB, risk management, trading parameters).

## License

MIT — for educational and research purposes. Consult financial advisors before investing.
