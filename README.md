# Smart Advice - AI-Powered Stock Analysis Platform

Smart Advice is a stock market analysis platform combining technical, fundamental, and ML-based analysis with paper trading and multi-strategy backtesting.

## Features

- **Technical Analysis**: 55+ indicators, chart patterns, and swing trading gates (trend, volatility, volume, multi-timeframe)
- **Strategy System**: JSON-configured strategies in `backend/strategies/` with entry patterns, multi-target exits, and pyramiding
- **Paper Trading**: Full simulation with portfolio tracking, ATR-based stops, and circuit breakers
- **Walk-Forward Backtesting**: Rolling 6-month windows with Monte Carlo sampling for robustness validation
- **Options OI Analysis**: NSE Option Chain with PCR and unwinding detection
- **Telegram Bot**: Remote control for analysis and portfolio monitoring
- **ML/RL Models**: LSTMs, HMM regime detection, and reinforcement learning agents

## Architecture

- **Frontend**: Next.js 15 + React 19 + TypeScript + Tailwind CSS + Chart.js
- **Backend**: Flask + Pandas/NumPy + TA-Lib + PyTorch + scikit-learn
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

## Walk-Forward Backtesting

Validate strategies across time periods and stock universes:

```bash
cd backend
python scripts/run_portfolio_backtest.py --strategy Swing_Trading --walk-forward --mc-iterations 8 --period 4y --max-stocks 50
```

Splits history into rolling 6-month windows, runs Monte Carlo sampling (70% stocks, N iterations), and aggregates CAGR, Sharpe, max drawdown, win rate, and robustness score.

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

# Walk-forward backtest
python scripts/run_portfolio_backtest.py --strategy Swing_Trading --walk-forward --mc-iterations 8

# Tests
npm run test              # Frontend
python tests/test_complete_system.py  # Backend integration
```

## Configuration

Strategy configs live in `backend/strategies/*.json`. Main settings in `backend/config.py` (MongoDB, risk management, trading parameters).

## License

MIT — for educational and research purposes. Consult financial advisors before investing.
