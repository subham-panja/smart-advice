# Smart Advice Backend — Technical Guide & Command Reference

An institutional-grade algorithmic swing trading, multi-timeframe analysis, and vectorized backtesting engine for the Indian Equity Market (NSE).

---

## 🚀 Key CLI & Execution Commands

### 1. Ultimate Strategy Backtest Suite
Runs full multi-phase statistical validation (historical simulation, MongoDB saving, Deflated Sharpe Ratio calculation, stress testing, and composite scoring).

```bash
# Basic 12-month backtest
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 12

# Fast 12-month backtest (skipping statistical & stress phases)
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 12 --skip-phases 2,3,4

# Full 24-month backtest with Walk-Forward Monte Carlo (12 iterations per window)
python scripts/run_ultimate_backtest.py --strategy Swing_Trading --months 24 --mc-iterations 12
```

#### Command Arguments:
- `--strategy <NAME>`: Strategy JSON file name in `backend/strategies/` (e.g. `Swing_Trading`, `Momentum_Trading`, `Hybrid_Trading`). Default: `Swing_Trading`.
- `--months <N>`: Historical lookback period in months (e.g. `6`, `12`, `24`, `60`). Default: `120`.
- `--mc-iterations <N>`: Enables Phase 3 Walk-Forward Monte Carlo simulations with N iterations per window.
- `--skip-phases <P1,P2...>`: Comma-separated phase numbers to skip (e.g. `--skip-phases 2,3,4`).
- `--telegram`: Sends backtest summary report to Telegram bot.

---

### 2. Fast Portfolio Backtest Engine
Runs a single-pass historical simulation over cached NSE OHLCV data.

```bash
# Run 1-year backtest
python scripts/run_portfolio_backtest.py --strategy Swing_Trading --period 1y

# Run 2-year backtest without saving to MongoDB
python scripts/run_portfolio_backtest.py --strategy Swing_Trading --period 2y --no-db
```

#### Command Arguments:
- `--strategy <NAME>`: Strategy name (e.g. `Swing_Trading`).
- `--period <PERIOD>`: Lookback period (`6m`, `1y`, `2y`, `5y`, `10y`).
- `--no-db`: Disable MongoDB persistence for fast scratch runs.
- `--verbose`: Enable detailed trade-by-trade logging.

---

### 3. Live & Paper Trading Orchestrator
Executes live or paper trading scans across NSE stocks using configured strategy rules.

```bash
python main_orchestrator.py
```

---

### 4. Flask API Server
Launches the backend REST API server for frontend interaction.

```bash
python app.py
```

---

### 5. Real-Time Stock Scanner
Scans the NSE universe for current buy candidates based on technical and fundamental strategy gates.

```bash
python scripts/run_scanner.py --strategy Swing_Trading
```

---

### 6. Automated Unit Test Suite
Runs pytest unit tests across backend components and handlers.

```bash
pytest tests/
```

---

## 🛠 Project Configuration

- **Strategy Definitions**: JSON files located in `backend/strategies/` (`swing_trading.json`, `momentum_trading.json`, `hybrid_trading.json`).
- **App Configuration**: Core parameters, MongoDB connections, and execution options in `config.py`.
  - `TRADING_OPTIONS`: Controls live/paper trading settings (`is_paper_trading`, `initial_capital`).
  - `PORTFOLIO_BACKTEST_CONFIG`: Controls backtesting defaults (`brokerage_charges`, `slippage_pct`, `initial_capital`).
