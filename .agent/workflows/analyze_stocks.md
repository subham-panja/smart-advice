# Workflow: Stock Analysis Execution

Follow this workflow to trigger and verify a stock analysis run.

## 1. Environment Setup
- Ensure MongoDB is running: `brew services start mongodb/brew/mongodb-community`
- Ensure Redis is running: `redis-server` (if not autostarted)

## 2. Configuration Check
- Open `backend/config.py`.
  - Verify `TRADING_OPTIONS["circuit_breaker"]` is `False` (or `True` if you want to halt trading).
  - Verify `USE_MULTIPROCESSING_PIPELINE` and `NUM_WORKER_PROCESSES`.
- Open `backend/strategies/*.json` files.
  - Verify `enabled: true` for the strategies you want to run.
  - Check `stock_filters`, `swing_trading_gates`, and `entry_patterns` for correctness.
- For testing, reduce the number of symbols by editing `max_stocks` in the strategy JSON or using a small symbol group.

## 3. Execution Options

### Option A: Run Analysis Only
Runs the two-phase pipeline (fetch + analyze) for all enabled strategies:
```bash
cd backend
python run_analysis.py
```

### Option B: Run Full Trading Cycle
Runs analysis + paper/live trading execution + portfolio monitoring:
```bash
cd backend
python main_orchestrator.py
```

### Option C: Trigger via Frontend
Use the "Start Full Analysis" button on `http://localhost:3000`.

### Option D: Trigger via Telegram
Send "Run Analysis" to the configured Telegram bot.

## 4. Verification
- **Logs**: Check `backend/logs/` for any "Network Error" or "API 429" warnings.
- **Database**: Check MongoDB collections:
  - `recommended_shares` - New recommendations
  - `backtest_results` - Backtest metrics
  - `scan_runs` - Scan execution history
  - `swing_gate_results` - Gate pass/fail data
- **Audit Log**: Check `backend/logs/audit_log.json` for detailed per-stock analysis logs.
- **Frontend**: Refresh `http://localhost:3000` to see the new dashboard data.
- **Positions**: If running the full cycle, check `positions` collection for new/open trades.

## 5. Position Monitoring (Post-Analysis)
To monitor existing positions for exits (SL hit, target hit, time-stop):
```bash
cd backend
python scripts/portfolio_monitor_paper.py
```
