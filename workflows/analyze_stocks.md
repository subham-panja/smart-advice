# Workflow: Stock Analysis Execution

Follow this workflow to trigger and verify a stock analysis run.

## 1. Environment Setup
- Ensure MongoDB is running: `brew services start mongodb/brew/mongodb-community`
- Ensure Redis is running: `redis-server` (if not autostarted)

## 2. Configuration Check
- Open `backend/config.py`.
- Verify `MIN_RECOMMENDATION_SCORE` and `STRATEGY_CONFIG`.
- For testing, set `max_stocks` to a small number (e.g., 10).

## 3. Execution
Run the analysis from the `backend` directory:
```bash
python run_analysis.py --max-stocks 10
```

## 4. Verification
- **Logs**: Check `backend/logs/` for any "Network Error" or "API 429" warnings.
- **Database**: Run `python archive/check_results.py` to confirm results were saved to MongoDB.
- **Frontend**: Refresh `localhost:3000` to see the new dashboard data.
