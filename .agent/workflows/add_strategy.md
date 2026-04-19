# Workflow: Adding a Technical Strategy

Use this workflow to add a new indicator or trading strategy to the engine.

## 1. Create Strategy File
- Location: `backend/scripts/strategies/`
- Filename: `[strategy_name].py` (snake_case)
- Pattern: Inherit from `BaseStrategy` (found in `strategy_evaluator.py`).

## 2. Implement Logic
- Use `TA-Lib` for standard calculations.
- Return a signal: `1` (Buy), `-1` (Sell), or `0` (Neutral).

## 3. Register in Config
- Add the strategy to `STRATEGY_CONFIG` in `backend/config.py`.
- Set to `True` for testing.

## 4. Test Strategy
- Run a single-stock analysis to ensure no import errors or math overflows.
```bash
python run_analysis.py --single-threaded --symbol RELIANCE
```
