# Workflow: Adding a Strategy

There are two types of "strategies" in this system:
1. **Technical Indicator Modules** - Reusable TA-Lib calculations in `backend/scripts/strategies/`.
2. **Trading Strategy Configs** - Complete JSON strategy files in `backend/strategies/`.

Use the appropriate workflow below.

---

## A. Adding a Technical Indicator Module

Use this for new indicator calculations (e.g., a new oscillator or moving average strategy).

### 1. Create Module File
- Location: `backend/scripts/strategies/`
- Filename: `[strategy_name].py` (snake_case)
- Pattern: Inherit from `BaseStrategy` (found in `strategy_evaluator.py`).

### 2. Implement Logic
- Use `TA-Lib` for standard calculations.
- Return a signal: `1` (Buy), `-1` (Sell), or `0` (Neutral).
- Ensure vectorized operations (no Python loops over DataFrame rows).

### 3. Register in Strategy JSON
- Add the module to the `strategy_config` section of the relevant `backend/strategies/*.json` file.
- Set `enabled: true/false` and `is_bonus: true/false`.
  - `is_bonus: false` = Hard requirement (must pass for BUY).
  - `is_bonus: true` = Bonus indicator (adds to score but doesn't block).

### 4. Test Module
```bash
cd backend
python -c "from scripts.strategies.your_module import YourStrategy; print('OK')"
```

---

## B. Adding a Trading Strategy Config

Use this for a complete new trading system with gates, entry patterns, and exit rules.

### 1. Create JSON File
- Location: `backend/strategies/`
- Filename: `[strategy_name].json` (snake_case)
- Template: Copy from `backend/strategies/delayed_ep.json`.

### 2. Define Key Sections
- `name`: Unique strategy name.
- `enabled`: Set to `true` to activate.
- `analysis_config`: Toggle technical/fundamental/sentiment/sector/backtesting/risk/options_oi.
- `analysis_weights`: Weight distribution (must sum to 1.0).
- `stock_filters`: Price, volume, market cap, moving average filters.
- `swing_trading_gates`: TREND_GATE, VOLATILITY_GATE, VOLUME_GATE, MTF_GATE.
- `entry_patterns`: pullback_to_ema, bollinger_squeeze_breakout, macd_zero_cross, etc.
- `exit_rules`: Targets (ATR-based), trailing stop, breakeven, time-stop.
- `strategy_config`: Individual indicator modules with `enabled` and `is_bonus` flags.

### 3. Validate JSON
```bash
cd backend
python -c "import json; json.load(open('strategies/your_strategy.json')); print('JSON Valid')"
```

### 4. Test Strategy
```bash
cd backend
python -c "from utils.strategy_loader import StrategyLoader; s = StrategyLoader.get_strategy_by_name('YourStrategy'); print('Loaded:', s['name'])"
```

### 5. Run Analysis
```bash
cd backend
python run_analysis.py
```
Note: The pipeline automatically loads all enabled strategies. No need to pass `--symbol` or `--max-stocks` flags directly to `run_analysis.py` unless testing specific paths.

---

## C. Adding an Entry Pattern

Entry patterns are defined inside the JSON strategy file under `entry_patterns`.

Supported patterns:
- `pullback_to_ema`: Price near EMA with RSI in range.
- `bollinger_squeeze_breakout`: Bandwidth squeeze + upper band breakout.
- `macd_zero_cross`: MACD crosses from below to above zero.
- `higher_low_structure`: Rising swing lows.
- `volatility_contraction`: Decreasing ATR over last 5 days.

To add a new pattern type, you must update both:
1. The JSON schema in the strategy file.
2. The logic in `backend/scripts/swing_trading_signals.py` (`analyze_swing_opportunity` method).
