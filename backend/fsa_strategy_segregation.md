# FSA Strategy Segregation Plan — COMPLETED

## Overview
Transition the trading system from a monolithic configuration to a modular, multi-strategy architecture. Each strategy is self-contained in its own JSON configuration, allowing for dynamic loading and parallel/sequential execution.

**Status**: Fully implemented and operational.

## 1. Strategy JSON Structure — COMPLETED
Each strategy is defined in a JSON file located in `backend/strategies/`.

### Available Strategies

| File | Name | Status | Description |
|------|------|--------|-------------|
| `swing_trading.json` | Swing_Trading | Enabled | Classic swing with gates, multi-target exits, pyramiding |
| `momentum_trading.json` | Momentum_Trading | Enabled | 52-week high breakouts, RS leaders, wider stops |
| `hybrid_trading.json` | Hybrid_Trading | Enabled | Multi-factor combined strategy |
| `triple_confirm.json` | Nitin_Triple_Confirm_Retracement | Enabled | Retracement-based strategy |

### Strategy JSON Schema
Each strategy file contains:
- `name`: Unique identifier
- `enabled`: Boolean to activate/deactivate
- `description`: Strategy description
- `version`: Version string
- `analysis_config`: Toggle technical/fundamental/sentiment/sector/backtesting/risk/options_oi
- `analysis_weights`: Weight distribution (technical, fundamental, sentiment, sector)
- `stock_filters`: Price, volume, market cap, moving average filters
- `swing_trading_gates`: TREND_GATE, VOLATILITY_GATE, VOLUME_GATE, MTF_GATE
- `entry_patterns`: pullback_to_ema, bollinger_squeeze_breakout, macd_zero_cross, etc.
- `exit_rules`: Multi-target ATR-based exits, trailing stop, breakeven, time-stop
- `risk_management`: Position sizing, stop loss type, regime adaptive risk
- `pyramiding`: Multi-step position adding with ATR triggers
- `strategy_config`: Individual indicator modules with `enabled` and `is_bonus` flags
- `fundamental_config`: ROCE, debt-to-equity, profit growth thresholds
- `chartink_config`: Screening cache settings

## 2. Refined `config.py` — COMPLETED
`config.py` is limited to core infrastructure settings:
- MongoDB Connection (Host, Port, DB Name)
- Broker Credentials (5Paisa API keys, User ID, Password)
- Global Flags (`IS_PAPER`, `VERBOSE_LOGGING`, `AUTO_EXECUTE`)
- Global Timeouts and Threading counts
- Trading cost configuration (STT, stamp duty, brokerage, slippage)

## 3. Dynamic Strategy Loader — COMPLETED
**File**: `backend/utils/strategy_loader.py`

Implementation:
1. Scans `backend/strategies/*.json` using `os.listdir()`
2. Parses each file and validates mandatory `name` and `enabled` fields
3. `load_all_strategies()` returns all enabled strategies
4. `get_strategy_by_name(name)` fetches a specific strategy by name
5. Used by `run_ultimate_backtest.py`, `run_analysis.py`, and `main_orchestrator.py`

## 4. Execution Workflow — COMPLETED
For each enabled strategy:
1. **Initialize Engine**: Load the specific weights and parameters from JSON
2. **Analysis**: Run the full market scan using that strategy's logic
3. **Execution**: Identify recommendations and trigger paper/live orders
4. **Persistence**: Every recommendation and position includes `strategy_name` field

## 5. Risk & Portfolio Management — COMPLETED
- **Consolidated Monitor**: Single `PortfolioMonitor` tracks all open positions, applies specific `exit_rules` based on `strategy_name` metadata
- **Unified Capital**: All strategies share the `initial_capital` pool
- **Regime Adaptive Risk**: Each strategy has bull/bear risk parameters in JSON

## 6. Implementation Status
- [x] Initial EP Logic Hardened
- [x] Paper Trading Engine Verified
- [x] Create `backend/strategies/` directory
- [x] Migrate strategies to JSON (4 strategies defined)
- [x] Implement Dynamic Strategy Loader
- [x] Update Database Schema to include `strategy_name`
- [x] Multi-strategy backtest support in ultimate backtest runner
- [x] Strategy-specific Chartink screening rules

---
*Last Updated: 2026-05-17*
