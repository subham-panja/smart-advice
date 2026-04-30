# FSA Strategy Segregation Plan

## Overview
Transition the trading system from a monolithic configuration to a modular, multi-strategy architecture. Each strategy will be self-contained in its own JSON configuration, allowing for dynamic loading and parallel/sequential execution.

## 1. Strategy JSON Structure
Each strategy will be defined in a JSON file located in `backend/strategies/`.

**Example: `delayed_ep.json`**
```json
{
    "name": "Delayed_EP",
    "enabled": true,
    "weights": {
        "technical": 0.90,
        "fundamental": 0.05,
        "sector": 0.05
    },
    "entry_patterns": [
        {
            "name": "pullback_to_ema",
            "ema_period": 10,
            "rsi_range": [40, 70]
        }
    ],
    "exit_rules": {
        "targets": [
            {"name": "Target 1", "atr_multiplier": 3.0, "sell_percentage": 0.5},
            {"name": "Target 2", "atr_multiplier": 4.5, "sell_percentage": 1.0}
        ],
        "atr_stop_multiplier": 1.5,
        "trail_stop_atr": 3.0,
        "time_stop_bars": 15,
        "breakeven_at_target_1": true
    }
}
```

## 2. Refined `config.py`
The `config.py` file will be limited to core infrastructure settings:
- MongoDB Connection (Host, Port, DB Name)
- Broker Credentials (5Paisa API keys, User ID, Password)
- Global Flags (`IS_PAPER`, `VERBOSE_LOGGING`, `AUTO_EXECUTE`)
- Global Timeouts and Threading counts

## 3. Dynamic Strategy Loader
Implementation of an automated loader in the `main_orchestrator.py` or a dedicated `StrategyManager`:
1. Use `os.listdir()` to scan `backend/strategies/*.json`.
2. Parse each file and validate the mandatory `"name"` field.
3. Register the strategy in the global execution registry.

## 4. Execution Workflow
For each enabled strategy in the registry:
1. **Initialize Engine**: Load the specific weights and parameters for the strategy.
2. **Analysis**: Run the full market scan (Phase 2) using that strategy's logic.
3. **Execution**: Identify recommendations and trigger paper/live orders (Phase 3).
4. **Persistence**: Every recommendation and position MUST include a `strategy_name` field for tracking.

## 5. Risk & Portfolio Management
- **Consolidated Monitor**: A single `PortfolioMonitor` will track all open positions but will apply the specific `exit_rules` based on the `strategy_name` metadata stored in each position.
- **Unified Capital**: All strategies share the `initial_capital` pool unless sub-allocation logic is implemented later.

## 6. Implementation Status
- [x] Initial EP Logic Hardened
- [x] Paper Trading Engine Verified
- [ ] Create `backend/strategies/` directory
- [ ] Migrate `Delayed_EP` settings to JSON
- [ ] Implement Dynamic Strategy Loader
- [ ] Update Database Schema to include `strategy_name`
