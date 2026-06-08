# Smart Advice — Improvement Plan

## Current State (as of June 2026)

**Backend**: Flask API on port 5001 with MongoDB. Orchestrator runs trading cycles (monitor, analyze, execute). Telegram bot sends notifications. Paper trading mode with position sizing based on risk management.

**Frontend**: Next.js dashboard with charts, recommendations table, and analysis trigger. Settings page with theme toggle.

**Strategies**: 5 strategy files (Swing_Trading enabled, others disabled). Risk-based position sizing with 2% risk per trade, 30% max position, 4 max positions.

**Capital**: Rs.10,000 paper trading capital.

---

## Completed (This Update)

### Backend APIs Added
- `GET /stream-logs` — SSE log streaming (fixes existing Terminal component)
- `POST /run-orchestrator` — Run trading cycle with mode support (live/replay/date)
- `GET /orchestrator-status` — Poll orchestrator progress
- `GET /positions` — List positions with status filter
- `POST /positions` — Create position manually (for real buy entry)
- `PATCH /positions/<symbol>` — Update any field except symbol (entry price, qty, SL, target, etc.)
- `DELETE /positions/<symbol>` — Manual close position
- `GET /strategies` — List all strategies with enabled/disabled status
- `GET /settings/trading` — Return TRADING_OPTIONS from config
- `GET /cycle-stats` — Portfolio summary (equity, cash, PnL)

### Database Helpers Added
- `get_all_positions(status_filter)` — Query positions with optional filter
- `get_position_by_symbol(symbol, status)` — Find single position
- `insert_position()` — Duplicate guard (checks existing OPEN before insert)
- `update_position()` — History tracking (logs every change type)

### Frontend Changes
- **Sidebar**: F&O pages commented out, Positions page added, Analysis renamed to "Trading Cycle"
- **Trading Cycle page** (`/analysis`): Run orchestrator with mode selection (Paper/Replay/Date), embedded SSE terminal
- **Positions page** (`/positions`): Open positions table with inline edit/close, Add Position form, Today's BUY signals with auto-fill
- **Settings page** (`/settings`): Trading Config display (paper mode badge, capital, brokerage), Strategies table (enabled/disabled)
- **Dashboard** (`/`): Portfolio summary cards (positions, equity, cash, PnL), "Run Trading Cycle" button
- **API client** (`api.ts`): 8 new typed API functions with interfaces

---

## Planned Features

### P1 — High Priority

1. **Position Alerts**: Telegram + dashboard notification when SL is trailed, target hit, or time stop triggered. Currently only shows in logs.

2. **Closed Trades History**: Separate tab on Positions page showing closed positions with PnL, exit reason, and lifecycle events.

3. **Recommendation Persistence**: Recommendations are purged between cycles. Add a "Saved Signals" feature to keep important BUY signals across days.

4. **Error Recovery**: If orchestrator fails mid-cycle (e.g., Chartink API down), positions should not be left in inconsistent state. Add transaction rollback.

### P2 — Medium Priority

5. **PnL Chart Over Time**: Daily equity curve chart on dashboard. Track equity snapshots in a new `equity_snapshots` collection after each cycle.

6. **Telegram → Dashboard Sync**: When Telegram bot sends position updates, sync those updates to the dashboard. Currently Telegram and dashboard read from the same DB but there's no push notification to the web UI.

7. **Strategy Toggle from UI**: Enable/disable strategies from the Settings page instead of editing JSON files manually.

8. **Backtest Integration**: Run backtests from the dashboard with results displayed inline. Currently only possible via CLI scripts.

### P3 — Nice to Have

9. **VTT Order Templates**: Generate 5paisa VTT order JSON/templates from positions that can be copy-pasted or auto-submitted.

10. **Multi-Broker Support**: Abstract the broker layer so other brokers (Zerodha, Angel One) can be plugged in alongside 5paisa.

11. **Market Regime Dashboard Widget**: Show current regime (BULL/BEAR) with a visual indicator on the dashboard.

12. **Notification Preferences**: Choose which events trigger Telegram vs dashboard-only notifications.

---

## Known Issues

1. **Same-day re-entry**: If a stock is closed by TIME_STOP in Phase 1 and re-signaled as BUY in Phase 2, the orchestrator creates a new position the same cycle. Consider adding a "cooldown" list for symbols closed this cycle.

2. **Delivery volume 403 errors**: NSE delivery volume API returns HTTP 403 for many stocks. Needs a fallback or rate-limiting strategy.

3. **MongoDB positions empty after cycle**: Positions may not persist between runs if the DB is being reset externally. Investigate if a cron job or external process is clearing the collection.

4. **Log noise**: `app.log` accumulates 37K+ lines of "Fetching data for XYZ" noise. Consider log rotation or reducing verbosity for routine data fetches.

---

## Architecture Notes

```
backend/
  app.py              → Flask API (10+ routes)
  main_orchestrator.py → Trading cycle engine
  database.py          → MongoDB CRUD with history tracking
  config.py            → TRADING_OPTIONS, MongoDB config
  strategies/          → JSON strategy files
  utils/
    strategy_loader.py → Loads enabled + disabled strategies
    persistence_handler.py → Saves analysis results
    trading_clock.py   → Simulated date support for replay

frontend/
  src/app/
    page.tsx           → Dashboard with portfolio summary
    analysis/page.tsx  → Trading Cycle runner
    positions/page.tsx → Position management
    settings/page.tsx  → Trading config + strategies
    recommendations/   → Recommendations table
  src/lib/api.ts       → All API client functions
```
