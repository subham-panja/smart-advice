# Claude Code Instructions — Smart Advice Project

## MANDATORY: Before ANY Task (Non-Negotiable)

You MUST complete this checklist before writing code or answering questions:

### Step 1: Run Startup Script
```bash
python .agent/startup_check.py
```
Reads all key docs and saves read receipt to `.agent/last_read.json`.

### Step 2: Read Core Documents (in order)
1. **AGENT.md** — Project mission, tech stack, rules, directory structure
2. **All 6 Skills** (read EVERY file completely):
   - `skills/backtest_validation.md`
   - `skills/data_validation.md`
   - `skills/entry_pattern_optimization.md`
   - `skills/performance_debugging.md`
   - `skills/risk_management.md`
   - `skills/strategy_analysis.md`
3. **All 3 Workflows** (read EVERY file completely):
   - `.agent/workflows/analyze_stocks.md`
   - `.agent/workflows/add_strategy.md`
   - `.agent/workflows/frontend_development.md`

### Step 3: Verify Read Receipt
Check `.agent/last_read.json` exists with timestamps for all docs.

## Communication Rules (MANDATORY)
1. **Keep it SHORT**: 1-3 sentences by default. Expand only if user says "explain", "detail", "show me".
2. **NO CODE by default**: Never show code unless explicitly requested.
3. **Direct answers only**: No preamble, no examples unless asked.
4. **Prove on request**: Show code/logs only when user says "prove", "show code".

## Code Writing Rules (MANDATORY)
1. **MINIMAL CODE**: Write the absolute minimum lines needed. If it can be done in 50 lines, do NOT write 500 or 5000.
2. **NO VERBOSE COMMENTS**: No comments explaining what code does. Only comment on non-obvious WHY (bug workarounds, hidden constraints).
3. **NO BLOAT**: No unnecessary abstractions, helpers, wrappers, or "future-proofing." Do exactly what's asked.
4. **NO BOILERPLATE**: No excessive docstrings, redundant validation, unused type hints, or defensive code for impossible scenarios.
5. **HUMAN STANDARD**: Write like a senior human dev — short, clean, direct. If it looks like a textbook, it's too much.
6. **EDIT OVER CREATE**: Prefer editing existing files. Only create new files when absolutely necessary.
7. **ONE TASK = ONE CHANGE**: Don't refactor surrounding code or add unrequested features.

## Quick Reference
- **Project**: Stock analysis platform for Indian Equity Market (NSE)
- **Backend**: Python (Flask), MongoDB, Redis, TA-Lib, vectorbt
- **Frontend**: Next.js 15.5, React 19, Tailwind CSS v4
- **Strategies**: JSON-configured in `backend/strategies/`
- **Data cache**: `backend/data/historical/` (parquet files)
- **Workflows**: `.agent/workflows/`
- **Skills**: `skills/`

## Key Rules
- Strategies are JSON-configured in `backend/strategies/`
- Use vectorized operations (numpy/pandas/vectorbt)
- Never bypass circuit breaker or risk controls
- Data cached in `backend/data/historical/` as parquet files
- All backtest results save to MongoDB
- **ALWAYS read relevant skills before working** (see AGENT.md)

## Task Routing
1. Check `AGENT.md` for project context
2. Check `.agent/workflows/` for task-specific SOP
3. Read relevant `skills/*.md` files completely
4. Execute & verify with tests

## Technology Stack
- Backend: Python (Flask), MongoDB, Redis
- Frontend: Next.js 15.5, React 19, Tailwind CSS v4
- Analysis: TA-Lib, yfinance, vectorbt
- ML: PyTorch, HuggingFace Transformers, stable-baselines3

Read AGENT.md for complete details.
