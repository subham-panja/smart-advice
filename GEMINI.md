# AI Agent Steering Guidelines — Smart Advice Project

> **CRITICAL DIRECTIVE FOR ALL AI AGENTS & SESSIONS:**
> **ALWAYS USE SUBAGENTS AND BREAK DOWN EVERY TASK ACROSS MULTIPLE SPECIALIZED SUBAGENTS.**

---

## 1. Multi-Subagent Orchestration Protocol (MANDATORY)

Every AI assistant working in this repository must operate as an **Orchestrator** and adhere to the following workflow:

1. **Task Decomposition Before Execution**:
   - Never attempt to solve a non-trivial user prompt in a single monolithic pass.
   - Immediately decompose the user request into modular, concurrent, or sequential sub-tasks:
     - **Researcher / Scout**: Investigates codebase files, interfaces, schemas, and dependencies.
     - **Implementer / Coder**: Writes clean, minimal, human-standard code for targeted modules.
     - **Tester / Verifier**: Runs test suites, validates API endpoints, checks linting/types, and audits behavior.
     - **Reviewer / Auditor**: Verifies that changes strictly adhere to risk controls, strategy integrity, and API contracts.

2. **Always Invoke Subagents**:
   - For every task (feature, bugfix, refactor, API addition, analysis, or test), launch specialized subagents to handle distinct workstreams.
   - Utilize subagents in parallel whenever sub-tasks are independent.
   - Pass concise, unambiguous prompts with explicit scope and required outputs to each subagent.

3. **Synthesis & Verification**:
   - Aggregate outputs from all subagents.
   - Run end-to-end verification and integration tests before presenting solutions.

---

## 2. API-First Strategy Target

The Smart Advice trading system is designed to be **API-First**:
- All capabilities (screeners, technical strategies, backtest engines, positions, execution orders, market clocks) must be accessible via clean, documented REST endpoints and Swagger/OpenAPI.
- Every API endpoint must have a corresponding MCP (Model Context Protocol) tool registered so AI agents can execute trading workflows programmatically.
- Client applications (Next.js frontend, Telegram bot, autonomous agents, external scripts) consume these standardized APIs.

---

## 3. Core Development Guardrails

1. **Minimal Code & Human Standard**:
   - Write the absolute minimum lines required. If it can be done in 50 lines, do not write 500.
   - Avoid boilerplate, unnecessary abstractions, and redundant comments.
   - Prefer editing existing files over creating new ones.
2. **Strategy Integrity**:
   - Strategies are JSON-configured in `backend/strategies/`. Never hardcode rules in Python or mutate strategy JSONs without explicit user consent.
3. **High Liquidity Standards**:
   - Always filter for tradeable stocks (>100k volume, >500cr market cap) unless explicitly instructed.
4. **Strict Risk Controls**:
   - Respect stop-loss limits, capital allocation, max drawdown, and circuit breaker logic.
5. **Verifiable Work**:
   - Every change must be verified by running the tests or hitting the live/mock endpoints.
