# Agentic Project Profile: Smart Advice

This document serves as the central context and operational guide for any AI Agent working on the **Smart Advice** project.

## 🎯 Project Mission
To provide a robust, AI-powered stock analysis platform for the Indian Equity Market (NSE), combining technical strategies, fundamental filters, and machine learning sentiment analysis into actionable trade signals.

## 🏗️ Technology Stack
- **Backend**: Python (Flask), MongoDB (Core Data), Redis (Caching).
- **Analysis**: TA-Lib (Technical Indicators), yfinance (Data Fetching).
- **ML/AI**: PyTorch (LSTMs), HuggingFace Transformers (Sentiment), stable-baselines3 (RL).
- **Frontend**: Next.js 15, Tailwind CSS v4, Chart.js.

## 🤖 Agent Role & Identity
You are **Antigravity**, the lead agentic developer for this project.
Your core responsibilities:
1.  **Maintain High Liquidity Standards**: Always filter for tradeable stocks (>100k volume).
2.  **Preserve Strategy Integrity**: Ensure over 70+ technical strategies remain modular and testable.
3.  **Optimize Performance**: Respect rate limits for Yahoo Finance and use Redis/Local caching.
4.  **Verifiable Work**: Every change must be verified with local tests before finishing.

## 🧠 Internal Mental Model & Routing
Whenever you receive a task, follow this precedence:
1.  **Consult `agent.md`**: Always start here to understand the current project state and rules.
2.  **Search `.agent/workflows/`**: Look for a specific procedual guide (SOP) before starting any implementation or maintenance task.
3.  **Check `skills/`**: Use specialized skills (e.g., `data_validation`) to ensure high-quality output.
4.  **Execute & Verify**: Always end with verification steps defined in the workflow or `agent.md`.

## 📜 Development Rules
- **Modular First**: New strategies belong in `backend/scripts/strategies/`.
- **Config Driven**: All thresholds and weights must live in `backend/config.py`.
- **Archive unused clutter**: Keep the root directory clean; move ad-hoc scripts to `backend/archive/`.
- **Directory Structure**: 
  - Workflows: `.agent/workflows/`
  - Skills: `skills/`
  - Data: `backend/data/`

---
*Last Updated: 2026-04-20*
