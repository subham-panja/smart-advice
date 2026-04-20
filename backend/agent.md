# Agent Instructions & Project Context

This file contains persistent rules and context that every agent (AI assistant) should follow when working on this repository.

## Core Principles
1. **Institutional Grade**: All trading logic must maintain pre-defined risk parameters (Market Cap > 5000Cr, Delivery > 55%).
2. **Performance First**: Favor vectorized operations (numpy/pandas) over iterative loops to protect Apple Silicon hardware.
3. **Macro Awareness**: Never suggest or execute individual stock analysis without verifying the NIFTY 50 macro-gate first.

## Working with this Pipeline
- **Primary Script**: `run_analysis.py`
- **Configuration**: `backend/config.py`
- **Strategies**: `backend/scripts/strategies/`
- **Database**: MongoDB (`super_advice`)

## Instruction Checklist for Agents
- [ ] Read `backend/config.py` before suggesting any parameter changes.
- [ ] Check Nifty-50 gate logic in `run_analysis.py` before modifying the pipeline.
- [ ] Ensure all new strategies are vectorized.
