#!/usr/bin/env python3
"""
Workflow Report Generator for Smart Advice Trading Orchestrator
==============================================================
Generates a complete, human-readable manual workflow report explaining
how `main_orchestrator.py` selects, filters, gates, scores, and executes
stocks based dynamically on any strategy configuration JSON file.

Usage:
  python backend/scripts/generate_workflow_report.py
  python backend/scripts/generate_workflow_report.py --strategy swing_trading.json
  python backend/scripts/generate_workflow_report.py --strategy momentum_trading.json --symbol RELIANCE
  python backend/scripts/generate_workflow_report.py --all
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure backend root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.filter_translator import FilterTranslator  # noqa: E402


def load_strategy_json(strategy_identifier: str) -> Dict[str, Any]:
    """Load a strategy JSON file by relative name or path."""
    strategies_dir = BACKEND_DIR / "strategies"

    # 1. Check direct path
    candidate_path = Path(strategy_identifier)
    if candidate_path.is_file():
        with open(candidate_path, "r") as f:
            return json.load(f)

    # 2. Check within backend/strategies/
    cand2 = strategies_dir / strategy_identifier
    if cand2.is_file():
        with open(cand2, "r") as f:
            return json.load(f)

    # 3. Check with .json extension added
    cand3 = strategies_dir / f"{strategy_identifier}.json"
    if cand3.is_file():
        with open(cand3, "r") as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Strategy file not found for identifier: '{strategy_identifier}' (searched in {strategies_dir})"
    )


def list_available_strategies() -> List[str]:
    """Return all available strategy JSON file names in backend/strategies/."""
    strategies_dir = BACKEND_DIR / "strategies"
    if not strategies_dir.exists():
        return []
    return [p.name for p in strategies_dir.glob("*.json")]


def format_filter_rule(f: Dict[str, Any]) -> str:
    """Format a single stock_filter dictionary into human-readable text."""
    ftype = f.get("type", "unknown")
    op = f.get("op", "")

    if ftype == "price":
        if op == "between":
            return f"Share Price: ₹{f.get('min', 0)} to ₹{f.get('max', 0)}"
        return f"Share Price {op} ₹{f.get('value', 0)}"

    elif ftype == "volume":
        return f"Daily Volume {op} {f.get('value', 0):,}"

    elif ftype == "market_cap":
        return f"Market Cap {op} ₹{f.get('value', 0):,} Cr"

    elif ftype == "rsi":
        period = f.get("period", 14)
        if op == "between":
            return f"RSI({period}): between {f.get('min')} and {f.get('max')}"
        return f"RSI({period}) {op} {f.get('value')}"

    elif ftype == "moving_average":
        kind = f.get("kind", "SMA").upper()
        period = f.get("period", 20)
        target = f.get("target", "close")
        if op == "monitor":
            return f"Monitor {kind}({period}) vs {target} (Local evaluation)"
        return f"Current {target} {op} {kind}({period})"

    elif ftype == "price_distance_sma":
        period = f.get("period", 20)
        max_dist = f.get("max_distance_pct", 5.0)
        return f"Price within ±{max_dist}% of SMA({period})"

    elif ftype == "volume_spike_lookup":
        lookback = f.get("lookback_days", 10)
        mult = f.get("multiplier", 1.2)
        ma_p = f.get("ma_period", 50)
        return f"Volume Spike: At least 1 day in last {lookback} days had Volume > {mult}x SMA({ma_p}) Volume"

    return f"{ftype} ({op}): {f}"


def build_workflow_markdown(strategy: Dict[str, Any], audit_result: Optional[Dict[str, Any]] = None) -> str:
    """Generate the full comprehensive Markdown workflow report."""
    strat_name = strategy.get("name", "Unnamed_Strategy")
    description = strategy.get("description", "No description provided.")
    ana_cfg = strategy.get("analysis_config", {})
    weights = strategy.get("analysis_weights", {})
    thresholds = strategy.get("recommendation_thresholds", {})
    stock_filters = strategy.get("stock_filters", [])
    gates = strategy.get("swing_trading_gates", {})
    entry_patterns = strategy.get("entry_patterns", [])
    strategy_indicators = strategy.get("strategy_config", {})
    fundamental_cfg = strategy.get("fundamental_config", {})
    risk_cfg = strategy.get("risk_management", {})
    exit_rules = strategy.get("exit_rules", {})
    pyramiding = strategy.get("pyramiding", {})
    regime_cfg = strategy.get("market_regime_config", {})
    breadth_cfg = strategy.get("market_breadth_filter", {})
    options_cfg = strategy.get("options_oi_config", {})

    # Generate Screener Query via FilterTranslator
    try:
        screener_clause = FilterTranslator.translate_to_scan_clause(stock_filters)
    except Exception as e:
        screener_clause = f"Error translating filters: {e}"

    md = []
    md.append("# 📊 Stock Selection & Filtering Workflow Report")
    md.append(f"**Strategy**: `{strat_name}`  ")
    md.append(f"**Generated Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    md.append(f"**Description**: {description}  ")
    md.append("\n---\n")

    # 1. Architecture Flowchart
    md.append("## 1. End-to-End Orchestrator Flowchart")
    md.append("The trading cycle in `main_orchestrator.py` follows a strict multi-phase pipeline:")
    md.append("```mermaid")
    md.append("flowchart TD")
    md.append("    Start([Trading Cycle Start]) --> Phase0[Phase 0: Settle T+1 Funds & Clean Positions]")
    md.append("    Phase0 --> Phase1[Phase 1: Portfolio Monitor - Trail Stops & Process Exits]")
    md.append("    Phase1 --> Phase1b[Phase 1b: ExecutionEngine Manage Advanced Exits]")
    md.append("    Phase1b --> Phase2[Phase 2: Stock Analysis Pipeline]")
    md.append("    ")
    md.append("    subgraph Phase 2: Stock Analysis Pipeline")
    md.append("        Phase2 --> MacroCheck{Macro Regime Check: ^NSEI > SMA 200?}")
    md.append("        MacroCheck -- Fail & Pause Enabled --> HaltMacro[Halt Strategy Scan]")
    md.append("        MacroCheck -- Pass or Non-blocking --> Discovery[Step A: Screener API Discovery]")
    md.append("        Discovery --> ScreenerFilters[Apply stock_filters: Price, Vol, MCap, RSI, SMA]")
    md.append("        ScreenerFilters --> CandleFetch[Step B: Fetch Historical Daily Candles]")
    md.append("        CandleFetch --> GateCheck{Step C: Hard Swing Gates?}")
    md.append("        GateCheck -- Fail --> DropStock[Reject Candidate]")
    md.append("        GateCheck -- Pass --> TechScore[Step D: Technical Pattern & Signal Scoring]")
    md.append("        TechScore --> FundScore[Step E: Fundamental & Smart Money Scoring]")
    md.append("        FundScore --> CombScore[Step F: Weighted Combination & Floor Checks]")
    md.append("        CombScore --> RRCheck{Step G: Risk-to-Reward >= Min RR?}")
    md.append("        RRCheck -- No --> RejectRR[Reject: Poor R:R]")
    md.append("        RRCheck -- Yes --> RecList[Save to MongoDB recommended_shares]")
    md.append("    end")
    md.append("    ")
    md.append("    RecList --> Phase3[Phase 3: Execution Engine]")
    md.append("    subgraph Phase 3: Execution & Portfolio Allocation")
    md.append("        Phase3 --> SlotCheck{Portfolio Full? < max_positions}")
    md.append("        SlotCheck -- No Slots --> SkipBuy[Skip Buys]")
    md.append("        SlotCheck -- Has Slots --> DDCheck{Drawdown Circuit Breaker?}")
    md.append("        DDCheck -- DD >= Threshold --> PauseBuys[Pause Buys]")
    md.append("        DDCheck -- Normal --> BreadthCheck{Market Breadth >= Threshold?}")
    md.append("        BreadthCheck -- Weak Breadth --> PauseBreadth[Pause Buys: Weak Breadth]")
    md.append("        BreadthCheck -- Healthy --> SortRank[Sort by Multi-Factor Composite Score]")
    md.append("        SortRank --> SizePos[Risk-Based Position Sizing]")
    md.append("        SizePos --> ExecBuy[Execute BUY Orders]")
    md.append("    end")
    md.append("    ")
    md.append("    ExecBuy --> Phase4[Phase 4: Exit Confirmations & Telegram Summary]")
    md.append("```")
    md.append("\n---\n")

    # 2. Phase-by-Phase Detailed Breakdown
    md.append("## 2. Detailed Step-by-Step Stock Selection Workflow")

    # Phase 0 & 1
    md.append("### Phase 0 & 1: Cash Settlement & Position Maintenance")
    md.append(
        "- **T+1 Cash Settlement**: `settle_pending_funds()` checks if sale proceeds from previous trades have settled (T+1 trading day in NSE). Unsettled funds cannot be used for new stock purchases."
    )
    md.append(
        "- **Trailing Stop & Exit Monitor**: `PortfolioMonitor().monitor_all_positions()` scans all open positions. If stop-loss or targets are touched, or trailing criteria met, exit signals are raised."
    )
    md.append(
        "- **Exit Confirmation**: In live/paper mode, prompts for exit verification so released capital immediately returns to available balance."
    )
    md.append("")

    # Phase 2 - Step 1: Macro Regime
    md.append("### Step 1: Global Macro & Market Regime Check")
    if ana_cfg.get("market_regime_detection", True):
        idx = regime_cfg.get("index", "^NSEI")
        rule = regime_cfg.get("bull_market_rule", "latest close > sma(200)")
        pause = regime_cfg.get("pause_buying_if_bearish", True)
        md.append(f"- **Benchmark Index**: `{idx}` (Nifty 50)")
        md.append(f"- **Bull Market Condition**: `{rule}`")
        md.append(
            f"- **Behavior if Bearish**: `{'PAUSE NEW BUYS (Skip Strategy)' if pause else 'Proceed with cautious risk sizing'}`"
        )
    else:
        md.append("- **Market Regime Check**: Disabled in strategy configuration.")
    md.append("")

    # Phase 2 - Step 2: Universe Discovery & Screener Filters
    md.append("### Step 2: Universe Discovery & Screener Filtering (`stock_filters`)")
    md.append(
        "The system first translates the JSON `stock_filters` into a fast screener scan clause to filter down the entire universe (2,000+ NSE stocks) into pre-qualified candidates:"
    )
    md.append("")
    md.append("#### Generated Screener Query:")
    md.append(f"```text\n{screener_clause}\n```")
    md.append("")
    md.append("#### Filter Rules Applied:")
    md.append("| Filter Type | Exact Rule / Threshold in Strategy JSON | Purpose |")
    md.append("|:---|:---|:---|")
    for f in stock_filters:
        rule_desc = format_filter_rule(f)
        ftype = f.get("type", "")
        purpose = "Liquidity / Trading viability"
        if ftype == "price":
            purpose = "Eliminate penny stocks and ultra-high illiquid names"
        elif ftype == "volume":
            purpose = "Ensure adequate daily trading liquidity"
        elif ftype == "market_cap":
            purpose = "Target institutional-grade mid to large cap companies"
        elif ftype == "rsi":
            purpose = "Avoid deeply overbought or oversold falling knives"
        elif ftype == "moving_average":
            purpose = "Confirm price is above key moving averages (bullish trend)"
        elif ftype == "volume_spike_lookup":
            purpose = "Detect institutional accumulation surge in past days"
        md.append(f"| `{ftype}` | {rule_desc} | {purpose} |")
    md.append("")

    # Phase 2 - Step 3: Hard Swing Trading Gates
    md.append("### Step 3: Hard Swing Trading Gatekeeper Checks (`swing_trading_gates`)")
    md.append(
        "Every candidate that passes the screener must pass **ALL** enabled gates in `SwingTradingSignalAnalyzer`. A failure at any single gate immediately eliminates the stock or suppresses its score to 0.10 max."
    )
    md.append("")

    t_gate = gates.get("TREND_GATE", {})
    if t_gate.get("enabled", False):
        p = t_gate.get("params", {})
        md.append("#### 1. 📈 Trend Gate (`TREND_GATE`: Enabled)")
        md.append(f"- **ADX Minimum**: Must be `> {p.get('adx_min', 20)}` (confirms strong directional momentum).")
        md.append("- **Directional Movement (+DI / -DI)**: Requires `+DI > -DI` (buyers in control).")
        md.append(f"- **Price vs SMA**: Close must be `> SMA({p.get('sma_period', 50)})` and `> SMA(200)`.")
        md.append(
            f"- **SMA Stack**: `{'Requires SMA(50) > SMA(150) > SMA(200)' if p.get('require_sma_stack') else 'Not strictly required'}`."
        )
    else:
        md.append("- **Trend Gate**: Disabled.")

    vol_gate = gates.get("VOLUME_GATE", {})
    if vol_gate.get("enabled", False):
        p = vol_gate.get("params", {})
        md.append("#### 2. 📊 Volume Gate (`VOLUME_GATE`: Enabled)")
        md.append(
            f"- **Volume Ratio**: Latest volume must be `>= {p.get('min_volume_ratio', 0.8)}x` the 20-day average volume."
        )
        if p.get("obv_trend_lookback"):
            md.append(
                f"- **OBV Trend**: On-Balance-Volume slope over last `{p.get('obv_trend_lookback')}` bars must be `> 0` (accumulation confirmation)."
            )
    else:
        md.append("- **Volume Gate**: Disabled.")

    vola_gate = gates.get("VOLATILITY_GATE", {})
    if vola_gate.get("enabled", False):
        p = vola_gate.get("params", {})
        md.append("#### 3. ⚡ Volatility Gate (`VOLATILITY_GATE`: Enabled)")
        md.append(
            f"- **ATR Percentile Range**: Current 14-day ATR must sit between `{p.get('min_percentile', 5)}th` and `{p.get('max_percentile', 90)}th` percentile over `{p.get('lookback_days', 100)}` days (filters out frozen dead stocks and extreme panic-volatility stocks)."
        )
    else:
        md.append("- **Volatility Gate**: Disabled.")

    mtf_gate = gates.get("MTF_GATE", {})
    if mtf_gate.get("enabled", False):
        p = mtf_gate.get("params", {})
        md.append("#### 4. ⏱ Multi-Timeframe Gate (`MTF_GATE`: Enabled)")
        md.append(
            f"- Weekly trend check: `{p.get('weekly_trend_check')}` | Weekly SMA({p.get('weekly_sma_fast')}) > SMA({p.get('weekly_sma_slow')}) | Weekly RSI >= {p.get('rsi_alignment_min')}."
        )
    md.append("")

    # Phase 2 - Step 4: Technical Pattern & Signal Evaluation
    md.append("### Step 4: Technical Patterns & Strategy Signals (`strategy_config` & `entry_patterns`)")
    md.append("Stocks passing the hard gates are evaluated for specific technical patterns and setups:")
    md.append("")
    md.append("#### Active Technical Indicators:")
    md.append("| Indicator Module | Status | Role / Parameters |")
    md.append("|:---|:---|:---|")
    for ind_name, ind_cfg in strategy_indicators.items():
        enabled = ind_cfg.get("enabled", False)
        is_bonus = ind_cfg.get("is_bonus", False)
        status_str = (
            "✅ Enabled (Bonus)" if (enabled and is_bonus) else ("✅ Enabled (Core)" if enabled else "❌ Disabled")
        )
        details = ", ".join([f"{k}={v}" for k, v in ind_cfg.items() if k not in ("enabled", "is_bonus")])
        md.append(f"| `{ind_name}` | {status_str} | {details} |")
    md.append("")

    if entry_patterns:
        md.append("#### Active Swing Entry Patterns:")
        md.append("| Pattern Name | Status | Setup Trigger Logic |")
        md.append("|:---|:---|:---|")
        for ep in entry_patterns:
            name = ep.get("name", "unnamed")
            en = ep.get("enabled", False)
            status = "✅ Active" if en else "⚪ Inactive"
            cfg_items = ", ".join([f"{k}: {v}" for k, v in ep.items() if k not in ("name", "enabled")])
            md.append(f"| `{name}` | {status} | {cfg_items} |")
        md.append("")

    # Phase 2 - Step 5: Fundamental & Smart Money Checks
    md.append("### Step 5: Fundamental Quality & Smart Money Tracking")
    if ana_cfg.get("fundamental_analysis", True):
        md.append("Fundamental health checks from `fundamental_config`:")
        md.append(f"- **Min ROCE**: `>= {fundamental_cfg.get('min_roce', 15.0)}%`")
        md.append(f"- **Max Debt-to-Equity**: `<= {fundamental_cfg.get('max_debt_to_equity', 1.0)}`")
        md.append(f"- **Min 3-Year Profit Growth**: `>= {fundamental_cfg.get('min_profit_growth_3y', 10.0)}%`")
        md.append(f"- **Min 5-Year Profit Growth**: `>= {fundamental_cfg.get('min_profit_growth_5y', 10.0)}%`")
        md.append(f"- **Min Quarterly EPS Growth**: `>= {fundamental_cfg.get('min_quarterly_eps_growth', 0.1)}`")
    else:
        md.append("- **Fundamental Analysis**: Disabled in configuration.")

    # Smart Money & Options
    md.append("")
    md.append("Smart Money Boosters:")
    md.append("- **Delivery Volume Check**: If NSE delivery volume is `> 40%`, adds a `+0.05` technical score bonus.")
    if ana_cfg.get("options_oi", False) and options_cfg.get("enabled", False):
        md.append(
            f"- **Options Open Interest (PCR)**: If Put-Call Ratio >= `{options_cfg.get('pcr_bullish_threshold', 1.0)}`, adds a `+{options_cfg.get('weight', 0.15)}` technical score bonus."
        )
    md.append("")

    # Phase 2 - Step 6: Multi-Pillar Scoring & Thresholds
    md.append("### Step 6: Multi-Pillar Scoring, Floors, and Recommendation Thresholds")
    t_w = weights.get("technical", 0.65)
    f_w = weights.get("fundamental", 0.20)
    s_w = weights.get("sector", 0.15)
    sent_w = weights.get("sentiment", 0.00)

    md.append("The combined score is calculated using weighted pillar components:")
    md.append(
        "$$\\text{Combined Score} = (\\text{Technical} \\times "
        + f"{t_w:.2f}) + (\\text{{Fundamental}} \\times {f_w:.2f}) + (\\text{{Sector}} \\times {s_w:.2f}) + (\\text{{Sentiment}} \\times {sent_w:.2f})$$"
    )
    md.append("")
    md.append("#### Qualification Gates (Floors):")
    md.append(f"1. **Technical Floor**: Must have `Technical Score >= {thresholds.get('technical_minimum', 0.35)}`")
    md.append(
        f"2. **Fundamental Floor**: Must have `Fundamental Score >= {thresholds.get('fundamental_minimum', 0.10)}`"
    )
    md.append(f"3. **Buy Combined Threshold**: Must have `Combined Score >= {thresholds.get('buy_combined', 0.60)}`")
    md.append("4. **Risk-to-Reward Ratio (RR)**:")
    min_rr = thresholds.get("min_risk_reward_ratio", 1.2)
    md.append(
        f"   $$\\text{{RR Ratio}} = \\frac{{\\text{{Target Price}} - \\text{{Buy Price}}}}{{\\text{{Buy Price}} - \\text{{Stop Loss}}}} \\ge {min_rr}$$"
    )
    md.append(f"   *If $\\text{{RR}} < {min_rr}$, the recommendation is strictly rejected.*")
    md.append(
        f"5. **52-Week High Proximity**: Must be within `{thresholds.get('proximity_to_52_week_high_pct', 25.0)}%` of its 52-week high."
    )
    md.append("")

    # Multi-Factor Leadership Score
    md.append("#### Multi-Factor Momentum Leadership Score (Ranking Formula):")
    md.append("All qualifying recommendations are sorted so that true market leaders get capital first:")
    md.append(
        "$$\\text{Composite Score} = (\\text{3-Month Momentum} \\times 2.0) + \\left(\\frac{\\text{Close}}{\\text{52W High}}\\right) + \\text{Technical Score}$$"
    )
    md.append("- Stocks with highest `Composite Score` receive first priority in execution.")
    md.append("")

    # Phase 3: Execution & Risk Management
    md.append("### Phase 3: Portfolio Execution & Risk Gatekeepers")
    max_pos = risk_cfg.get("max_positions", 6)
    risk_per_trade = risk_cfg.get("risk_per_trade_pct", 2.0)
    max_pos_pct = risk_cfg.get("max_position_pct", 30.0)
    dd_pause = risk_cfg.get("drawdown_pause", {})

    md.append("Before `ExecutionEngine` buys any recommended share, three strict safety gatekeepers are enforced:")
    md.append(
        f"1. **Portfolio Capacity Check**: Maximum `{max_pos}` simultaneous positions. If current positions >= `{max_pos}`, no new buys occur."
    )

    if dd_pause.get("enabled", True):
        p_thresh = dd_pause.get("pause_threshold_pct", 8.5)
        md.append(
            f"2. **Drawdown Circuit Breaker**: If total portfolio drawdown from peak equity reaches `>= {p_thresh}%`, **ALL new purchases are paused** to protect capital."
        )
    else:
        md.append("2. **Drawdown Circuit Breaker**: Disabled.")

    if breadth_cfg.get("enabled", True):
        adv_pct = breadth_cfg.get("min_advance_pct", 35)
        sma_b = breadth_cfg.get("sma_period", 20)
        md.append(
            f"3. **Leading Market Breadth Filter**: If less than `{adv_pct}%` of stocks are trading above their `{sma_b}-day SMA`, new buys are paused (broad market weakness)."
        )
    else:
        md.append("3. **Market Breadth Filter**: Disabled.")

    md.append("")
    md.append("#### Position Sizing Formula:")
    md.append(f"- **Risk-Based Sizing**: Risk is limited to `{risk_per_trade}%` of total account equity per trade.")
    md.append(
        "  $$\\text{Quantity} = \\left\\lfloor \\frac{\\text{Equity} \\times "
        + f"{risk_per_trade / 100.0:.4f}"
        + "}{\\text{Buy Price} - \\text{Stop Loss}} \\right\\rfloor$$"
    )
    md.append(f"- **Maximum Capital Allocation per Stock**: Capped at `{max_pos_pct}%` of portfolio equity.")
    md.append(
        "- **Cash & Settlement Availability**: Trade cost (`Buy Price * Quantity`) must be `<=` currently settled available cash."
    )
    md.append("")

    # Exit Rules & Pyramiding
    md.append("### Exit Rules & Pyramiding")
    targets = exit_rules.get("targets", [])
    if targets:
        md.append("#### Profit Targets:")
        for t in targets:
            name = t.get("name", "Target")
            mult = t.get("atr_multiplier", 3.0)
            pct = t.get("sell_percentage", 0.5) * 100
            md.append(f"- **{name}**: Entry + `{mult}x ATR` (Sell `{pct:.0f}%` of position)")

    md.append(f"- **Initial Stop Loss**: Entry - `{exit_rules.get('atr_stop_multiplier', 2.4)}x ATR`")
    if exit_rules.get("breakeven_at_target_1"):
        md.append("- **Breakeven Rule**: Stop Loss automatically moves to Break-Even price as soon as Target 1 is hit.")
    if exit_rules.get("trail_stop_atr"):
        md.append(
            f"- **Trailing Stop**: Trails at `{exit_rules.get('trail_stop_atr')}x ATR` after gaining `{exit_rules.get('trail_min_gain_pct', 5.0)}%`."
        )

    if pyramiding.get("enabled", False):
        md.append("#### Pyramiding (Adding to Winners):")
        for step in pyramiding.get("steps", []):
            md.append(
                f"- **{step.get('name')}**: When price gains `+{step.get('trigger_step_atr')}x ATR`, add `{step.get('add_size_pct', 0.5) * 100:.0f}%` additional shares."
            )
    md.append("\n---\n")

    # 3. Quick Manual Verification Checklist
    md.append("## 3. Quick Manual Verification Checklist (How to Check Any Stock)")
    md.append(
        "To manually verify whether a stock like **TATASTEEL** or **RELIANCE** will be selected by `main_orchestrator`, check these 7 questions:"
    )
    md.append("")
    md.append("| Step | Manual Check Question | Required Condition |")
    md.append("|:---|:---|:---|")
    md.append("| **1. Macro** | Is Nifty 50 above its 200-day SMA? | Yes (`^NSEI > SMA 200`) |")
    min_p = strategy.get("stock_filters", [{}])[0].get("min", 50) if strategy.get("stock_filters") else 50
    max_p = strategy.get("stock_filters", [{}])[0].get("max", 8000) if strategy.get("stock_filters") else 8000
    md.append(
        f"| **2. Price & Liquidity** | Is price within bounds and volume high? | Price ₹{min_p}–₹{max_p}, Vol > 1.5L |"
    )
    md.append(
        "| **3. Trend Gate** | Is ADX strong with positive directional bias? | ADX(14) > 20 and +DI > -DI and Price > 50 & 200 SMA |"
    )
    md.append(
        "| **4. Volatility Gate** | Is ATR within reasonable historical bounds? | 14-ATR between 5th and 90th percentile |"
    )
    md.append(
        f"| **5. Technical Score** | Does it pass core patterns & floor threshold? | Tech Score >= {thresholds.get('technical_minimum', 0.35)} |"
    )
    md.append(
        f"| **6. Risk-to-Reward** | Is potential upside significantly higher than risk? | Target / Stop-Loss RR >= {thresholds.get('min_risk_reward_ratio', 1.2)} |"
    )
    md.append(
        f"| **7. Portfolio Slot** | Are open positions less than maximum allowed? | Open Positions < {max_pos} and Drawdown < {dd_pause.get('pause_threshold_pct', 8.5)}% |"
    )
    md.append("\n---\n")

    # 4. Optional Live Audit Table (if supplied)
    if audit_result:
        md.append("## 4. Live Stock Audit Demonstration")
        md.append(f"**Audited Symbol**: `{audit_result.get('symbol')}`  ")
        md.append(f"**Company Name**: `{audit_result.get('company_name', audit_result.get('symbol'))}`  ")
        md.append(
            f"**Final Recommendation**: **`{audit_result.get('recommendation_strength', 'HOLD')}`** (Score: `{audit_result.get('combined_score', 0):.2f}`)  "
        )
        md.append("")
        md.append("### Detailed Audit Steps:")
        md.append("| Step | Check Description | Value / Details | Status |")
        md.append("|:---|:---|:---|:---|")

        steps = audit_result.get("audit_log", {}).get("steps", [])
        for s in steps:
            st_name = s.get("step", "")
            st_status = s.get("status", "")
            st_reason = s.get("reason", "")
            icon = "✅ PASS" if st_status == "PASS" else ("❌ FAIL" if st_status == "FAIL" else f"⚪ {st_status}")
            md.append(f"| `{st_name}` | {st_reason if st_reason else 'Rule evaluated'} | {st_reason} | {icon} |")
        md.append("\n---\n")

    md.append("*Report generated automatically by `backend/scripts/generate_workflow_report.py`*")
    return "\n".join(md)


def run_live_audit(symbol: str, strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch historical data for a symbol and run StockAnalyzer to get an audit trace."""
    try:
        from scripts.analyzer import StockAnalyzer
        from scripts.data_fetcher import get_benchmark_data, get_historical_data

        thresholds = strategy.get("recommendation_thresholds", {})
        backtest_period = thresholds.get("backtest_period", "2y")

        print(f"[*] Fetching historical candle data for live audit: {symbol} ({backtest_period})...")
        hist = get_historical_data(symbol, backtest_period)
        if hist.empty:
            print(f"[!] Warning: Could not fetch candle data for {symbol}.")
            return None

        print("[*] Fetching benchmark data (^NSEI)...")
        bench = get_benchmark_data(backtest_period)

        analyzer = StockAnalyzer()
        res = analyzer.analyze_stock_with_data(symbol, symbol, hist, strategy, index_data=bench)
        return res
    except Exception as e:
        print(f"[!] Live audit error for {symbol}: {e}")
        return None


def print_cli_summary(strategy: Dict[str, Any], report_path: Path):
    """Print an aesthetic, readable terminal summary of the workflow."""
    name = strategy.get("name", "Unnamed")
    thresh = strategy.get("recommendation_thresholds", {})
    weights = strategy.get("analysis_weights", {})
    risk = strategy.get("risk_management", {})
    gates = strategy.get("swing_trading_gates", {})

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    print(f"\n{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"{BOLD}{GREEN}  🎯 WORKFLOW & STOCK SELECTION REPORT: {name}{RESET}")
    print(f"{BOLD}{CYAN}{'='*70}{RESET}")
    print(f"  {BOLD}Strategy File:{RESET} {name}.json")
    print(f"  {BOLD}Saved Report:{RESET}  {report_path}")
    print(f"{CYAN}{'-'*70}{RESET}")

    print(f"{BOLD}  [1] UNIVERSE & SCREENER FILTERS{RESET}")
    for f in strategy.get("stock_filters", [])[:5]:
        print(f"      • {format_filter_rule(f)}")
    if len(strategy.get("stock_filters", [])) > 5:
        print(f"      • ... and {len(strategy.get("stock_filters", [])) - 5} more filters")

    print(f"\n{BOLD}  [2] HARD SWING TRADING GATES{RESET}")
    for g_name, g_val in gates.items():
        status = f"{GREEN}ACTIVE{RESET}" if g_val.get("enabled") else f"{YELLOW}OFF{RESET}"
        print(f"      • {g_name:18} : {status}")

    print(f"\n{BOLD}  [3] SCORING WEIGHTS & FLOORS{RESET}")
    print(
        f"      • Weights: Tech {weights.get('technical',0)*100:.0f}%, Fund {weights.get('fundamental',0)*100:.0f}%, Sector {weights.get('sector',0)*100:.0f}%"
    )
    print(
        f"      • Floors : Tech >= {thresh.get('technical_minimum',0)}, Fund >= {thresh.get('fundamental_minimum',0)}, Buy Score >= {thresh.get('buy_combined',0)}"
    )
    print(f"      • Risk/Reward Ratio: >= {thresh.get('min_risk_reward_ratio', 1.2)}x")

    print(f"\n{BOLD}  [4] EXECUTION & RISK CONSTRAINTS{RESET}")
    print(f"      • Max Positions     : {risk.get('max_positions', 6)}")
    print(f"      • Risk Per Trade    : {risk.get('risk_per_trade_pct', 2.0)}% of equity")
    print(
        f"      • Drawdown Circuit  : Pause buys if DD >= {risk.get('drawdown_pause', {}).get('pause_threshold_pct', 8.5)}%"
    )
    print(f"{CYAN}{'='*70}{RESET}")
    print(f"  {BOLD}Open the Markdown file for the complete manual verification guide:{RESET}")
    print(f"  👉 {GREEN}{report_path}{RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate detailed manual workflow and stock selection report.")
    parser.add_argument(
        "--strategy",
        "-s",
        type=str,
        default="swing_trading.json",
        help="Strategy JSON file name (e.g. swing_trading.json, momentum_trading.json, or 'all').",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Optional NSE symbol (e.g. RELIANCE, TCS) to run a live step-by-step audit demonstration.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(BACKEND_DIR / "reports"),
        help="Directory to save generated Markdown reports.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate workflow reports for all strategy files in backend/strategies/.",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_strategies = []
    if args.all or args.strategy.lower() == "all":
        target_strategies = list_available_strategies()
    else:
        target_strategies = [args.strategy]

    if not target_strategies:
        print("[!] No strategy files found.")
        sys.exit(1)

    print(f"\n[*] Generating Stock Selection Workflow Report for: {', '.join(target_strategies)}")

    for strat_file in target_strategies:
        try:
            strategy_data = load_strategy_json(strat_file)
            strat_name = strategy_data.get("name", Path(strat_file).stem)

            audit_res = None
            if args.symbol:
                audit_res = run_live_audit(args.symbol.upper(), strategy_data)

            report_md = build_workflow_markdown(strategy_data, audit_result=audit_res)

            output_file = out_dir / f"workflow_{strat_name.lower()}.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_md)

            print_cli_summary(strategy_data, output_file)

        except Exception as e:
            print(f"[!] Failed to generate report for '{strat_file}': {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
