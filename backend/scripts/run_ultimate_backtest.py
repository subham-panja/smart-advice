#!/usr/bin/env python3
"""
Ultimate Backtest Runner
=========================

Runs all validation phases from the ultimate_backtest.md plan:
1. Historical backtest with realistic execution costs (gap risk, STT, slippage)
2. Walk-forward Monte Carlo analysis
3. Stress tests (regime, param sensitivity, cost sensitivity)
4. Trade diagnostics (from MongoDB)
5. Statistical validation (DSR, Monte Carlo permutation, MLRS)
6. Composite confidence score

Usage:
    cd backend
    python scripts/run_ultimate_backtest.py
    python scripts/run_ultimate_backtest.py --strategy Swing_Trading --max-stocks 50 --months 120 --telegram
    python scripts/run_ultimate_backtest.py --skip-wf --skip-stress  # Quick run (historical only)
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from scripts.run_portfolio_backtest import (
    _prepare_index_data,
    fetch_symbols_data,
    run_walk_forward_backtest,
)
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.strategy_loader import StrategyLoader


class Timer:
    """Track elapsed time and estimate remaining time per phase."""

    def __init__(self):
        self.start = time.time()
        self.phase_times: list[dict] = []

    def elapsed(self) -> str:
        s = time.time() - self.start
        if s < 60:
            return f"{s:.1f}s"
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {sec}s"
        return f"{m}m {sec}s"

    def phase_start(self, name: str):
        print(f"\n⏱️  [{self.elapsed()}] Starting: {name}")
        self.phase_times.append({"name": name, "start": time.time()})

    def phase_end(self):
        if self.phase_times:
            p = self.phase_times[-1]
            took = time.time() - p["start"]
            if took < 60:
                print(f"⏱️  [{self.elapsed()}] Completed: {p['name']} ({took:.1f}s)")
            else:
                m, s = divmod(int(took), 60)
                h, m = divmod(m, 60)
                if h:
                    print(f"⏱️  [{self.elapsed()}] Completed: {p['name']} ({h}h {m}m {s}s)")
                else:
                    print(f"⏱️  [{self.elapsed()}] Completed: {p['name']} ({m}m {s}s)")

    def estimate_remaining(self, phases_done: int, total_phases: int) -> str:
        if phases_done == 0:
            return "calculating..."
        elapsed = time.time() - self.start
        avg = elapsed / phases_done
        remaining = avg * (total_phases - phases_done)
        if remaining < 60:
            return f"~{remaining:.0f}s remaining"
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        if h:
            return f"~{h}h {m}m remaining"
        return f"~{m}m {s}s remaining"


def _send_telegram(message: str):
    """Send message to Telegram."""
    import requests

    tg = getattr(config, "TELEGRAM_CONFIG", {})
    if not tg.get("enabled", False):
        return
    token = tg.get("bot_token", "")
    chat_ids = tg.get("allowed_user_ids", [])
    if not token or not chat_ids:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
            if resp.status_code == 200:
                print(f"Telegram sent (chat_id={chat_id})")
        except Exception:
            pass


def run_historical_with_realistic_costs(
    strategy_name: str,
    end_date: str,
    months: int,
    period: str = "5y",
) -> dict:
    """Phase 1: Historical backtest with execution realism.

    Runs the strategy over real historical data with:
    - Gap risk on entry (overnight price jumps)
    - STT, stamp duty, SEBI charges
    - Slippage on entry/exit
    - Brokerage charges
    """
    t_phase = time.time()  # Track this phase's own timing
    print(f"\n{'='*70}")
    print("PHASE 1: HISTORICAL BACKTEST (REALISTIC COSTS)")
    print(f"{'='*70}\n")

    # Load strategy config from YAML/DB
    print("[1/6] Loading strategy config...")
    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy '{strategy_name}' not found")

    # Enable market regime detection (bull/bear/sideways filter)
    strategy.setdefault("analysis_config", {})["market_regime_detection"] = True

    # ── Use point-in-time universe to eliminate survivorship bias ────────────
    # For backtests starting before 2025, use stocks that were actually in the
    # index at that time — not the 2026 winners we can see with hindsight.
    print("[2/6] Building point-in-time universe...")
    from scripts.universe_builder import get_point_in_time_symbols

    pit_start = (pd.Timestamp(end_date) - pd.DateOffset(months=months)).tz_localize(None)
    symbols = get_point_in_time_symbols(pit_start, strategy, max_stocks=200)
    print(f"  → Point-in-time universe ({pit_start.year}): {len(symbols)} stocks")

    # If we got the fallback live universe, also apply market cap pre-filter
    if pit_start.year >= 2025:
        min_cap = next(
            (
                f.get("value")
                for f in strategy.get("stock_filters", [])
                if f.get("type") == "market_cap" and f.get("op") == ">"
            ),
            None,
        )
        if min_cap is not None:
            print(f"  → Pre-filtering by market cap > {min_cap}...")
            from scripts.data_fetcher import get_market_caps

            mc_cache = get_market_caps(list(symbols.keys()))
            symbols = {sym: symbols[sym] for sym in symbols.keys() if mc_cache.get(sym, 0) > min_cap}
            print(f"  → Filtered: {len(symbols)} stocks remaining")

    # Scanned symbols returned in results["_scanned_symbols"] for reuse

    print(f"[3/6] Fetching {period} historical OHLCV data...")
    symbols_data = fetch_symbols_data(symbols, period=period, verbose=False)

    # Fetch NIFTY 50 index data for regime detection warmup
    index_data = _prepare_index_data(strategy, symbols_data, period)

    # Filter out stocks with insufficient history (< 250 trading days)
    print("[4/6] Filtering stocks with insufficient data...")
    MIN_DAYS = 250
    excluded = 0
    for sym, df in list(symbols_data.items()):
        if len(df) < MIN_DAYS:
            excluded += 1
            del symbols_data[sym]
    if excluded:
        print(f"  → Excluded {excluded} stocks with < {MIN_DAYS} days")
    print(f"  → Remaining: {len(symbols_data)} stocks")

    # Calculate simulation date range (tz-naive for universal compatibility)
    sim_end = pd.Timestamp(end_date).tz_localize(None)
    sim_start = (sim_end - pd.DateOffset(months=months)).tz_localize(None)

    # Advance sim_start if needed for regime warmup (needs 250 days of index data)
    if index_data is not None:
        stock_only = {
            k: v for k, v in symbols_data.items() if k != strategy.get("market_regime_config", {}).get("index", "^NSEI")
        }
        all_sets = [
            set(df.index.tz_localize(None) if df.index.tz is not None else df.index) for df in stock_only.values()
        ]
        if all_sets:
            union_dates = sorted(set.union(*all_sets))
            for d in union_dates:
                if len(index_data.loc[:d]) >= 250:
                    if sim_start < d:
                        print(f"  → Advancing sim_start to {d.date()} for regime warmup (need 250 index days)")
                        sim_start = d
                    break

    print(f"[5/6] Simulation range: {sim_start.date()} → {sim_end.date()} ({months} months)")

    # Pre-compute all indicators with vectorbt (one-time vectorized batch across all symbols × dates)
    # This replaces ~26 per-symbol per-date TA-Lib calls with O(1) lookups
    print("[6/6] Pre-computing indicators with vectorbt (batch across all symbols)...")
    from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

    indicators = compute_all_indicators(symbols_data, strategy_config=strategy)
    store = IndicatorStore(indicators)
    print(
        f"  → Pre-computed {len(indicators.symbols)} stocks × {len(indicators.dates)} dates = {len(indicators.symbols) * len(indicators.dates):,} indicator values"
    )

    # Compute per-date stock prefilter
    from scripts.vectorbt_signal_generator import compute_stock_prefilter

    stock_prefilter = compute_stock_prefilter(indicators, strategy)
    n_pass = int(stock_prefilter.iloc[-1].sum()) if len(stock_prefilter) > 0 else 0
    print(f"  → Stock prefilter: {n_pass} stocks pass on latest date")

    # Pre-compute ALL signals ONCE (Phase 1 optimization: signal pre-computation)
    print("  → Pre-computing signals for all eligible symbols...")
    from scripts.run_portfolio_backtest import precompute_full_signals

    precomputed_signals = precompute_full_signals(symbols_data, strategy, indicators, stock_prefilter, num_workers=4)
    print(f"  → Signals pre-computed: {len(precomputed_signals)} symbols with buy signals")

    # Create simulation engine with vectorbt store and index data override
    engine = PortfolioBacktestSession(strategy_config=strategy)
    engine.set_indicator_store(store)
    engine._stock_prefilter = stock_prefilter
    if index_data is not None:
        engine._index_data_override = index_data

    print(f"  → Execution realism: gap_risk={'ON' if engine.use_realistic_costs else 'OFF'}")
    print("  → IndicatorStore: ENABLED (O(1) lookups, no TA-Lib during simulation)")
    # Count trading days in simulation range
    if symbols_data:
        first_sym = next(iter(symbols_data))
        first_idx = symbols_data[first_sym].index
        if first_idx.tz is not None:
            first_idx = first_idx.tz_localize(None)
        num_days = len([d for d in first_idx if sim_start <= d <= sim_end])
    else:
        num_days = "?"
    print(f"\n▶ Running day-by-day simulation ({num_days} trading days) with pre-computed signals...")
    results = engine.run_with_signals(
        symbols_data,
        precomputed_signals,
        sim_start_date=sim_start,
        sim_end_date=sim_end,
    )
    print(f"  → Simulation done ({time.time() - t_phase:.1f}s total for Phase 1)")

    # Print summary
    print(f"\n{'='*60}")
    print("HISTORICAL RESULTS (REALISTIC)")
    print(f"{'='*60}")
    print(f"Date Range:        {results['date_range']['start_date']} -> {results['date_range']['end_date']}")
    print(f"Initial Capital:   Rs {results['initial_capital']:,.0f}")
    print(f"Final Value:       Rs {results['final_portfolio_value']:,.0f}")
    print(f"Total Return:      {results['total_return_pct']:+.2f}%")
    print(f"CAGR:              {results['cagr']:.2f}%")
    print(f"Max Drawdown:      {results['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio:      {results['sharpe_ratio']:.2f}")
    print(f"Total Trades:      {results['total_trades']}")
    print(f"Win Rate:          {results['win_rate']:.1f}%")
    print(f"Profit Factor:     {results['profit_factor']:.2f}")
    print(f"Excluded Stocks:   {excluded}")
    print(f"{'='*60}\n")

    results["_scanned_symbols"] = symbols
    results["_symbols_data"] = symbols_data
    results["_precomputed_signals"] = precomputed_signals
    results["_indicators"] = indicators
    results["_prefilter"] = stock_prefilter
    return results


def run_validation_phase(daily_snapshots: list) -> dict:
    """Phase 2: Statistical validation tests."""
    print(f"\n{'='*70}")
    print("PHASE 2: STATISTICAL VALIDATION")
    print(f"{'='*70}\n")

    from scripts.validation_tests import run_all_validations

    # Extract daily returns from snapshots
    daily_returns = []
    for i in range(1, len(daily_snapshots)):
        prev = daily_snapshots[i - 1]["portfolio_value"]
        curr = daily_snapshots[i]["portfolio_value"]
        daily_returns.append((curr - prev) / prev if prev > 0 else 0)

    if not daily_returns:
        return {"status": "error", "reason": "No daily returns available"}

    results = run_all_validations(daily_returns, n_trials=30, mc_sims=5000)

    # Print results
    dsr = results.get("dsr", {})
    print("Deflated Sharpe Ratio:")
    print(f"  Observed SR: {dsr.get('sr_observed', 'N/A')}")
    print(f"  DSR: {dsr.get('dsr', 'N/A')}")
    print(f"  Significant: {'YES' if dsr.get('significant') else 'NO'}")
    print(f"  Confidence: {dsr.get('confidence_pct', 'N/A')}%")

    mc = results.get("monte_carlo_permutation", {})
    print("\nMonte Carlo Permutation:")
    print(f"  Actual Sharpe: {mc.get('actual_sharpe', 'N/A')}")
    print(f"  Z-score: {mc.get('z_score', 'N/A')}")
    print(f"  p-value: {mc.get('p_value', 'N/A')}")
    print(f"  Significant: {'YES' if mc.get('significant') else 'NO'}")

    mlrs = results.get("minimum_track_record", {})
    print("\nMinimum Track Record:")
    print(f"  Required: {mlrs.get('mlrs_days', 'N/A')} days ({mlrs.get('mlrs_years', 'N/A')} years)")
    print(f"  Actual: {mlrs.get('actual_days', 'N/A')} days ({mlrs.get('actual_years', 'N/A')} years)")
    print(f"  Sufficient: {'YES' if mlrs.get('sufficient') else 'NO'}")

    print(f"\nOverall: {results['tests_passed']}/{results['total_tests']} tests passed")
    print(f"Edge Verified: {'YES' if results['edge_verified'] else 'NO'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ultimate Backtest Runner")
    parser.add_argument("--strategy", type=str, default="Swing_Trading", help="Strategy name")
    parser.add_argument("--months", type=int, default=120, help="Lookback months (default 120 = 10y)")
    parser.add_argument("--telegram", action="store_true", help="Send summary to Telegram")
    parser.add_argument(
        "--mc-iterations",
        type=int,
        default=8,
        help="Walk-forward MC iterations per window (default: 8). Set 0 to skip walk-forward.",
    )
    parser.add_argument(
        "--skip-wf",
        action="store_true",
        help="Skip walk-forward Monte Carlo validation (Phase 3). Saves ~10 minutes.",
    )
    parser.add_argument(
        "--skip-phases",
        type=str,
        default=None,
        help="Comma-separated phase numbers to skip (e.g. --skip-phases 2,3,4)",
    )
    args = parser.parse_args()

    skipped_phases = set()
    if args.skip_phases:
        for p in args.skip_phases.split(","):
            p = p.strip()
            if p.isdigit():
                skipped_phases.add(int(p))

    setup_logging()
    timer = Timer()
    end_date = datetime.now().strftime("%Y-%m-%d")
    run_wf = (not args.skip_wf) and (args.mc_iterations > 0) and (3 not in skipped_phases)

    print(f"\n{'='*70}")
    print(f"ULTIMATE BACKTEST — {args.strategy}")
    print(f"{'='*70}")
    print("Configuration:")
    print(f"  Lookback: {args.months} months ({args.months//12}y)")
    print(f"  End date: {end_date} (today)")
    print(f"  Walk-Forward: {'YES' if run_wf else 'SKIPPED'}")
    print(f"  Skipped Phases: {sorted(skipped_phases) if skipped_phases else 'None'}")
    print(f"  Send Telegram: {args.telegram}")
    print(f"{'='*70}")

    phases_done = 0
    total_phases = 6

    # ─── PHASE 1: Historical Backtest ───
    # Runs the strategy over real historical data with realistic execution costs
    timer.phase_start("Phase 1: Historical Backtest (Realistic Costs)")
    # Calculate fetch period from --months + 1 year buffer for warmup
    # Snap to valid yfinance periods: 1y, 2y, 5y, 10y, max
    needed_years = (args.months + 11) // 12 + 1
    for y in (1, 2, 5, 10, 15, 20):
        if y >= needed_years:
            period = f"{y}y"
            break
    else:
        period = "max"
    historical = run_historical_with_realistic_costs(
        args.strategy,
        end_date,
        args.months,
        period=period,
    )
    timer.phase_end()
    phases_done += 1
    print(f"   ⏳ {timer.estimate_remaining(phases_done, total_phases)}")

    # ─── PHASE 1b: Save Results to MongoDB ───
    # Persists trades, daily snapshots, and summary metrics for later review
    timer.phase_start("Phase 1b: Save Results to MongoDB")
    session_id = None
    try:
        persister = PersistenceHandler()
        print("  → Creating backtest session...")
        session_id = persister.create_backtest_session(
            strategy_name=args.strategy,
            strategy_config={"months": args.months},
            capital_config={"initial_capital": 100000},
            symbols=[],
        )
        print(f"  → Session ID: {session_id}")
        print(f"  → Saving {len(historical.get('trades', []))} trades...")
        persister.save_portfolio_backtest_trades(session_id, historical.get("trades", []))
        print(f"  → Saving {len(historical.get('daily_snapshots', []))} daily snapshots...")
        persister.save_portfolio_backtest_snapshots(session_id, historical.get("daily_snapshots", []))
        summary_metrics = {
            "initial_capital": historical["initial_capital"],
            "final_portfolio_value": historical["final_portfolio_value"],
            "total_return_pct": historical["total_return_pct"],
            "cagr": historical["cagr"],
            "max_drawdown_pct": historical["max_drawdown_pct"],
            "sharpe_ratio": historical["sharpe_ratio"],
            "total_trades": historical["total_trades"],
            "win_rate": historical["win_rate"],
            "profit_factor": historical["profit_factor"],
            "expectancy": historical["expectancy"],
        }
        print("  → Saving summary metrics...")
        persister.complete_backtest_session(
            session_id,
            summary_metrics,
            date_range=historical.get("date_range"),
        )
        print("  ✅ Results saved to MongoDB")
    except Exception as e:
        print(f"  ❌ DB save failed: {e}")
    timer.phase_end()
    phases_done += 1
    print(f"   ⏳ {timer.estimate_remaining(phases_done, total_phases)}")

    # ─── PHASE 2: Statistical Validation ───
    validation = {}
    if 2 not in skipped_phases:
        timer.phase_start("Phase 2: Statistical Validation (DSR + MC Permutation + MLRS)")
        validation = run_validation_phase(historical.get("daily_snapshots", []))
        timer.phase_end()

    # ─── PHASE 3: Walk-Forward Monte Carlo ───
    wf_results = None
    if run_wf and 3 not in skipped_phases:
        timer.phase_start(f"Phase 3: Walk-Forward Monte Carlo ({args.mc_iterations} iterations per window)")
        try:
            wf_results = run_walk_forward_backtest(
                strategy_name=args.strategy,
                period=period,
                symbols=historical.get("_scanned_symbols"),
                mc_iterations=args.mc_iterations,
                verbose=False,
                save_to_db=False,
                symbols_data=historical.get("_symbols_data"),
                indicators=historical.get("_indicators"),
                prefilter=historical.get("_prefilter"),
                precomputed_signals=historical.get("_precomputed_signals"),
            )
        except Exception as e:
            print(f"  ❌ Walk-forward failed: {e}")
            wf_results = {"status": "failed", "error": str(e)}
        timer.phase_end()

    # ─── PHASE 4: Stress Tests ───
    stress_results = None
    if 4 not in skipped_phases:
        timer.phase_start("Phase 4: Stress Tests (Regime + Param Sensitivity + Cost Sensitivity)")
        from scripts.stress_tests import run_all_stress_tests

        try:
            stress_results = run_all_stress_tests(
                strategy_name=args.strategy,
                symbols=historical.get("_scanned_symbols"),
                symbols_data=historical.get("_symbols_data"),
                indicators=historical.get("_indicators"),
                prefilter=historical.get("_prefilter"),
                precomputed_signals=historical.get("_precomputed_signals"),
            )
        except Exception as e:
            print(f"  ❌ Stress tests failed: {e}")
            stress_results = {"status": "failed", "error": str(e)}
        timer.phase_end()

    # ─── PHASE 5: Trade Diagnostics ───
    # Analyzes individual trade characteristics:
    # avg hold time, win/loss distribution, exit reasons, max concurrent positions
    timer.phase_start("Phase 5: Trade Diagnostics")
    from scripts.trade_diagnostics import run_all_diagnostics

    diagnostics = run_all_diagnostics(strategy_name=args.strategy)
    timer.phase_end()
    phases_done += 1
    print(f"   ⏳ {timer.estimate_remaining(phases_done, total_phases)}")

    # ─── PHASE 6: Composite Confidence Score ───
    # Combines all phase results into a 0-100 score
    # Weights: WF 15%, DSR 15%, MC Perm 15%, Stress 15%, Param 10%, Cost 10%, Data 5%
    timer.phase_start("Phase 6: Composite Confidence Score")
    from scripts.confidence_scorer import compute_confidence_score, print_confidence_report

    param_results = stress_results.get("param_sensitivity", []) if stress_results else None
    cost_results = stress_results.get("cost_sensitivity", []) if stress_results else None
    regime_results = stress_results.get("regime_tests", []) if stress_results else None

    confidence = compute_confidence_score(
        walk_forward_results=wf_results,
        validation_results=validation,
        stress_test_results={"regime_tests": regime_results} if regime_results else None,
        param_sensitivity_results=param_results,
        cost_sensitivity_results=cost_results,
        base_cagr=historical.get("cagr", 0),
    )
    print_confidence_report(confidence)
    timer.phase_end()
    phases_done += 1

    # Save all Phase 2-6 results to MongoDB
    if session_id:
        try:
            persister.save_ultimate_backtest_phases(
                session_id,
                {
                    "validation": validation,
                    "walk_forward": wf_results,
                    "stress_tests": stress_results,
                    "trade_diagnostics": diagnostics,
                    "confidence_score": confidence,
                },
            )
            print("  All phase results saved to MongoDB")
        except Exception as e:
            print(f"  Failed to save phase results: {e}")

    # ===== PRINT RESULTS IN TABLE FORMAT =====
    print(f"\n{'='*70}")
    print("ULTIMATE BACKTEST RESULTS")
    print(f"{'='*70}\n")

    # Table 1: Performance Summary
    perf_data = {
        "Metric": [
            "Date Range",
            "Initial Capital",
            "Final Value",
            "Total Return",
            "CAGR",
            "Max Drawdown",
            "Sharpe Ratio",
            "Profit Factor",
            "Win Rate",
            "Total Trades",
            "Expectancy",
            "Avg Positions Held",
        ],
        "Value": [
            f"{historical['date_range']['start_date']} → {historical['date_range']['end_date']}",
            f"₹{historical['initial_capital']:,.0f}",
            f"₹{historical['final_portfolio_value']:,.0f}",
            f"{historical['total_return_pct']:+.2f}%",
            f"{historical['cagr']:.2f}%",
            f"{historical['max_drawdown_pct']:.2f}%",
            f"{historical['sharpe_ratio']:.2f}",
            f"{historical['profit_factor']:.2f}",
            f"{historical['win_rate']:.1f}%",
            str(historical["total_trades"]),
            f"₹{historical['expectancy']:,.2f}",
            f"{historical['avg_positions_held']:.1f}",
        ],
    }
    perf_df = pd.DataFrame(perf_data)
    print(perf_df.to_string(index=False))

    # Table 2: Statistical Validation
    dsr = validation.get("dsr", {})
    mc = validation.get("monte_carlo_permutation", {})
    mlrs = validation.get("minimum_track_record", {})
    stat_data = {
        "Test": [
            "Deflated Sharpe Ratio",
            "DSR Confidence",
            "Monte Carlo p-value",
            "Min Track Record (years)",
            "Actual Track Record (years)",
            "Edge Verified",
        ],
        "Result": [
            f"{dsr.get('dsr', 'N/A')}",
            f"{dsr.get('confidence_pct', 'N/A')}%",
            f"{mc.get('p_value', 'N/A')}",
            f"{mlrs.get('mlrs_years', 'N/A')}",
            f"{mlrs.get('actual_years', 'N/A')}",
            "YES" if validation.get("edge_verified") else "NO",
        ],
        "Status": [
            "PASS" if dsr.get("significant") else "FAIL",
            "PASS" if dsr.get("dsr", 0) > 0.5 else "FAIL",
            "PASS" if mc.get("significant") else "FAIL",
            "PASS" if mlrs.get("sufficient") else "FAIL",
            "PASS" if mlrs.get("sufficient") else "FAIL",
            "PASS" if validation.get("edge_verified") else "FAIL",
        ],
    }
    stat_df = pd.DataFrame(stat_data)
    print(f"\n{stat_df.to_string(index=False)}")

    # Table 3: Confidence Score Components
    components = confidence.get("components", {})
    comp_data = {
        "Component": [
            "Walk-Forward Stability",
            "Deflated Sharpe Ratio",
            "Monte Carlo Permutation",
            "Stress Tests",
            "Parameter Stability",
            "Cost Resilience",
            "Data Sufficiency",
        ],
        "Score": [
            f"{components.get('walk_forward', {}).get('score', 0):.0f}/15",
            f"{components.get('dsr', {}).get('score', 0):.0f}/15",
            f"{components.get('mc_permutation', {}).get('score', 0):.0f}/15",
            f"{components.get('stress_tests', {}).get('score', 0):.0f}/15",
            f"{components.get('param_stability', {}).get('score', 0):.0f}/10",
            f"{components.get('cost_resilience', {}).get('score', 0):.0f}/10",
            f"{components.get('data_sufficiency', {}).get('score', 0):.0f}/5",
        ],
        "Weight": ["15%", "15%", "15%", "15%", "10%", "10%", "5%"],
    }
    comp_df = pd.DataFrame(comp_data)
    print(f"\n{comp_df.to_string(index=False)}")
    print(f"\n{'='*70}")
    print(f"TOTAL CONFIDENCE: {confidence['total_score']}/100  |  Level: {confidence['confidence_level']}")
    print(f"Realistic CAGR Projection: {confidence['realistic_cagr']:.1f}%")
    print(f"Edge Verified: {'YES' if confidence['edge_verified'] else 'NO'}")
    print(f"{'='*70}")

    # Telegram summary
    if args.telegram:
        msg = (
            f"🔬 <b>Ultimate Backtest Complete</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Strategy: {args.strategy}\n"
            f"Period: {historical['date_range']['start_date']} → {historical['date_range']['end_date']}\n\n"
            f"📊 <b>Realistic Results</b>:\n"
            f"• CAGR: {historical['cagr']:.1f}%\n"
            f"• Max DD: {historical['max_drawdown_pct']:.1f}%\n"
            f"• Sharpe: {historical['sharpe_ratio']:.2f}\n"
            f"• Win Rate: {historical['win_rate']:.1f}%\n\n"
            f"🎯 <b>Confidence</b>:\n"
            f"• Score: {confidence['total_score']}/100\n"
            f"• Level: {confidence['confidence_level']}\n"
            f"• Realistic CAGR: {confidence['realistic_cagr']:.1f}%\n"
            f"• Edge Verified: {'YES' if confidence['edge_verified'] else 'NO'}\n\n"
            f"⏱️ Duration: {timer.elapsed()}"
        )
        _send_telegram(msg)

    print(f"\n{'='*70}")
    print(f"TOTAL RUNTIME: {timer.elapsed()}")
    print(f"{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
