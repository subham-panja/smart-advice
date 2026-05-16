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
from utils.stock_scanner import StockScanner
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
    max_stocks: int = 50,
) -> dict:
    """Phase 1: Historical backtest with execution realism."""
    print(f"\n{'='*70}")
    print("PHASE 1: HISTORICAL BACKTEST (REALISTIC COSTS)")
    print(f"{'='*70}\n")

    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy '{strategy_name}' not found")

    strategy.setdefault("analysis_config", {})["market_regime_detection"] = True

    symbols = StockScanner().get_symbols(strategy_config=strategy)
    symbols = dict(list(symbols.items())[:max_stocks])
    print(f"Selected {len(symbols)} stocks")

    print("Fetching 10y historical data...")
    symbols_data = fetch_symbols_data(symbols, period="10y", verbose=False)

    index_data = _prepare_index_data(strategy, symbols_data, "10y")

    # Minimum data filter
    MIN_DAYS = 250
    excluded = 0
    for sym, df in list(symbols_data.items()):
        if len(df) < MIN_DAYS:
            excluded += 1
            del symbols_data[sym]
    if excluded:
        print(f"Excluded {excluded} stocks with < {MIN_DAYS} days")
    print(f"Remaining: {len(symbols_data)} stocks")

    sim_end = pd.Timestamp(end_date, tz="Asia/Kolkata")
    sim_start = sim_end - pd.DateOffset(months=months)

    # Regime warmup
    if index_data is not None:
        stock_only = {
            k: v for k, v in symbols_data.items() if k != strategy.get("market_regime_config", {}).get("index", "^NSEI")
        }
        all_sets = [set(df.index) for df in stock_only.values()]
        if all_sets:
            union_dates = sorted(set.union(*all_sets))
            for d in union_dates:
                if len(index_data.loc[:d]) >= 250:
                    if sim_start < d:
                        print(f"Advancing sim_start to {d.date()} for regime warmup")
                        sim_start = d
                    break

    print(f"Simulation: {sim_start.date()} -> {sim_end.date()} ({months} months)")

    # Pre-compute all indicators with vectorbt (one-time batch computation)
    print("Pre-computing indicators with vectorbt...")
    from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

    indicators = compute_all_indicators(symbols_data, strategy_config=strategy)
    store = IndicatorStore(indicators)
    print(f"Indicators pre-computed for {len(indicators.symbols)} stocks × {len(indicators.dates)} dates")

    engine = PortfolioBacktestSession(strategy_config=strategy)
    engine.set_indicator_store(store)
    if index_data is not None:
        engine._index_data_override = index_data

    print(f"Execution realism: gap_risk={'ON' if engine.use_realistic_costs else 'OFF'}")
    print("IndicatorStore: ENABLED (O(1) lookups)")
    results = engine.run(symbols_data, sim_start_date=sim_start, sim_end_date=sim_end)

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
    print(f"  Shuffled mean: {mc.get('shuffled_sharpe_mean', 'N/A')}")
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
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date")
    parser.add_argument("--months", type=int, default=120, help="Lookback months (default 120 = 10y)")
    parser.add_argument("--max-stocks", type=int, default=50, help="Max stocks")
    parser.add_argument("--telegram", action="store_true", help="Send summary to Telegram")
    parser.add_argument("--skip-wf", action="store_true", help="Skip walk-forward analysis")
    parser.add_argument("--skip-stress", action="store_true", help="Skip stress tests")
    args = parser.parse_args()

    setup_logging(verbose=False)
    timer = Timer()

    # Count total phases for ETA
    total_phases = 6
    if args.skip_wf:
        total_phases -= 1
    if args.skip_stress:
        total_phases -= 1
    phases_done = 0

    print(f"\n{'='*70}")
    print(f"ULTIMATE BACKTEST — {args.strategy}")
    print(
        f"Stocks: {args.max_stocks} | Months: {args.months} | Skip WF: {args.skip_wf} | Skip Stress: {args.skip_stress}"
    )
    print(f"{'='*70}")

    # Phase 1: Historical with realistic costs
    timer.phase_start("Phase 1: Historical Backtest (Realistic Costs)")
    historical = run_historical_with_realistic_costs(args.strategy, args.end_date, args.months, args.max_stocks)
    timer.phase_end()
    phases_done += 1
    print(f"   {timer.estimate_remaining(phases_done, total_phases)}")

    # Save to DB
    timer.phase_start("Phase 1b: Save Results to MongoDB")
    session_id = None
    try:
        persister = PersistenceHandler()
        session_id = persister.create_backtest_session(
            strategy_name=args.strategy,
            strategy_config={"months": args.months, "max_stocks": args.max_stocks},
            capital_config={"initial_capital": 100000},
            symbols=[],
        )
        print(f"DB Session ID: {session_id}")
        persister.save_portfolio_backtest_trades(session_id, historical.get("trades", []))
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
        persister.complete_backtest_session(
            session_id,
            summary_metrics,
            date_range=historical.get("date_range"),
        )
        print("Results saved to MongoDB")
    except Exception as e:
        print(f"DB save failed: {e}")
    timer.phase_end()
    phases_done += 1
    print(f"   {timer.estimate_remaining(phases_done, total_phases)}")

    # Phase 2: Statistical validation
    timer.phase_start("Phase 2: Statistical Validation")
    validation = run_validation_phase(historical.get("daily_snapshots", []))
    timer.phase_end()
    phases_done += 1
    print(f"   {timer.estimate_remaining(phases_done, total_phases)}")

    # Phase 3: Walk-forward (optional)
    wf_results = None
    if not args.skip_wf:
        timer.phase_start("Phase 3: Walk-Forward Monte Carlo")
        try:
            wf_results = run_walk_forward_backtest(
                strategy_name=args.strategy,
                period="10y",
                max_stocks=args.max_stocks,
                mc_iterations=8,
                verbose=False,
                save_to_db=False,
            )
        except Exception as e:
            print(f"Walk-forward failed: {e}")
            wf_results = {"status": "failed", "error": str(e)}
        timer.phase_end()
        phases_done += 1
        print(f"   {timer.estimate_remaining(phases_done, total_phases)}")

    # Phase 4: Stress tests (optional)
    stress_results = None
    if not args.skip_stress:
        timer.phase_start("Phase 4: Stress Tests (Regime + Param + Cost)")
        from scripts.stress_tests import run_all_stress_tests

        stress_results = run_all_stress_tests(args.strategy, args.max_stocks)
        timer.phase_end()
        phases_done += 1
        print(f"   {timer.estimate_remaining(phases_done, total_phases)}")

    # Phase 5: Trade diagnostics
    timer.phase_start("Phase 5: Trade Diagnostics")
    from scripts.trade_diagnostics import run_all_diagnostics

    _ = run_all_diagnostics(strategy_name=args.strategy)
    timer.phase_end()
    phases_done += 1
    print(f"   {timer.estimate_remaining(phases_done, total_phases)}")

    # Phase 6: Confidence score
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
