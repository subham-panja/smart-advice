"""
Stress Tests for Swing Trading Strategy
=========================================

Runs the strategy across different market conditions, parameter variations,
stock universes, and cost levels to verify robustness.

Tests:
1. Regime-specific performance (bull, bear, crash, sideways)
2. Parameter sensitivity (+/- 20%)
3. Universe sensitivity (different stock pools)
4. Transaction cost sensitivity
"""

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from scripts.run_portfolio_backtest import _prepare_index_data, fetch_symbols_data
from utils.stock_scanner import StockScanner
from utils.strategy_loader import StrategyLoader

# ---------------------------------------------------------------------------
# 1. Regime-Specific Performance Tests
# ---------------------------------------------------------------------------

REGIME_TEST_PERIODS = [
    {
        "name": "Bull Market (2017)",
        "end_date": "2018-01-01",
        "months": 12,
        "min_cagr": 15.0,
        "max_dd_tolerance": -10.0,
    },
    {
        "name": "Bear Market (2018 Crash)",
        "end_date": "2019-03-01",
        "months": 15,
        "min_cagr": -5.0,
        "max_dd_tolerance": -15.0,
    },
    {
        "name": "COVID Crash (2020)",
        "end_date": "2021-03-01",
        "months": 13,
        "min_cagr": -10.0,
        "max_dd_tolerance": -20.0,
    },
    {
        "name": "Strong Recovery (2020-2021)",
        "end_date": "2022-01-01",
        "months": 21,
        "min_cagr": 20.0,
        "max_dd_tolerance": -15.0,
    },
    {
        "name": "Sideways Market (2022-2023)",
        "end_date": "2024-01-01",
        "months": 24,
        "min_cagr": 10.0,
        "max_dd_tolerance": -12.0,
    },
]


def run_regime_tests(
    strategy_name: str = "Swing_Trading",
    max_stocks: int = 50,
    verbose: bool = False,
    symbols: dict = None,
    symbols_data: dict = None,
    index_data=None,
    indicators=None,
    prefilter=None,
    precomputed_signals=None,
) -> List[Dict[str, Any]]:
    """Run historical backtests on specific market regimes.

    Returns:
        List of results with pass/fail assessment per regime
    """
    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy '{strategy_name}' not found")

    strategy.setdefault("analysis_config", {})["market_regime_detection"] = True

    print(f"\n{'='*70}")
    print("REGIME-SPECIFIC STRESS TESTS")
    print(f"{'='*70}\n")

    # Use pre-fetched data if provided, otherwise fetch
    if symbols is None:
        symbols = StockScanner().get_symbols(strategy_config=strategy)
        symbols = dict(list(symbols.items())[:max_stocks])
    if symbols_data is None:
        symbols_data = fetch_symbols_data(symbols, period="10y", verbose=False)
        index_data = _prepare_index_data(strategy, symbols_data, "10y")

    # Use pre-computed indicators if provided, otherwise compute locally
    from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

    if indicators is None:
        indicators = compute_all_indicators(symbols_data, strategy_config=strategy)
    indicator_store = IndicatorStore(indicators)

    if prefilter is None:
        from scripts.vectorbt_signal_generator import compute_stock_prefilter

        prefilter = compute_stock_prefilter(indicators, strategy)

    # Minimum data filter
    MIN_DAYS = 250
    for sym, df in list(symbols_data.items()):
        if len(df) < MIN_DAYS:
            del symbols_data[sym]

    results = []
    for test in REGIME_TEST_PERIODS:
        print(f"\n--- {test['name']} ---")

        sim_end = pd.Timestamp(test["end_date"], tz="Asia/Kolkata")
        sim_start = sim_end - pd.DateOffset(months=test["months"])

        # Regime warmup
        if index_data is not None:
            stock_only = {
                k: v
                for k, v in symbols_data.items()
                if k != strategy.get("market_regime_config", {}).get("index", "^NSEI")
            }
            all_sets = [set(df.index) for df in stock_only.values()]
            if all_sets:
                union_dates = sorted(set.union(*all_sets))
                for d in union_dates:
                    if len(index_data.loc[:d]) >= 250:
                        if sim_start < d:
                            sim_start = d
                        break

        engine = PortfolioBacktestSession(strategy_config=strategy)
        engine.set_indicator_store(indicator_store)
        engine._stock_prefilter = prefilter
        if index_data is not None:
            engine._index_data_override = index_data

        if precomputed_signals:
            result = engine.run_with_signals(
                symbols_data, precomputed_signals, sim_start_date=sim_start, sim_end_date=sim_end, verbose=False
            )
        else:
            result = engine.run(symbols_data, sim_start_date=sim_start, sim_end_date=sim_end)

        cagr = result["cagr"]
        max_dd = result["max_drawdown_pct"]
        passed = cagr >= test["min_cagr"] and max_dd >= test["max_dd_tolerance"]

        entry = {
            "regime": test["name"],
            "date_range": result["date_range"],
            "cagr": round(cagr, 2),
            "max_dd": round(max_dd, 2),
            "sharpe": round(result["sharpe_ratio"], 2),
            "win_rate": round(result["win_rate"], 1),
            "total_trades": result["total_trades"],
            "min_cagr_required": test["min_cagr"],
            "max_dd_tolerance": test["max_dd_tolerance"],
            "passed": passed,
        }
        results.append(entry)

        status = "PASS" if passed else "FAIL"
        print(
            f"  CAGR: {cagr:.1f}% (min {test['min_cagr']}%) | "
            f"Max DD: {max_dd:.1f}% (tol {test['max_dd_tolerance']}%) | "
            f"[{status}]"
        )

    return results


# ---------------------------------------------------------------------------
# 2. Parameter Sensitivity Analysis
# ---------------------------------------------------------------------------

# Fine-grained ATR stop sweep (replaces coarse low/high test)
ATR_STOP_SWEEP = [2.0, 2.5, 2.8, 3.0, 3.05, 3.1, 3.15, 3.2, 3.3, 3.5, 4.0]

PARAM_VARIATIONS = {
    "time_stop_bars": {"base": 12, "low": 10, "high": 14},
    "min_rsi": {"base": 50, "low": 45, "high": 55},
    "min_volume_ratio": {"base": 0.6, "low": 0.5, "high": 0.7},
    "max_positions": {"base": 8, "low": 6, "high": 10},
    "risk_per_trade_pct": {"base": 2.0, "low": 1.6, "high": 2.4},
}


def run_param_sensitivity(
    strategy_name: str = "Swing_Trading",
    max_stocks: int = 50,
    test_months: int = 60,
    end_date: str = "2026-05-15",
    symbols: dict = None,
    symbols_data: dict = None,
    index_data=None,
    indicators=None,
    prefilter=None,
    precomputed_signals=None,
) -> List[Dict[str, Any]]:
    """Test strategy performance with +/- 20% parameter variations.

    Returns:
        List of results per parameter variation
    """
    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy '{strategy_name}' not found")

    strategy.setdefault("analysis_config", {})["market_regime_detection"] = True

    if symbols_data is None:
        symbols = StockScanner().get_symbols(strategy_config=strategy)
        symbols = dict(list(symbols.items())[:max_stocks])
        symbols_data = fetch_symbols_data(symbols, period="10y", verbose=False)
        index_data = _prepare_index_data(strategy, symbols_data, "10y")

    MIN_DAYS = 250
    for sym, df in list(symbols_data.items()):
        if len(df) < MIN_DAYS:
            del symbols_data[sym]

    sim_end = pd.Timestamp(end_date, tz="Asia/Kolkata")
    sim_start = sim_end - pd.DateOffset(months=test_months)

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
                        sim_start = d
                    break

    # Use pre-computed indicators if provided, otherwise compute locally
    from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

    if indicators is None:
        indicators = compute_all_indicators(symbols_data, strategy_config=strategy)
    indicator_store = IndicatorStore(indicators)

    if prefilter is None:
        from scripts.vectorbt_signal_generator import compute_stock_prefilter

        prefilter = compute_stock_prefilter(indicators, strategy)

    # Run base case
    print(f"\n{'='*70}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*70}\n")

    base_engine = PortfolioBacktestSession(strategy_config=strategy)
    base_engine.set_indicator_store(indicator_store)
    base_engine._stock_prefilter = prefilter
    if index_data is not None:
        base_engine._index_data_override = index_data
    if precomputed_signals:
        base_result = base_engine.run_with_signals(
            symbols_data, precomputed_signals, sim_start_date=sim_start, sim_end_date=sim_end
        )
    else:
        base_result = base_engine.run(symbols_data, sim_start_date=sim_start, sim_end_date=sim_end)
    base_cagr = base_result["cagr"]
    print(f"Base case CAGR: {base_cagr:.1f}%\n")

    results = [{"param": "BASE", "value": "base", "cagr": round(base_cagr, 2), "passed": True}]

    # --- ATR Stop Multiplier Fine-Grained Sweep ---
    print("--- ATR Stop Multiplier Sweep ---")
    atr_cagrs = []
    for atr_val in ATR_STOP_SWEEP:
        test_strat = _deep_copy_strategy(strategy)
        _apply_param_change(test_strat, "atr_stop_multiplier", atr_val)
        engine = PortfolioBacktestSession(strategy_config=test_strat)
        engine.set_indicator_store(indicator_store)
        engine._stock_prefilter = prefilter
        if index_data is not None:
            engine._index_data_override = index_data
        if precomputed_signals:
            result = engine.run_with_signals(
                symbols_data, precomputed_signals, sim_start_date=sim_start, sim_end_date=sim_end, verbose=False
            )
        else:
            result = engine.run(symbols_data, sim_start_date=sim_start, sim_end_date=sim_end)
        test_cagr = result["cagr"]
        atr_cagrs.append((atr_val, test_cagr))
        results.append({"param": "atr_stop_multiplier", "value": atr_val, "cagr": round(test_cagr, 2), "passed": True})

    # Print ATR sweep table with cliff detection
    print(f"  {'ATR':>5}  {'CAGR':>8}  {'Delta%':>8}  {'Cliff?':>7}")
    cliff_detected = False
    prev_cagr = None
    for atr_val, test_cagr in atr_cagrs:
        if prev_cagr is not None and base_cagr != 0:
            delta = ((test_cagr - prev_cagr) / abs(base_cagr)) * 100
            is_cliff = abs(delta) > 50
            if is_cliff:
                cliff_detected = True
            cliff_flag = "CLIFF!" if is_cliff else ""
        else:
            delta = 0
            cliff_flag = ""
        if prev_cagr is None:
            prev_cagr = test_cagr
        else:
            prev_cagr = test_cagr
        print(f"  {atr_val:>5.1f}  {test_cagr:>7.1f}%  {delta:>+7.1f}%  {cliff_flag:>7}")
    if cliff_detected:
        print("  *** CLIFF DETECTED: ATR stop multiplier is a fragile parameter ***")
    print()

    tolerance = 0.30  # CAGR must stay within +/- 30% of base

    for param_name, values in PARAM_VARIATIONS.items():
        for variation in ["low", "high"]:
            test_strat = _deep_copy_strategy(strategy)
            new_value = values[variation]

            # Apply parameter change
            _apply_param_change(test_strat, param_name, new_value)

            engine = PortfolioBacktestSession(strategy_config=test_strat)
            engine.set_indicator_store(indicator_store)
            engine._stock_prefilter = prefilter
            if index_data is not None:
                engine._index_data_override = index_data

            if precomputed_signals:
                result = engine.run_with_signals(
                    symbols_data, precomputed_signals, sim_start_date=sim_start, sim_end_date=sim_end
                )
            else:
                result = engine.run(symbols_data, sim_start_date=sim_start, sim_end_date=sim_end)
            test_cagr = result["cagr"]

            # Check if within tolerance
            deviation = 0.0
            if base_cagr != 0:
                deviation = abs(test_cagr - base_cagr) / abs(base_cagr)
                passed = deviation <= tolerance
            else:
                passed = test_cagr >= 0

            results.append(
                {
                    "param": param_name,
                    "value": new_value,
                    "variation": variation,
                    "cagr": round(test_cagr, 2),
                    "deviation_pct": round(deviation * 100, 1) if base_cagr != 0 else 0,
                    "passed": passed,
                }
            )

            status = "PASS" if passed else "FAIL"
            print(
                f"  {param_name} = {new_value} ({variation}): CAGR {test_cagr:.1f}% "
                f"(dev {deviation*100:.0f}%) [{status}]"
            )

    return results


def _deep_copy_strategy(strategy: dict) -> dict:
    """Deep copy strategy config dict."""
    import copy

    return copy.deepcopy(strategy)


def _apply_param_change(strategy: dict, param_name: str, value: Any):
    """Apply a parameter change to the strategy config."""
    param_map = {
        "atr_stop_multiplier": lambda s, v: _set_nested(s, ["exit_rules", "atr_stop_multiplier"], v),
        "time_stop_bars": lambda s, v: _set_nested(s, ["exit_rules", "time_stop_bars"], v),
        "min_rsi": lambda s, v: _set_nested(s, ["rsi_momentum_filter", "min_rsi"], v),
        "min_volume_ratio": lambda s, v: _set_nested(
            s, ["swing_trading_gates", "VOLUME_GATE", "params", "min_volume_ratio"], v
        ),
        "max_positions": lambda s, v: _set_nested(s, ["risk_management", "max_positions"], v),
        "risk_per_trade_pct": lambda s, v: _set_nested(s, ["risk_management", "risk_per_trade_pct"], v),
    }
    if param_name in param_map:
        param_map[param_name](strategy, value)


def _set_nested(d: dict, keys: list, value: Any):
    """Set a nested dict value."""
    for key in keys[:-1]:
        if key not in d:
            return
        d = d[key]
    d[keys[-1]] = value


# ---------------------------------------------------------------------------
# 3. Transaction Cost Sensitivity
# ---------------------------------------------------------------------------

COST_SCENARIOS = [
    {"name": "Ultra-low", "brokerage": 0.0001, "slippage": 0.0005},
    {"name": "Realistic", "brokerage": 0.0003, "slippage": 0.0015},
    {"name": "High", "brokerage": 0.0005, "slippage": 0.003},
    {"name": "Extreme", "brokerage": 0.001, "slippage": 0.005},
]


def run_cost_sensitivity(
    strategy_name: str = "Swing_Trading",
    max_stocks: int = 50,
    test_months: int = 60,
    end_date: str = "2026-05-15",
    symbols: dict = None,
    symbols_data: dict = None,
    index_data=None,
    indicators=None,
    prefilter=None,
    precomputed_signals=None,
) -> List[Dict[str, Any]]:
    """Test performance at different cost levels.

    Returns:
        List of results per cost scenario
    """
    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        raise RuntimeError(f"Strategy '{strategy_name}' not found")

    strategy.setdefault("analysis_config", {})["market_regime_detection"] = True

    if symbols_data is None:
        symbols = StockScanner().get_symbols(strategy_config=strategy)
        symbols = dict(list(symbols.items())[:max_stocks])
        symbols_data = fetch_symbols_data(symbols, period="10y", verbose=False)
        index_data = _prepare_index_data(strategy, symbols_data, "10y")

    MIN_DAYS = 250
    for sym, df in list(symbols_data.items()):
        if len(df) < MIN_DAYS:
            del symbols_data[sym]

    sim_end = pd.Timestamp(end_date, tz="Asia/Kolkata")
    sim_start = sim_end - pd.DateOffset(months=test_months)

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
                        sim_start = d
                    break

    # Use pre-computed indicators if provided, otherwise compute locally
    from scripts.vectorbt_indicator_batch import IndicatorStore, compute_all_indicators

    if indicators is None:
        indicators = compute_all_indicators(symbols_data, strategy_config=strategy)
    indicator_store = IndicatorStore(indicators)

    if prefilter is None:
        from scripts.vectorbt_signal_generator import compute_stock_prefilter

        prefilter = compute_stock_prefilter(indicators, strategy)

    print(f"\n{'='*70}")
    print("TRANSACTION COST SENSITIVITY")
    print(f"{'='*70}\n")

    results = []
    base_cagr = None

    for scenario in COST_SCENARIOS:
        engine = PortfolioBacktestSession(strategy_config=strategy)
        engine.set_indicator_store(indicator_store)
        engine._stock_prefilter = prefilter
        if index_data is not None:
            engine._index_data_override = index_data
        engine.brokerage = scenario["brokerage"]
        engine.slippage = scenario["slippage"]

        if precomputed_signals:
            result = engine.run_with_signals(
                symbols_data, precomputed_signals, sim_start_date=sim_start, sim_end_date=sim_end, verbose=False
            )
        else:
            result = engine.run(symbols_data, sim_start_date=sim_start, sim_end_date=sim_end)

        cagr = result["cagr"]
        if base_cagr is None:
            base_cagr = cagr

        entry = {
            "scenario": scenario["name"],
            "brokerage": scenario["brokerage"],
            "slippage": scenario["slippage"],
            "cagr": round(cagr, 2),
            "impact_vs_base": round(cagr - base_cagr, 2),
            "total_trades": result["total_trades"],
        }
        results.append(entry)
        print(
            f"  {scenario['name']:12s} (brk={scenario['brokerage']*100:.2f}%, slp={scenario['slippage']*100:.2f}%): "
            f"CAGR {cagr:.1f}% (impact {cagr - base_cagr:+.1f}pp)"
        )

    return results


# ---------------------------------------------------------------------------
# Combined Stress Test Runner
# ---------------------------------------------------------------------------


def run_all_stress_tests(
    strategy_name: str = "Swing_Trading",
    max_stocks: int = 50,
    symbols: dict = None,
    symbols_data: dict = None,
    index_data=None,
    indicators=None,
    prefilter=None,
    precomputed_signals=None,
) -> Dict[str, Any]:
    """Run all stress tests and return combined results.

    Computes indicators ONCE globally and runs 3 sub-tests in parallel
    for ~3x speedup. Falls back to sequential if parallel fails.
    Reuses pre-computed indicators/signals from Phase 1 when passed in.
    """
    # Pre-compute index_data if not provided
    if index_data is None and symbols_data is not None:
        strategy = StrategyLoader.get_strategy_by_name(strategy_name)
        if not strategy:
            raise RuntimeError(f"Strategy '{strategy_name}' not found")
        strategy.setdefault("analysis_config", {})["market_regime_detection"] = True
        index_data = _prepare_index_data(strategy, symbols_data, "10y")

    # Reuse pre-computed data from Phase 1 if available, otherwise compute fresh
    if indicators is not None and precomputed_signals is not None:
        print(f"  Reusing pre-computed indicators and signals from Phase 1 ({len(precomputed_signals)} symbols)")
    elif symbols_data is not None:
        try:
            strategy = StrategyLoader.get_strategy_by_name(strategy_name)
            strategy.setdefault("analysis_config", {})["market_regime_detection"] = True

            from scripts.vectorbt_indicator_batch import compute_all_indicators

            print("  Pre-computing indicators for stress tests (shared across all sub-tests)...")
            indicators = compute_all_indicators(symbols_data, strategy_config=strategy)

            from scripts.vectorbt_signal_generator import compute_stock_prefilter

            prefilter = compute_stock_prefilter(indicators, strategy)
            print("  Indicators + prefilter computed once for all stress tests")

            # Pre-compute signals ONCE for all stress test sub-tests
            from scripts.run_portfolio_backtest import precompute_full_signals

            print("  Pre-computing signals for stress tests...")
            precomputed_signals = precompute_full_signals(symbols_data, strategy, indicators, prefilter, num_workers=4)
            print(f"  Signals pre-computed: {len(precomputed_signals)} symbols with buy signals")
        except Exception as e:
            print(f"  Indicator pre-computation failed: {e}, sub-tests will compute individually")

    # Run sub-tests in parallel using fork (copy-on-write shares indicators/symbols_data)
    try:
        fork_ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=3, mp_context=fork_ctx) as executor:
            regime_f = executor.submit(
                run_regime_tests,
                strategy_name,
                max_stocks,
                symbols=symbols,
                symbols_data=symbols_data,
                index_data=index_data,
                indicators=indicators,
                prefilter=prefilter,
                precomputed_signals=precomputed_signals,
            )
            param_f = executor.submit(
                run_param_sensitivity,
                strategy_name,
                max_stocks,
                symbols=symbols,
                symbols_data=symbols_data,
                index_data=index_data,
                indicators=indicators,
                prefilter=prefilter,
                precomputed_signals=precomputed_signals,
            )
            cost_f = executor.submit(
                run_cost_sensitivity,
                strategy_name,
                max_stocks,
                symbols=symbols,
                symbols_data=symbols_data,
                index_data=index_data,
                indicators=indicators,
                prefilter=prefilter,
                precomputed_signals=precomputed_signals,
            )

            regime_results = regime_f.result()
            param_results = param_f.result()
            cost_results = cost_f.result()
    except Exception as e:
        print(f"  Parallel stress tests failed ({e}), falling back to sequential...")
        regime_results = run_regime_tests(
            strategy_name,
            max_stocks,
            symbols=symbols,
            symbols_data=symbols_data,
            index_data=index_data,
            indicators=indicators,
            prefilter=prefilter,
            precomputed_signals=precomputed_signals,
        )
        param_results = run_param_sensitivity(
            strategy_name,
            max_stocks,
            symbols=symbols,
            symbols_data=symbols_data,
            index_data=index_data,
            indicators=indicators,
            prefilter=prefilter,
            precomputed_signals=precomputed_signals,
        )
        cost_results = run_cost_sensitivity(
            strategy_name,
            max_stocks,
            symbols=symbols,
            symbols_data=symbols_data,
            index_data=index_data,
            indicators=indicators,
            prefilter=prefilter,
            precomputed_signals=precomputed_signals,
        )

    # Summary
    regime_pass = sum(1 for r in regime_results if r["passed"])
    regime_total = len(regime_results)
    param_pass = sum(1 for r in param_results if r["passed"])
    param_total = len(param_results)
    cost_pass = sum(1 for r in cost_results if r["cagr"] > 12.0)  # CAGR > 12% even at high costs
    cost_total = len(cost_results)

    return {
        "regime_tests": regime_results,
        "param_sensitivity": param_results,
        "cost_sensitivity": cost_results,
        "summary": {
            "regime_pass_rate": f"{regime_pass}/{regime_total}",
            "param_pass_rate": f"{param_pass}/{param_total}",
            "cost_pass_rate": f"{cost_pass}/{cost_total}",
            "overall_pass_rate": f"{regime_pass + param_pass + cost_pass}/{regime_total + param_total + cost_total}",
        },
    }
