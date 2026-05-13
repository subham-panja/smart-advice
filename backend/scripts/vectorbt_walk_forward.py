"""
Vectorbt-based walk-forward window generation and MC symbol subsampling.

Replaces the manual window-building loop and random sampling in
run_walk_forward_backtest with deterministic, reproducible functions.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_walk_forward_windows(
    all_dates: pd.DatetimeIndex,
    window_days: int = 180,
    step_days: int = 90,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Generate walk-forward (start, end) date tuples.

    Args:
        all_dates: Sorted DatetimeIndex of all available trading dates
        window_days: Test window size in calendar days
        step_days: Step/roll forward in calendar days

    Returns:
        List of (window_start, window_end) tuples
    """
    start_date = all_dates[0]
    end_date = all_dates[-1]

    windows = []
    current_start = start_date
    while current_start + pd.Timedelta(days=window_days) <= end_date:
        window_end = current_start + pd.Timedelta(days=window_days)
        windows.append((current_start, window_end))
        current_start += pd.Timedelta(days=step_days)

    return windows


def generate_mc_symbol_masks(
    n_symbols: int,
    n_iterations: int,
    sample_pct: float = 0.7,
    min_symbols: int = 20,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate boolean mask array for MC symbol subsampling.

    Args:
        n_symbols: Total number of symbols available
        n_iterations: Number of MC iterations
        sample_pct: Fraction of symbols to sample per iteration
        min_symbols: Minimum number of symbols to include
        seed: Random seed for reproducibility

    Returns:
        2D bool array (n_iterations x n_symbols) where masks[i, j] is True
        if symbol j is included in iteration i.
    """
    sample_size = max(int(n_symbols * sample_pct), min_symbols)

    rng = np.random.RandomState(seed)
    masks = np.zeros((n_iterations, n_symbols), dtype=bool)

    for i in range(n_iterations):
        indices = rng.choice(n_symbols, size=sample_size, replace=False)
        masks[i, indices] = True

    return masks


def slice_window_data(
    symbols_data: dict,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    warmup_days: int = 200,
    min_rows: int = 200,
) -> dict:
    """Slice symbols_data to a walk-forward window with warmup.

    Args:
        symbols_data: {symbol: DataFrame} full historical data
        window_start: Window start date
        window_end: Window end date
        warmup_days: Days of lookback before window_start
        min_rows: Minimum rows required for a symbol to be included

    Returns:
        {symbol: DataFrame} sliced to window + warmup
    """
    warmup_start = window_start - pd.Timedelta(days=warmup_days)
    window_data = {}

    for sym, df in symbols_data.items():
        sliced = df[(df.index >= warmup_start) & (df.index <= window_end)]
        if len(sliced) >= min_rows:
            window_data[sym] = sliced

    return window_data


def aggregate_walk_forward_results(
    all_results: List[dict],
) -> dict:
    """Aggregate walk-forward MC results into summary statistics.

    Args:
        all_results: List of result dicts from MC workers, each with keys:
            window, mc_iteration, status, cagr, total_return, max_drawdown,
            sharpe, total_trades, win_rate, profit_factor

    Returns:
        Summary dict with aggregated metrics
    """
    if not all_results:
        return {"status": "failed", "reason": "No successful runs"}

    successful = [r for r in all_results if r.get("status") == "success"]
    if not successful:
        return {"status": "failed", "reason": "No successful runs", "errors": all_results}

    cagrs = [r["cagr"] for r in successful]
    win_rates = [r["win_rate"] for r in successful]
    sharpe_ratios = [r["sharpe"] for r in successful]
    max_drawdowns = [r["max_drawdown"] for r in successful]
    profit_factors = [r["profit_factor"] for r in successful]

    def mean(lst):
        return sum(lst) / len(lst)

    def std(lst):
        m = mean(lst)
        return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5

    def median(lst):
        s = sorted(lst)
        n = len(s)
        return s[n // 2]

    # Consistency score: % of runs with positive CAGR
    positive_cagr_pct = sum(1 for c in cagrs if c > 0) / len(cagrs) * 100

    # Robustness score: 100 - (coefficient_of_variation * 100)
    mean_cagr = mean(cagrs)
    std_cagr = std(cagrs)
    cv = abs(std_cagr / mean_cagr) if mean_cagr != 0 else 999
    robustness_score = max(0, 100 - cv * 100)

    return {
        "status": "success",
        "total_runs": len(successful),
        "cagr": {
            "mean": mean(cagrs),
            "std": std(cagrs),
            "min": min(cagrs),
            "max": max(cagrs),
            "median": median(cagrs),
        },
        "win_rate": {
            "mean": mean(win_rates),
            "min": min(win_rates),
            "max": max(win_rates),
        },
        "sharpe": {
            "mean": mean(sharpe_ratios),
            "min": min(sharpe_ratios),
            "max": max(sharpe_ratios),
        },
        "max_drawdown": {
            "mean": mean(max_drawdowns),
            "worst": min(max_drawdowns),
        },
        "profit_factor": {
            "mean": mean(profit_factors),
        },
        "positive_cagr_pct": positive_cagr_pct,
        "robustness_score": robustness_score,
        "is_robust": robustness_score > 60 and positive_cagr_pct > 70,
    }
