"""
Backtesting Validation Utilities
===============================

Implements walk-forward evaluation and related helpers for robust strategy validation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple, Optional


SignalFunc = Callable[[pd.DataFrame], pd.Series]
MetricFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class WalkForwardConfig:
    # Date strings in YYYY-MM-DD format or year-only strings like '2018'
    train_start: str = "2018-01-01"
    train_end: str = "2021-12-31"
    val_start: str = "2022-01-01"
    val_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2024-12-31"


def _slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if df.empty:
        return df
    idx = pd.to_datetime(df.index)
    mask = (idx >= pd.to_datetime(start)) & (idx <= pd.to_datetime(end))
    return df.loc[mask]


def walk_forward_evaluate(
    price_df: pd.DataFrame,
    signal_fn: SignalFunc,
    backtest_fn: Callable[[pd.DataFrame, pd.Series], Dict[str, Any]],
    config: WalkForwardConfig = WalkForwardConfig(),
) -> Dict[str, Any]:
    """
    Perform strict walk-forward evaluation using fixed windows:
    - Train: 2018–2021 (signals may be fit/tuned upstream if needed)
    - Validate: 2022
    - Test: 2023–2024

    This function assumes signal_fn is deterministic given price_df and any
    upstream configuration; it returns section-wise backtest results.
    """
    if price_df is None or price_df.empty:
        return {"error": "empty_price_data"}

    # Ensure index is datetime
    if not np.issubdtype(price_df.index.dtype, np.datetime64):
        try:
            price_df = price_df.copy()
            price_df.index = pd.to_datetime(price_df.index)
        except Exception:
            return {"error": "invalid_index_type"}

    sections = {
        "train": (config.train_start, config.train_end),
        "validate": (config.val_start, config.val_end),
        "test": (config.test_start, config.test_end),
    }

    results: Dict[str, Any] = {"sections": {}}

    for name, (start, end) in sections.items():
        data_slice = _slice_by_date(price_df, start, end)
        if data_slice.empty or len(data_slice) < 60:
            results["sections"][name] = {"error": "insufficient_data", "start": start, "end": end}
            continue

        try:
            signals = signal_fn(data_slice)
            bt_result = backtest_fn(data_slice, signals)
        except Exception as e:
            results["sections"][name] = {"error": f"exception: {e}", "start": start, "end": end}
            continue

        results["sections"][name] = {
            "start": start,
            "end": end,
            "backtest": bt_result,
        }

    # Aggregate top-level summary where possible
    def _safe_get(section: str, key_path: List[str], default: Any = 0.0):
        obj = results["sections"].get(section, {})
        cur: Any = obj
        for k in key_path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    # Example aggregated metrics: CAGR and win_rate across sections where available
    agg = {}
    for sec in ("train", "validate", "test"):
        agg[f"{sec}_cagr"] = _safe_get(sec, ["backtest", "cagr"], 0.0)
        agg[f"{sec}_win_rate"] = _safe_get(sec, ["backtest", "win_rate"], 0.0)
        agg[f"{sec}_total_trades"] = _safe_get(sec, ["backtest", "total_trades"], 0)

    results["summary"] = agg
    return results


def _time_series_kfold_slices(index: pd.DatetimeIndex, k: int = 5, purge_ratio: float = 0.05) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Create contiguous time-based k-fold slices with a purge gap between folds to avoid leakage.
    Returns list of (start, end) for each test fold.
    """
    if k <= 1 or len(index) < k:
        return [(index[0], index[-1])]
    n = len(index)
    fold_size = n // k
    slices: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size - 1 if i < k - 1 else n - 1
        # Apply purge at both ends within bounds
        purge = max(1, int(fold_size * purge_ratio))
        s = max(start_idx, start_idx + purge)
        e = max(s, end_idx - purge)
        slices.append((index[s], index[e]))
    return slices


def purged_kfold_evaluate(
    price_df: pd.DataFrame,
    signal_fn: SignalFunc,
    backtest_fn: Callable[[pd.DataFrame, pd.Series], Dict[str, Any]],
    k: int = 5,
    purge_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform purged K-fold cross-validation on time series data to reduce leakage from overlapping bars.
    """
    if price_df is None or price_df.empty:
        return {"error": "empty_price_data"}

    df = price_df.copy()
    if not np.issubdtype(df.index.dtype, np.datetime64):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return {"error": "invalid_index_type"}

    folds = _time_series_kfold_slices(df.index, k=k, purge_ratio=purge_ratio)
    results: Dict[str, Any] = {"folds": []}

    for i, (start, end) in enumerate(folds, start=1):
        data_slice = df.loc[(df.index >= start) & (df.index <= end)]
        if data_slice.empty or len(data_slice) < 60:
            results["folds"].append({"fold": i, "start": str(start), "end": str(end), "error": "insufficient_data"})
            continue
        try:
            signals = signal_fn(data_slice)
            bt_result = backtest_fn(data_slice, signals)
        except Exception as e:
            results["folds"].append({"fold": i, "start": str(start), "end": str(end), "error": f"exception: {e}"})
            continue
        results["folds"].append({"fold": i, "start": str(start), "end": str(end), "backtest": bt_result})

    # Aggregate summary
    def _get_fold_metric(fidx: int, key: str, default: float = 0.0):
        fold_obj = results["folds"][fidx]
        return fold_obj.get("backtest", {}).get(key, default)

    if results["folds"]:
        avg_cagr = np.mean([_get_fold_metric(i, "cagr", 0.0) for i in range(len(results["folds"]))])
        avg_win = np.mean([_get_fold_metric(i, "win_rate", 0.0) for i in range(len(results["folds"]))])
        results["summary"] = {"avg_cagr": float(avg_cagr), "avg_win_rate": float(avg_win)}
    else:
        results["summary"] = {"avg_cagr": 0.0, "avg_win_rate": 0.0}

    return results


def compute_buy_precision_at_horizons(signals: pd.Series, close: pd.Series, horizons: List[int] = [10, 20]) -> Dict[str, Any]:
    """
    Compute PPV (precision) of BUY signals at specified day horizons.
    A BUY is counted as a "hit" if future return over the horizon is > 0%.
    """
    if signals is None or close is None or signals.empty or close.empty:
        return {"error": "empty_input"}

    # Align indices
    s = signals.dropna()
    c = close.reindex(s.index).dropna()
    s = s.reindex(c.index)

    buy_idx = s[s == 'BUY'].index
    precision: Dict[str, float] = {}
    counts: Dict[str, int] = {}

    for h in horizons:
        hits = 0
        total = 0
        for dt in buy_idx:
            if dt not in c.index:
                continue
            # Future index h bars ahead
            future_idx = c.index.get_indexer([dt])[0] + h
            if future_idx >= len(c):
                continue
            base = float(c.loc[dt])
            fut = float(c.iloc[future_idx])
            if base > 0:
                ret = (fut - base) / base
                if ret > 0:
                    hits += 1
                total += 1
        key = f"ppv_{h}d"
        precision[key] = (hits / total) if total > 0 else 0.0
        counts[f"n_{h}d"] = total

    return {"precision": precision, "counts": counts}


def monte_carlo_resample_drawdown(trade_returns: List[float], n_trials: int = 1000, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Monte Carlo resampling of trade sequence to estimate downside risk via drawdown distribution.
    trade_returns should be per-trade return percentages (e.g., 0.05 for +5%).
    """
    if seed is not None:
        np.random.seed(seed)
    if not trade_returns:
        return {"error": "empty_trades"}

    def max_drawdown_from_path(path: List[float]) -> float:
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in path:
            equity *= (1.0 + r)
            peak = max(peak, equity)
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    dds = []
    for _ in range(n_trials):
        path = np.random.permutation(trade_returns)
        dds.append(max_drawdown_from_path(path))

    return {
        "dd_mean": float(np.mean(dds)),
        "dd_p95": float(np.percentile(dds, 95)),
        "dd_p99": float(np.percentile(dds, 99)),
        "trials": n_trials,
    }


def ablation_tests(
    price_df: pd.DataFrame,
    signal_fn_variants: Dict[str, SignalFunc],
    backtest_fn: Callable[[pd.DataFrame, pd.Series], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run ablation tests by evaluating multiple signal function variants (e.g., disabling gates).
    Returns a dict mapping variant name to backtest summary.
    """
    results: Dict[str, Any] = {}
    for name, fn in signal_fn_variants.items():
        try:
            signals = fn(price_df)
            bt = backtest_fn(price_df, signals)
            results[name] = bt
        except Exception as e:
            results[name] = {"error": str(e)}
    return {"variants": results}

