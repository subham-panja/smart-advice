"""
Trade Diagnostics
=================

Analyzes trade-level data from MongoDB to understand how the strategy
actually performs at the individual trade level.

Diagnostics:
1. Trade distribution (per year, holding period, consecutive losses)
2. Equity curve analysis (underwater chart, recovery time, monthly returns)
3. Regime performance breakdown
4. Signal quality analysis
"""

import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import get_mongodb

# ---------------------------------------------------------------------------
# 1. Trade Distribution Analysis
# ---------------------------------------------------------------------------


def analyze_trade_distribution(
    session_id: Optional[str] = None,
    strategy_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze trade-level statistics from MongoDB.

    Args:
        session_id: Specific backtest session ID (portfolio backtest)
        strategy_name: Filter by strategy name

    Returns:
        Dict with trade distribution metrics
    """
    db = get_mongodb()
    collection = db.portfolio_backtest_trades if session_id else db.recommended_shares

    if session_id:
        trades = list(collection.find({"session_id": session_id}))
    else:
        trades = list(
            collection.find(
                {
                    "strategy_name": strategy_name or "Swing_Trading",
                    "trade_type": {"$in": ["BUY", "SELL"]},
                }
            ).sort("entry_date", 1)
        )

    if not trades:
        return {"status": "error", "reason": "No trades found"}

    # Convert to DataFrame for analysis
    df = pd.DataFrame(trades)

    # Separate buys and sells
    buys = df[df.get("trade_type") == "BUY"] if "trade_type" in df.columns else pd.DataFrame()
    sells = df[df.get("trade_type") == "SELL"] if "trade_type" in df.columns else pd.DataFrame()

    results = {
        "total_trades": len(df),
        "total_buys": len(buys),
        "total_sells": len(sells),
    }

    # Trades per year
    if "entry_date" in df.columns:
        dates = pd.to_datetime(df["entry_date"])
        results["trades_per_year"] = {}
        for year in sorted(dates.dt.year.unique()):
            count = (dates.dt.year == year).sum()
            results["trades_per_year"][int(year)] = int(count)
        results["avg_trades_per_year"] = round(len(df) / len(dates.dt.year.unique()), 1)

    # Holding period analysis
    if "entry_date" in df.columns and "exit_date" in df.columns:
        entry_dates = pd.to_datetime(sells.get("entry_date", []))
        exit_dates = pd.to_datetime(sells.get("exit_date", []))
        valid = entry_dates.notna() & exit_dates.notna()
        if valid.any():
            holding_days = (exit_dates[valid] - entry_dates[valid]).dt.days
            results["holding_period"] = {
                "mean": round(float(holding_days.mean()), 1),
                "median": round(float(holding_days.median()), 1),
                "min": int(holding_days.min()),
                "max": int(holding_days.max()),
                "p90": round(float(holding_days.quantile(0.9)), 1),
            }

    # Consecutive losses
    if "pnl" in sells.columns:
        pnl_series = sells["pnl"].astype(float)
        win_loss = (pnl_series > 0).astype(int)
        max_consecutive_losses = 0
        current_streak = 0
        for wl in win_loss:
            if wl == 0:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        results["consecutive_losses"] = {
            "max": max_consecutive_losses,
            "win_rate": round(float((pnl_series > 0).mean() * 100), 1),
            "avg_win": round(float(pnl_series[pnl_series > 0].mean()), 2),
            "avg_loss": round(float(pnl_series[pnl_series <= 0].mean()), 2),
        }

        # Biggest winners and losers
        results["biggest_winners"] = (
            sells.nlargest(5, "pnl")[["symbol", "pnl", "pnl_pct", "exit_reason"]].to_dict("records")
            if len(sells) > 0
            else []
        )
        results["biggest_losers"] = (
            sells.nsmallest(5, "pnl")[["symbol", "pnl", "pnl_pct", "exit_reason"]].to_dict("records")
            if len(sells) > 0
            else []
        )

    # Exit reason breakdown
    if "exit_reason" in sells.columns:
        exit_counts = sells["exit_reason"].value_counts()
        results["exit_reasons"] = {}
        for reason, count in exit_counts.items():
            reason_trades = sells[sells["exit_reason"] == reason]
            win_rate = (reason_trades["pnl"].astype(float) > 0).mean() * 100
            results["exit_reasons"][reason] = {
                "count": int(count),
                "win_rate": round(float(win_rate), 1),
                "avg_pnl": round(float(reason_trades["pnl"].astype(float).mean()), 2),
            }

    return results


# ---------------------------------------------------------------------------
# 2. Equity Curve Analysis
# ---------------------------------------------------------------------------


def analyze_equity_curve(
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze portfolio equity curve from daily snapshots.

    Returns:
        Dict with equity curve metrics
    """
    db = get_mongodb()
    collection = db.portfolio_backtest_snapshots

    query = {}
    if session_id:
        query["session_id"] = session_id

    snapshots = list(collection.find(query).sort("date", 1))

    if not snapshots:
        return {"status": "error", "reason": "No snapshots found"}

    df = pd.DataFrame(snapshots)
    df["date"] = pd.to_datetime(df["date"])
    df["portfolio_value"] = df["portfolio_value"].astype(float)

    initial_capital = df["portfolio_value"].iloc[0]
    final_value = df["portfolio_value"].iloc[-1]

    # Daily returns
    df["daily_return"] = df["portfolio_value"].pct_change()

    # Underwater analysis (time below peak)
    df["peak_value"] = df["portfolio_value"].cummax()
    df["underwater_pct"] = ((df["peak_value"] - df["portfolio_value"]) / df["peak_value"]) * 100

    time_below_peak = (df["underwater_pct"] > 0).sum()
    total_days = len(df)

    # Recovery times
    drawdown_starts = []
    recovery_times = []
    in_dd = False
    dd_start = None

    for _, row in df.iterrows():
        if row["underwater_pct"] > 0 and not in_dd:
            in_dd = True
            dd_start = row["date"]
            drawdown_starts.append(dd_start)
        elif row["underwater_pct"] == 0 and in_dd:
            in_dd = False
            if dd_start:
                recovery_days = (row["date"] - dd_start).days
                recovery_times.append(recovery_days)

    # Monthly returns
    df["month"] = df["date"].dt.to_period("M")
    monthly_values = df.groupby("month")["portfolio_value"].last()
    monthly_returns = monthly_values.pct_change().dropna() * 100

    # Monthly heatmap data
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    monthly_pivot = df.groupby(["year", "month_num"])["portfolio_value"].last().unstack()
    _ = monthly_pivot.pct_change(axis=1) * 100  # monthly_returns_pivot for future use

    results = {
        "initial_capital": round(initial_capital, 2),
        "final_value": round(final_value, 2),
        "total_return_pct": round((final_value / initial_capital - 1) * 100, 2),
        "trading_days": total_days,
        "time_below_peak_pct": round(time_below_peak / total_days * 100, 1),
        "avg_drawdown_pct": round(float(df["underwater_pct"].mean()), 2),
        "max_drawdown_pct": round(float(df["underwater_pct"].max()), 2),
        "recovery_times_days": {
            "mean": round(np.mean(recovery_times), 1) if recovery_times else 0,
            "median": round(np.median(recovery_times), 1) if recovery_times else 0,
            "max": max(recovery_times) if recovery_times else 0,
            "count": len(recovery_times),
        },
        "monthly_returns": {
            "mean": round(float(monthly_returns.mean()), 2),
            "median": round(float(monthly_returns.median()), 2),
            "best_month": round(float(monthly_returns.max()), 2),
            "worst_month": round(float(monthly_returns.min()), 2),
            "positive_months_pct": round(float((monthly_returns > 0).mean() * 100), 1),
        },
    }

    return results


# ---------------------------------------------------------------------------
# 3. Regime Performance Breakdown
# ---------------------------------------------------------------------------


def analyze_regime_performance(
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze performance split by bull/bear regime from daily snapshots."""
    db = get_mongodb()
    snapshots = list(
        db.portfolio_backtest_snapshots.find({"session_id": session_id} if session_id else {}).sort("date", 1)
    )

    if not snapshots:
        return {"status": "error", "reason": "No snapshots found"}

    df = pd.DataFrame(snapshots)
    df["date"] = pd.to_datetime(df["date"])
    df["portfolio_value"] = df["portfolio_value"].astype(float)
    df["daily_return"] = df["portfolio_value"].pct_change()

    # Need regime data — check if snapshots include regime info
    if "regime" not in df.columns:
        # Approximate: positive returns = bull, negative = bear (rough)
        df["regime"] = df["daily_return"].apply(lambda x: "bull" if pd.notna(x) and x >= 0 else "bear")

    results = {}
    for regime in ["bull", "bear"]:
        regime_df = df[df["regime"] == regime]
        if len(regime_df) == 0:
            continue

        returns = regime_df["daily_return"].dropna()
        results[regime] = {
            "days": len(regime_df),
            "avg_daily_return": round(float(returns.mean()) * 100, 4),
            "volatility": round(float(returns.std()) * 100, 4),
            "positive_days_pct": round(float((returns > 0).mean() * 100), 1),
        }

    return results


# ---------------------------------------------------------------------------
# Combined Diagnostics
# ---------------------------------------------------------------------------


def run_all_diagnostics(
    session_id: Optional[str] = None,
    strategy_name: str = "Swing_Trading",
) -> Dict[str, Any]:
    """Run all trade diagnostics and return combined results."""
    print(f"\n{'='*70}")
    print("TRADE DIAGNOSTICS")
    print(f"{'='*70}\n")

    trade_dist = analyze_trade_distribution(session_id=session_id, strategy_name=strategy_name)
    equity_curve = analyze_equity_curve(session_id=session_id)
    regime_perf = analyze_regime_performance(session_id=session_id)

    # Print summary
    if "status" not in trade_dist:
        print(f"Total trades: {trade_dist.get('total_trades', 'N/A')}")
        print(f"Avg trades/year: {trade_dist.get('avg_trades_per_year', 'N/A')}")
        if "holding_period" in trade_dist:
            hp = trade_dist["holding_period"]
            print(f"Holding period: median {hp['median']} days, mean {hp['mean']} days")
        if "consecutive_losses" in trade_dist:
            cl = trade_dist["consecutive_losses"]
            print(f"Max consecutive losses: {cl['max']}")
            print(f"Win rate: {cl['win_rate']}%")

    if "status" not in equity_curve:
        print("\nEquity curve:")
        print(f"  Initial: Rs {equity_curve['initial_capital']:,.0f}")
        print(f"  Final: Rs {equity_curve['final_value']:,.0f}")
        print(f"  Time below peak: {equity_curve['time_below_peak_pct']}%")
        print(f"  Max drawdown: {equity_curve['max_drawdown_pct']}%")
        rt = equity_curve["recovery_times_days"]
        if rt["count"] > 0:
            print(f"  Avg recovery: {rt['mean']} days (max {rt['max']} days)")

    return {
        "trade_distribution": trade_dist,
        "equity_curve": equity_curve,
        "regime_performance": regime_perf,
    }
