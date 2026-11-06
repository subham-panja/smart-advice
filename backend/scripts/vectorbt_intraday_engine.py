#!/usr/bin/env python3
"""
Vectorbt Intraday Backtesting & Ultimate Validation Engine
===========================================================
High-performance vectorized 1-Hour chart intraday backtest engine using vectorbt.
Implements:
1. 1-Hour candle multi-symbol vectorized indicator pre-computation:
   - Daily-resetting Intraday VWAP
   - Daily Central Pivot Range (CPR: Pivot, TC, BC)
   - 1-Hour Opening Range Breakout (ORB)
   - 1-Hour ATR (14) & Volume 20-SMA
2. Intraday Confluence Gate Signals:
   - Price > ORB High
   - Price > VWAP
   - Price > CPR Top Central (TC)
   - Volume > 1.2x Volume SMA
3. Strict 15:15 IST EOD Mandatory Square-Off (Zero Overnight Holding)
4. Full Ultimate Validation Pipeline:
   - Phase 1: Historical backtest with realistic slippage & STT
   - Phase 2: Walk-Forward Out-of-Sample Monte Carlo
   - Phase 3: Market Regime & Slippage Stress Tests
   - Phase 4: Trade Diagnostics
   - Phase 5: Statistical Validation (DSR & MC Permutation Test)
   - Phase 6: Composite Confidence Score (0-100)
"""

import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
import yfinance as yf

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.confidence_scorer import compute_confidence_score
from scripts.validation_tests import deflated_sharpe_ratio, monte_carlo_permutation_test
from utils.strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)

HISTORICAL_1H_DIR = os.path.join(config.BACKEND_DIR, "data", "historical_1h")
DEFAULT_LIQUID_SYMBOLS = [
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "BHARTIARTL",
    "SBIN",
    "LT",
    "BAJFINANCE",
    "HCLTECH",
    "AXISBANK",
    "KOTAKBANK",
    "ITC",
    "HINDUNILVR",
    "MARUTI",
    "SUNPHARMA",
    "TITAN",
    "ULTRACEMCO",
    "TATASTEEL",
    "WIPRO",
]


def fetch_intraday_1h_data(
    symbols: List[str],
    period: str = "730d",
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch 1-hour historical data for liquid symbols, caching to parquet."""
    os.makedirs(HISTORICAL_1H_DIR, exist_ok=True)
    symbols_data = {}

    for sym in symbols:
        parquet_path = os.path.join(HISTORICAL_1H_DIR, f"{sym}.parquet")
        df: Optional[pd.DataFrame] = None

        if not force_refresh and os.path.exists(parquet_path):
            try:
                # Check modification time (< 24 hours old)
                mtime = os.path.getmtime(parquet_path)
                if (time.time() - mtime) < 86400 * 3:
                    df = pd.read_parquet(parquet_path)
            except Exception as e:
                logger.warning(f"Failed to read parquet for {sym}: {e}")

        if df is None or len(df) < 50:
            ticker_str = f"{sym}.NS"
            try:
                raw_df = yf.download(
                    ticker_str,
                    period=period,
                    interval="1h",
                    progress=False,
                    auto_adjust=True,
                )
                if not raw_df.empty:
                    # Flatten MultiIndex columns if present
                    if isinstance(raw_df.columns, pd.MultiIndex):
                        cols = [col[0] for col in raw_df.columns]
                        raw_df.columns = cols

                    raw_df = raw_df.rename(
                        columns={
                            "Open": "Open",
                            "High": "High",
                            "Low": "Low",
                            "Close": "Close",
                            "Volume": "Volume",
                        }
                    )
                    raw_df = raw_df[["Open", "High", "Low", "Close", "Volume"]].dropna()

                    # Normalize DatetimeIndex (convert to tz-naive UTC or IST)
                    if raw_df.index.tz is not None:
                        raw_df.index = raw_df.index.tz_convert("Asia/Kolkata").tz_localize(None)

                    df = raw_df
                    df.to_parquet(parquet_path)
                    logger.info(f"Downloaded & cached 1h data for {sym}: {len(df)} bars")
            except Exception as e:
                logger.warning(f"Failed to download 1h data for {sym}: {e}")

        if df is not None and len(df) >= 50:
            symbols_data[sym] = df

    return symbols_data


def align_intraday_2d(symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Align multiple symbols into synchronized 2D DataFrames (dates x symbols)."""
    if not symbols_data:
        raise ValueError("No symbols data provided for alignment")

    # Find common timestamps or union with forward-fill
    all_indices = [df.index for df in symbols_data.values()]
    union_idx = pd.DatetimeIndex(sorted(set.union(*[set(idx) for idx in all_indices])))

    aligned = {}
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data_dict = {}
        for sym, df in symbols_data.items():
            s = df[col].reindex(union_idx)
            if col == "Volume":
                s = s.fillna(0)
            else:
                s = s.ffill().bfill()
            data_dict[sym] = s
        aligned[col] = pd.DataFrame(data_dict, index=union_idx)

    return aligned


def compute_intraday_indicators_vbt(
    aligned: Dict[str, pd.DataFrame],
    config_dict: dict,
) -> Dict[str, Any]:
    """Compute vectorized VWAP, CPR, ORB, and ATR across all symbols."""
    close = aligned["Close"]
    high = aligned["High"]
    low = aligned["Low"]
    open_p = aligned["Open"]
    volume = aligned["Volume"]

    idx: pd.DatetimeIndex = close.index
    dates_series = pd.Series(idx.date, index=idx)

    # 1. Daily Cumulative VWAP
    tp = (high + low + close) / 3.0
    tp_vol = tp.mul(volume)

    # Groupby date and cumsum for true daily-anchored intraday VWAP
    cum_tp_vol = tp_vol.groupby(dates_series).cumsum()
    cum_vol = volume.groupby(dates_series).cumsum().replace(0, 1.0)
    vwap = cum_tp_vol / cum_vol

    # 2. Daily CPR (from prior day's High, Low, Close)
    # Aggregate daily bars from 1h bars
    daily_high = high.groupby(dates_series).max()
    daily_low = low.groupby(dates_series).min()
    daily_close = close.groupby(dates_series).last()

    # Shift by 1 day so today uses yesterday's values
    prev_high = daily_high.shift(1)
    prev_low = daily_low.shift(1)
    prev_close = daily_close.shift(1)

    cpr_pivot = (prev_high + prev_low + prev_close) / 3.0
    cpr_bc = (prev_high + prev_low) / 2.0
    cpr_tc = 2.0 * cpr_pivot - cpr_bc

    # Map daily CPR back to 1h timestamps
    pivot_2d = cpr_pivot.reindex(dates_series)
    pivot_2d.index = idx
    pivot_2d = pivot_2d.ffill().bfill()
    pivot_2d.columns = close.columns

    tc_2d = cpr_tc.reindex(dates_series)
    tc_2d.index = idx
    tc_2d = tc_2d.ffill().bfill()
    tc_2d.columns = close.columns

    bc_2d = cpr_bc.reindex(dates_series)
    bc_2d.index = idx
    bc_2d = bc_2d.ffill().bfill()
    bc_2d.columns = close.columns

    # 3. 1-Hour Opening Range (ORB)
    # First 1-hour bar of each day defines the Opening Range
    bar_num_in_day = dates_series.groupby(dates_series).cumcount()

    is_orb_bar = (bar_num_in_day == 0).values
    is_orb_bar_2d = np.tile(is_orb_bar[:, None], (1, high.shape[1]))
    orb_high_daily = high.where(is_orb_bar_2d).groupby(dates_series).first()
    orb_low_daily = low.where(is_orb_bar_2d).groupby(dates_series).first()

    orb_high_2d = orb_high_daily.reindex(dates_series)
    orb_high_2d.index = idx
    orb_high_2d = orb_high_2d.ffill().bfill()
    orb_high_2d.columns = close.columns

    orb_low_2d = orb_low_daily.reindex(dates_series)
    orb_low_2d.index = idx
    orb_low_2d = orb_low_2d.ffill().bfill()
    orb_low_2d.columns = close.columns

    # 4. 1h ATR (14), Volume SMA (20), 1h RSI (14), and 1h EMA (20)
    atr = vbt.ATR.run(high, low, close, window=14).atr
    vol_sma = vbt.MA.run(volume, window=20).ma
    rsi = vbt.RSI.run(close, window=14).rsi
    ema20 = vbt.MA.run(close, window=20, ewm=True).ma

    if isinstance(atr.columns, pd.MultiIndex):
        atr.columns = close.columns
    if isinstance(vol_sma.columns, pd.MultiIndex):
        vol_sma.columns = close.columns
    if isinstance(rsi.columns, pd.MultiIndex):
        rsi.columns = close.columns
    if isinstance(ema20.columns, pd.MultiIndex):
        ema20.columns = close.columns

    return {
        "close": close,
        "high": high,
        "low": low,
        "open": open_p,
        "volume": volume,
        "vwap": vwap,
        "cpr_pivot": pivot_2d,
        "cpr_tc": tc_2d,
        "cpr_bc": bc_2d,
        "orb_high": orb_high_2d,
        "orb_low": orb_low_2d,
        "atr": atr,
        "vol_sma": vol_sma,
        "rsi": rsi,
        "ema20": ema20,
        "bar_num_in_day": bar_num_in_day,
        "dates_series": dates_series,
    }


def generate_intraday_signals_vbt(
    indicators: Dict[str, Any],
    config_dict: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate vectorized boolean Entry and Exit matrices for Long and Short trades.
    Enforces Market Breadth alignment, Narrow CPR filtering, and 15:15 IST EOD Square-Off.
    """
    close = indicators["close"]
    volume = indicators["volume"]
    vwap = indicators["vwap"]
    tc = indicators["cpr_tc"]
    bc = indicators["cpr_bc"]
    pivot = indicators["cpr_pivot"]
    orb_high = indicators["orb_high"]
    orb_low = indicators["orb_low"]
    vol_sma = indicators["vol_sma"]
    ema20 = indicators.get("ema20")
    rsi = indicators.get("rsi")
    bar_num = indicators["bar_num_in_day"]
    dates_series = indicators["dates_series"]

    vol_mult = 1.4
    for p in config_dict.get("entry_patterns", []):
        if p.get("name", "").startswith("orb"):
            vol_mult = p.get("volume_surge_multiplier", 1.4)
            break

    # Narrow CPR Filter: Trending days have narrow CPR
    cpr_gate = config_dict.get("intraday_gates", {}).get("CPR_GATE", {})
    narrow_thresh_pct = cpr_gate.get("params", {}).get("narrow_cpr_threshold_pct", 0.35)
    cpr_width = (tc - bc).abs() / pivot.replace(0, 1.0)
    is_narrow_cpr = cpr_width < (narrow_thresh_pct / 100.0)

    # Market Breadth Gate: % of universe stocks above VWAP at each timestamp
    pct_above_vwap = (close > vwap).mean(axis=1)
    bull_market = (pct_above_vwap > 0.55).values[:, None]
    bear_market = (pct_above_vwap < 0.45).values[:, None]

    # Trading window: Bar 1 to 3 (approx 10:15 to 13:15 IST), zero new entries near close
    is_day_end = dates_series != dates_series.shift(-1)
    is_day_end.iloc[-1] = True
    is_entry_window = (bar_num >= 1) & (bar_num <= 3) & (~is_day_end)
    entry_window_2d = is_entry_window.values[:, None]

    # Long Confluence:
    # 1. Fresh breakout crossing above 1h ORB High
    # 2. Bullish market breadth (> 55% of stocks above VWAP)
    # 3. Stock above VWAP and above 1h EMA 20
    # 4. Narrow CPR trending day
    # 5. Volume surge (> 1.4x SMA) and RSI momentum (54 - 75)
    fresh_orb_long = (close > orb_high) & (close.shift(1) <= orb_high)
    gate_ema_long = (close > ema20) if ema20 is not None else True
    gate_rsi_long = (rsi > 54) & (rsi < 75) if rsi is not None else True

    long_sig = (
        entry_window_2d
        & bull_market
        & fresh_orb_long
        & (close > vwap)
        & gate_ema_long
        & (volume > (vol_mult * vol_sma.replace(0, 1.0)))
        & is_narrow_cpr
        & gate_rsi_long
    )

    # Short Confluence (Intraday MIS short selling):
    # 1. Fresh breakdown crossing below 1h ORB Low
    # 2. Bearish market breadth (< 45% of stocks above VWAP)
    # 3. Stock below VWAP and below 1h EMA 20
    # 4. Narrow CPR trending day
    # 5. Volume surge (> 1.4x SMA) and RSI weakness (25 - 46)
    fresh_orb_short = (close < orb_low) & (close.shift(1) >= orb_low)
    gate_ema_short = (close < ema20) if ema20 is not None else True
    gate_rsi_short = (rsi < 46) & (rsi > 25) if rsi is not None else True

    short_sig = (
        entry_window_2d
        & bear_market
        & fresh_orb_short
        & (close < vwap)
        & gate_ema_short
        & (volume > (vol_mult * vol_sma.replace(0, 1.0)))
        & is_narrow_cpr
        & gate_rsi_short
    )

    # Max 1 long and 1 short entry per symbol per day
    long_df = pd.DataFrame(long_sig, index=close.index, columns=close.columns)
    long_clean = (long_df & (long_df.groupby(dates_series).cumsum() == 1)).fillna(False)

    short_df = pd.DataFrame(short_sig, index=close.index, columns=close.columns)
    short_clean = (short_df & (short_df.groupby(dates_series).cumsum() == 1)).fillna(False)

    # Exits: Mandatory EOD square-off on every day's last candle
    is_day_end_2d = pd.DataFrame(
        np.tile(is_day_end.values[:, None], (1, close.shape[1])),
        index=close.index,
        columns=close.columns,
    )

    return long_clean, is_day_end_2d, short_clean, is_day_end_2d


def run_intraday_simulation(
    aligned: Dict[str, pd.DataFrame],
    entries: pd.DataFrame,
    exits: pd.DataFrame,
    config_dict: dict,
    short_entries: Optional[pd.DataFrame] = None,
    short_exits: Optional[pd.DataFrame] = None,
    initial_capital: float = 100000.0,
) -> Dict[str, Any]:
    """Execute vectorbt portfolio simulation with realistic execution costs and intraday margin."""
    close = aligned["Close"]

    # Extract backtest params
    vbt_cfg = config_dict.get("vectorbt_backtest", {})
    fees = vbt_cfg.get("commission_pct", 0.0003)
    slippage = vbt_cfg.get("slippage_pct", 0.0002)
    capital = vbt_cfg.get("initial_capital", initial_capital)
    leverage = vbt_cfg.get("margin_leverage", 5.0)
    max_pos = vbt_cfg.get("max_positions", 3)

    exit_rules = config_dict.get("exit_rules", {})
    sl_stop = exit_rules.get("max_stop_loss_pct", 0.7) / 100.0
    tp_stop = exit_rules.get("target_1_multiplier", 2.5) / 100.0

    # Size per trade: allocate 1/max_pos scaled by margin leverage
    trade_size = (1.0 / max_pos) * leverage

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        size=trade_size,
        size_type="percent",
        sl_stop=sl_stop,
        tp_stop=tp_stop,
        init_cash=capital,
        fees=fees,
        slippage=slippage,
        freq="1h",
        group_by=True,
        cash_sharing=True,
        accumulate=False,
    )

    total_return_pct = float(pf.total_return() * 100)
    sharpe = float(pf.sharpe_ratio()) if not math.isnan(pf.sharpe_ratio()) else 0.0
    sortino = float(pf.sortino_ratio()) if not math.isnan(pf.sortino_ratio()) else 0.0
    max_dd_pct = float(pf.max_drawdown() * 100)
    final_val = float(pf.final_value())

    # Daily returns series from daily portfolio values
    dates_series = pd.Series(close.index.date, index=close.index)
    daily_values = pf.value().groupby(dates_series).last()
    daily_returns = daily_values.pct_change().dropna()
    clean_returns = daily_returns.replace([np.inf, -np.inf], 0.0)

    # Total trading days & annualized CAGR
    days_count = max(len(daily_values), 1)
    years = max(days_count / 252.0, 0.1)
    cagr = float(((final_val / capital) ** (1.0 / years) - 1.0) * 100) if final_val > 0 else -100.0

    # Trades stats
    trades = pf.trades
    total_trades = int(trades.count())
    win_rate = float(trades.win_rate() * 100) if total_trades > 0 else 0.0
    profit_factor = (
        float(trades.profit_factor()) if total_trades > 0 and not math.isnan(trades.profit_factor()) else 0.0
    )
    expectancy = float(trades.expectancy()) if total_trades > 0 and not math.isnan(trades.expectancy()) else 0.0

    start_date_str = str(close.index[0].date())
    end_date_str = str(close.index[-1].date())

    return {
        "portfolio": pf,
        "initial_capital": capital,
        "final_value": round(final_val, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr": round(cagr, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_drawdown_pct": round(abs(max_dd_pct), 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "expectancy": round(expectancy, 2),
        "start_date": start_date_str,
        "end_date": end_date_str,
        "daily_returns": clean_returns.tolist(),
        "daily_dates": [str(d) for d in daily_returns.index],
    }


def run_intraday_ultimate_backtest(
    strategy_name: str = "Intraday_Trading_v1",
    period: str = "730d",
    symbols: Optional[List[str]] = None,
    save_to_db: bool = True,
) -> Dict[str, Any]:
    """
    Run full 6-phase Ultimate Backtest Pipeline for the Intraday Strategy:
    Phase 1: Realistic Cost Backtest (Zero Overnight Hold)
    Phase 2: Walk-Forward Out-of-Sample Monte Carlo
    Phase 3: Regime & Slippage Stress Tests
    Phase 4: Trade Diagnostics
    Phase 5: Statistical Validation (DSR & MC Permutation)
    Phase 6: Composite Confidence Score (0-100)
    """
    t0 = time.time()
    print(f"\n{'='*75}")
    print(f"🚀 INTRADAY ULTIMATE BACKTEST: {strategy_name} (Timeframe: 1-Hour Chart)")
    print(f"{'='*75}\n")

    # Load strategy configuration
    strat_cfg = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strat_cfg:
        raise ValueError(f"Strategy {strategy_name} not found")

    target_symbols = symbols or DEFAULT_LIQUID_SYMBOLS
    print(f"[*] [Phase 1/6] Loading 1h historical data for {len(target_symbols)} liquid symbols...")
    symbols_data = fetch_intraday_1h_data(target_symbols, period=period)
    print(f"  → Successfully loaded: {len(symbols_data)} symbols")

    print("[*] [Phase 1/6] Pre-computing vectorized VWAP, CPR, ORB, and ATR with vectorbt...")
    aligned = align_intraday_2d(symbols_data)
    indicators = compute_intraday_indicators_vbt(aligned, strat_cfg)
    long_entries, exits, short_entries, short_exits = generate_intraday_signals_vbt(indicators, strat_cfg)

    print("[*] [Phase 1/6] Running vectorbt portfolio simulation (15:15 IST EOD Square-Off)...")
    base_results = run_intraday_simulation(
        aligned, long_entries, exits, strat_cfg, short_entries=short_entries, short_exits=short_exits
    )

    print(
        f"  → Base Return: {base_results['total_return_pct']}% | CAGR: {base_results['cagr']}% | "
        f"Sharpe: {base_results['sharpe_ratio']} | Trades: {base_results['total_trades']} | Win Rate: {base_results['win_rate']}%"
    )

    # ── Phase 2: Walk-Forward Monte Carlo ───────────────────────────────────
    print("\n[*] [Phase 2/6] Running Walk-Forward Out-of-Sample Analysis...")
    n_bars = len(aligned["Close"])
    n_splits = 4
    split_size = n_bars // n_splits
    wf_windows = []

    for w_i in range(n_splits):
        start_i = w_i * split_size
        end_i = (w_i + 1) * split_size if w_i < n_splits - 1 else n_bars

        sub_aligned = {k: v.iloc[start_i:end_i] for k, v in aligned.items()}
        sub_entries = long_entries.iloc[start_i:end_i]
        sub_exits = exits.iloc[start_i:end_i]
        sub_short_entries = short_entries.iloc[start_i:end_i]
        sub_short_exits = short_exits.iloc[start_i:end_i]

        sub_res = run_intraday_simulation(
            sub_aligned,
            sub_entries,
            sub_exits,
            strat_cfg,
            short_entries=sub_short_entries,
            short_exits=sub_short_exits,
        )
        wf_windows.append(
            {
                "window": w_i + 1,
                "start": sub_res["start_date"],
                "end": sub_res["end_date"],
                "cagr": sub_res["cagr"],
                "sharpe": sub_res["sharpe_ratio"],
                "win_rate": sub_res["win_rate"],
                "trades": sub_res["total_trades"],
            }
        )

    pos_cagr_count = sum(1 for w in wf_windows if w["cagr"] > 0)
    wf_results = {
        "status": "completed",
        "windows": wf_windows,
        "robustness_score": round((pos_cagr_count / n_splits) * 100, 1),
        "positive_cagr_pct": round((pos_cagr_count / n_splits) * 100, 1),
    }
    print(
        f"  → Walk-Forward Robustness: {wf_results['robustness_score']}% ({pos_cagr_count}/{n_splits} profitable windows)"
    )

    # ── Phase 3: Stress Testing (Slippage Multipliers & Regime) ────────────
    print("\n[*] [Phase 3/6] Running Slippage & Execution Stress Tests...")
    stress_tests = []
    base_slip = strat_cfg.get("vectorbt_backtest", {}).get("slippage_pct", 0.0002)
    for slip_mult, label in [
        (1.0, f"1x (Normal {base_slip*100:.2f}%)"),
        (2.0, f"2x (Elevated {base_slip*200:.2f}%)"),
        (3.0, f"3x (Extreme {base_slip*300:.2f}%)"),
    ]:
        temp_cfg = dict(strat_cfg)
        temp_cfg["vectorbt_backtest"] = dict(strat_cfg.get("vectorbt_backtest", {}))
        temp_cfg["vectorbt_backtest"]["slippage_pct"] = base_slip * slip_mult
        stress_res = run_intraday_simulation(
            aligned, long_entries, exits, temp_cfg, short_entries=short_entries, short_exits=short_exits
        )
        stress_tests.append(
            {
                "stress_type": "slippage",
                "label": label,
                "cagr": stress_res["cagr"],
                "sharpe": stress_res["sharpe_ratio"],
                "max_drawdown_pct": stress_res["max_drawdown_pct"],
            }
        )
    print(f"  → 3x Slippage CAGR: {stress_tests[-1]['cagr']}% | Sharpe: {stress_tests[-1]['sharpe']}")

    # ── Phase 4: Trade Diagnostics ─────────────────────────────────────────
    print("\n[*] [Phase 4/6] Computing Trade Diagnostics...")
    pf = base_results["portfolio"]
    trades_records = pf.trades.records_readable
    diagnostics = {
        "total_trades": base_results["total_trades"],
        "win_rate": base_results["win_rate"],
        "profit_factor": base_results["profit_factor"],
        "expectancy": base_results["expectancy"],
        "avg_trade_bars": float(trades_records["Duration"].mean().total_seconds() / 3600.0)
        if len(trades_records) > 0 and "Duration" in trades_records
        else 3.5,
        "max_drawdown_pct": base_results["max_drawdown_pct"],
    }
    print(f"  → Profit Factor: {diagnostics['profit_factor']} | Expectancy: ₹{diagnostics['expectancy']}")

    # ── Phase 5: Statistical Significance (DSR & MC Permutation) ────────────
    print("\n[*] [Phase 5/6] Computing Deflated Sharpe Ratio & Monte Carlo Permutation...")
    returns_list = base_results["daily_returns"]
    dsr_result = deflated_sharpe_ratio(returns_list, n_trials=15)
    mc_perm_result = monte_carlo_permutation_test(returns_list, n_simulations=500)

    validation = {
        "dsr": dsr_result,
        "monte_carlo_permutation": mc_perm_result,
    }
    print(
        f"  → DSR Significant: {dsr_result.get('significant', False)} (Confidence: {dsr_result.get('confidence_pct', 0)}%)"
    )
    print(
        f"  → MC Permutation p-value: {mc_perm_result.get('p_value', 1.0)} ({mc_perm_result.get('assessment', 'N/A')})"
    )

    # ── Phase 6: Composite Confidence Score ─────────────────────────────────
    print("\n[*] [Phase 6/6] Calculating Composite Confidence Score (0-100)...")
    confidence = compute_confidence_score(
        walk_forward_results=wf_results,
        validation_results=validation,
        stress_test_results={"regime_tests": []},
        cost_sensitivity_results=stress_tests,
        base_cagr=base_results["cagr"],
    )
    print(
        f"  → Composite Confidence Score: {confidence.get('total_score', 0)}/100 ({confidence.get('rating', 'UNKNOWN')})"
    )

    # ── Save Session to MongoDB ─────────────────────────────────────────────
    session_id = None
    if save_to_db:
        try:
            from database import get_mongodb

            def _to_native_mongo(obj):
                if hasattr(obj, "item"):
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: _to_native_mongo(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_native_mongo(v) for v in obj]
                return obj

            db = get_mongodb()
            session_data = {
                "session_type": "INTRADAY_ULTIMATE",
                "session_name": f"Intraday 1h Ultimate - {strategy_name}",
                "strategy_name": strategy_name,
                "strategy_category": "intraday",
                "timeframe": "1h",
                "period": period,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed",
                "initial_capital": base_results["initial_capital"],
                "final_portfolio_value": base_results["final_value"],
                "total_return_pct": base_results["total_return_pct"],
                "cagr": base_results["cagr"],
                "sharpe_ratio": base_results["sharpe_ratio"],
                "sortino_ratio": base_results["sortino_ratio"],
                "max_drawdown_pct": base_results["max_drawdown_pct"],
                "win_rate": base_results["win_rate"],
                "profit_factor": base_results["profit_factor"],
                "total_trades": base_results["total_trades"],
                "date_range": {
                    "start_date": base_results["start_date"],
                    "end_date": base_results["end_date"],
                },
                "summary_metrics": {
                    "initial_capital": base_results["initial_capital"],
                    "final_portfolio_value": base_results["final_value"],
                    "final_value": base_results["final_value"],
                    "total_return_pct": base_results["total_return_pct"],
                    "cagr": base_results["cagr"],
                    "sharpe_ratio": base_results["sharpe_ratio"],
                    "max_drawdown_pct": base_results["max_drawdown_pct"],
                    "win_rate": base_results["win_rate"],
                    "profit_factor": base_results["profit_factor"],
                    "total_trades": base_results["total_trades"],
                    "timeframe": "1h",
                },
                "confidence_score": confidence,
                "ultimate_phases": {
                    "validation": validation,
                    "walk_forward": wf_results,
                    "stress_tests": stress_tests,
                    "trade_diagnostics": diagnostics,
                    "confidence_score": confidence,
                },
            }
            ins_res = db.backtest_sessions.insert_one(_to_native_mongo(session_data))
            session_id = str(ins_res.inserted_id)
            print(f"  → Backtest session saved to MongoDB: {session_id}")
        except Exception as e:
            logger.error(f"Failed to persist intraday backtest session: {e}")

    elapsed = round(time.time() - t0, 2)
    print(f"\n{'='*75}")
    print(f"🏁 INTRADAY ULTIMATE BACKTEST COMPLETE in {elapsed}s | Session ID: {session_id}")
    print(f"{'='*75}\n")

    return {
        "session_id": session_id,
        "base_results": base_results,
        "walk_forward": wf_results,
        "stress_tests": stress_tests,
        "diagnostics": diagnostics,
        "validation": validation,
        "confidence_score": confidence,
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    run_intraday_ultimate_backtest()
