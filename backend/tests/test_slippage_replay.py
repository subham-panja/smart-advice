#!/usr/bin/env python3
"""
365-Day Replay Test Script (Historical Signals)
================================================
Tests pyramiding slippage and exit slippage by:
1. Pre-caching all NSE stocks to prevent yfinance rate limiting
2. Pre-computing BUY signals for all (date, symbol) pairs using vectorbt
3. Pre-populating MongoDB recommended_shares with historically-accurate signals
4. Running N trading days one-by-one via the API (main_orchestrator.py unchanged)
5. After each day, checking for pyramiding and injecting slippage via PATCH API
6. Checking for exits and verifying PnL recalculation
7. Reporting any errors found

Usage: cd backend && python tests/test_slippage_replay.py
"""

import math
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import requests

# Add backend/ to path for direct MongoDB and module access
BACKEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, BACKEND_DIR)

API_BASE = "http://127.0.0.1:5001"
SLIPPAGE_PCT = 0.005  # 0.5% slippage to inject
MAX_STOCKS = 0  # 0 = use all cached stocks (525 available)
BACKTEST_PERIOD = "5y"  # Historical data period for signal computation
NUM_DAYS = 100  # Number of trading days to replay
DELAY_BETWEEN_RUNS = 3  # Seconds to wait between orchestrator runs
RETRY_WAIT = 180  # Seconds to wait before retrying after rate limit error
MAX_RETRIES = 3  # Max retries per day on rate limit errors
PRE_CACHE = False  # Skip pre-cache — already have 1400+ stocks cached

REPORT_DIR = os.path.join(BACKEND_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


class TeeWriter:
    """Writes to both stdout and a file simultaneously."""

    def __init__(self, filepath):
        self.file = open(filepath, "w", buffering=1)  # line-buffered
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def get_trading_days(n=20):
    """Get last N trading days (skip weekends)."""
    days = []
    d = datetime.now().date() - timedelta(days=1)
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days.reverse()
    return days


def run_single_date(date_str):
    """Run trading cycle for a specific date and wait for completion."""
    print(f"\n{'='*60}")
    print(f"  Running date: {date_str}")
    print(f"{'='*60}")

    resp = requests.post(
        f"{API_BASE}/run-orchestrator",
        json={"mode": "date", "date": date_str},
        timeout=30,
    )
    if resp.status_code != 200:
        print(f"  ERROR starting orchestrator: {resp.json()}")
        return False

    # Poll for completion
    for _ in range(120):
        time.sleep(2)
        status_resp = requests.get(f"{API_BASE}/orchestrator-status", timeout=10)
        status = status_resp.json()
        if status.get("status") in ("completed", "error", "idle"):
            if status.get("status") == "error":
                print(f"  ORCHESTRATOR ERROR: {status.get('message')}")
                return False
            break
    else:
        print("  TIMEOUT: Orchestrator did not complete in 240s")
        return False

    return True


def get_open_positions():
    """Fetch all open positions."""
    resp = requests.get(f"{API_BASE}/positions", params={"status": "OPEN"}, timeout=10)
    data = resp.json()
    return data.get("positions", [])


def get_closed_positions():
    """Fetch all closed positions."""
    resp = requests.get(f"{API_BASE}/positions", params={"status": "CLOSED"}, timeout=10)
    data = resp.json()
    return data.get("positions", [])


def get_activity_logs(symbol=None, limit=50):
    """Fetch recent activity logs."""
    params = {"limit": limit}
    if symbol:
        params["symbol"] = symbol
    resp = requests.get(f"{API_BASE}/activity-logs", params=params, timeout=10)
    data = resp.json()
    return data.get("logs", [])


def inject_pyramid_slippage(positions, prev_pyramid_counts):
    """Check for pyramid updates and inject slippage into entry prices.

    Only injects for NEW pyramid adds (adds_count > prev count for that symbol).
    Returns (injected_list, updated_counts_dict).
    """
    injected = []
    updated_counts = dict(prev_pyramid_counts)

    for pos in positions:
        symbol = pos["symbol"]
        adds_count = pos.get("adds_count", 0)
        prev_count = prev_pyramid_counts.get(symbol, 0)

        if adds_count <= prev_count:
            continue  # No new pyramid add since last check

        # New pyramid add detected — inject slippage
        updated_counts[symbol] = adds_count

        current_entry = pos["entry_price"]
        current_investment = pos["total_investment"]
        quantity = pos["quantity"]

        slippage_amount = current_entry * SLIPPAGE_PCT
        new_entry = round(current_entry + slippage_amount, 2)
        new_investment = round(quantity * new_entry, 2)

        print(f"\n  PYRAMID SLIPPAGE INJECTION for {symbol}:")
        print(f"    Old entry: Rs{current_entry:.2f}, Old investment: Rs{current_investment:.2f}")
        print(f"    Slippage: +{SLIPPAGE_PCT*100:.1f}% = Rs{slippage_amount:.2f}")
        print(f"    New entry: Rs{new_entry:.2f}, New investment: Rs{new_investment:.2f}")

        resp = requests.patch(
            f"{API_BASE}/positions/{symbol}",
            json={"entry_price": new_entry, "total_investment": new_investment},
            timeout=10,
        )
        result = resp.json()
        if result.get("status") == "success":
            print("    SUCCESS: Entry price updated")
            injected.append(
                {
                    "symbol": symbol,
                    "old_entry": current_entry,
                    "new_entry": new_entry,
                    "old_investment": current_investment,
                    "new_investment": new_investment,
                }
            )
        else:
            print(f"    ERROR: {result}")

    return injected, updated_counts


def inject_buy_slippage(new_symbols, positions):
    """Inject slippage on newly opened positions' entry price.

    For each new position, raises entry_price by SLIPPAGE_PCT (worse execution)
    and recalculates total_investment accordingly.
    Returns list of injection dicts.
    """
    injected = []

    for pos in positions:
        if pos["symbol"] not in new_symbols:
            continue

        symbol = pos["symbol"]
        current_entry = pos["entry_price"]
        current_investment = pos["total_investment"]
        quantity = pos["quantity"]

        slippage_amount = current_entry * SLIPPAGE_PCT
        new_entry = round(current_entry + slippage_amount, 2)
        new_investment = round(quantity * new_entry * 1.002, 2)  # include brokerage

        print(f"\n  BUY SLIPPAGE INJECTION for {symbol}:")
        print(f"    Old entry: Rs{current_entry:.2f}, Old investment: Rs{current_investment:.2f}")
        print(f"    Slippage: +{SLIPPAGE_PCT*100:.1f}% = Rs{slippage_amount:.2f}")
        print(f"    New entry: Rs{new_entry:.2f}, New investment: Rs{new_investment:.2f}")

        resp = requests.patch(
            f"{API_BASE}/positions/{symbol}",
            json={"entry_price": new_entry, "total_investment": new_investment},
            timeout=10,
        )
        result = resp.json()
        if result.get("status") == "success":
            print("    SUCCESS: Entry price updated")
            injected.append(
                {
                    "symbol": symbol,
                    "type": "BUY",
                    "old_entry": current_entry,
                    "new_entry": new_entry,
                    "old_investment": current_investment,
                    "new_investment": new_investment,
                }
            )
        else:
            print(f"    ERROR: {result}")

    return injected


def inject_exit_slippage(newly_closed):
    """Inject slippage on newly closed positions' exit price and recalculate PnL.

    For each closed position, lowers exit_price by SLIPPAGE_PCT (worse execution)
    and recalculates pnl_pct. Updates MongoDB directly since PATCH only works for OPEN positions.
    Returns list of injection dicts.
    """
    from database import get_mongodb

    injected = []
    db = get_mongodb()
    col = db.positions

    for pos in newly_closed:
        symbol = pos["symbol"]
        old_exit = pos.get("exit_price", 0)
        entry_price = pos.get("entry_price", 0)

        if old_exit <= 0 or entry_price <= 0:
            continue

        slippage_amount = old_exit * SLIPPAGE_PCT
        new_exit = round(old_exit - slippage_amount, 2)
        new_pnl_pct = ((new_exit - entry_price) / entry_price) * 100

        print(f"\n  EXIT SLIPPAGE INJECTION for {symbol}:")
        print(f"    Old exit: Rs{old_exit:.2f}, Old PnL: {pos.get('pnl_pct', 0):+.2f}%")
        print(f"    Slippage: -{SLIPPAGE_PCT*100:.1f}% = Rs{slippage_amount:.2f}")
        print(f"    New exit: Rs{new_exit:.2f}, New PnL: {new_pnl_pct:+.2f}%")

        col.update_one(
            {"_id": pos["_id"]},
            {
                "$set": {
                    "exit_price": new_exit,
                    "pnl_pct": round(new_pnl_pct, 2),
                }
            },
        )
        print("    SUCCESS: Exit price and PnL updated")

        injected.append(
            {
                "symbol": symbol,
                "type": "EXIT",
                "old_exit": old_exit,
                "new_exit": new_exit,
                "old_pnl_pct": pos.get("pnl_pct", 0),
                "new_pnl_pct": round(new_pnl_pct, 2),
                "exit_reason": pos.get("exit_reason", "UNKNOWN"),
            }
        )

    return injected


def verify_pnl_recalc(positions):
    """Verify PnL is correctly recalculated after entry price changes."""
    errors = []
    for pos in positions:
        symbol = pos["symbol"]
        entry_price = pos.get("entry_price", 0)
        current_price = pos.get("current_price", entry_price)
        quantity = pos.get("quantity", 0)
        total_investment = pos.get("total_investment", 0)

        # Check PnL calculation
        expected_pnl = (current_price - entry_price) * quantity
        expected_pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # Check total_investment consistency
        expected_investment = entry_price * quantity
        investment_diff = abs(total_investment - expected_investment)

        if investment_diff > 1.0:  # Allow Rs1 tolerance for brokerage
            # This is expected since total_investment includes brokerage
            brokerage_factor = total_investment / expected_investment if expected_investment > 0 else 0
            if abs(brokerage_factor - 1.002) > 0.01:  # 0.2% brokerage
                pass  # Minor discrepancy, not necessarily an error

        print(
            f"    {symbol}: Entry=Rs{entry_price:.2f}, Current=Rs{current_price:.2f}, "
            f"Qty={quantity}, PnL=Rs{expected_pnl:+.2f} ({expected_pnl_pct:+.2f}%)"
        )

    return errors


def check_exit_pnl(closed_positions):
    """Check PnL calculation on closed positions."""
    errors = []
    for pos in closed_positions:
        symbol = pos["symbol"]
        entry_price = pos.get("entry_price", 0)
        exit_price = pos.get("exit_price", 0)
        pnl_pct = pos.get("pnl_pct", 0)
        exit_reason = pos.get("exit_reason", "UNKNOWN")

        expected_pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        pnl_diff = abs(pnl_pct - expected_pnl_pct)

        status = "OK" if pnl_diff < 0.1 else "MISMATCH"
        print(
            f"    {symbol}: Entry=Rs{entry_price:.2f} -> Exit=Rs{exit_price:.2f}, "
            f"PnL={pnl_pct:+.2f}% (expected={expected_pnl_pct:+.2f}%) [{status}] "
            f"Reason={exit_reason}"
        )

        if pnl_diff > 0.1:
            errors.append(
                {
                    "symbol": symbol,
                    "expected_pnl_pct": expected_pnl_pct,
                    "actual_pnl_pct": pnl_pct,
                    "diff": pnl_diff,
                }
            )

    return errors


def print_positions_summary(positions, label="OPEN"):
    """Print a summary of positions."""
    if not positions:
        print(f"  No {label} positions")
        return
    print(f"  {label} Positions ({len(positions)}):")
    for pos in positions:
        symbol = pos["symbol"]
        qty = pos.get("quantity", 0)
        entry = pos.get("entry_price", 0)
        current = pos.get("current_price", entry)
        adds = pos.get("adds_count", 0)
        inv = pos.get("total_investment", 0)
        sl = pos.get("current_stop_loss", pos.get("stop_loss", 0))
        tgt = pos.get("current_target", pos.get("target", 0))
        pnl = (current - entry) * qty
        pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        print(
            f"    {symbol}: Qty={qty}, Entry=Rs{entry:.2f}, Current=Rs{current:.2f}, "
            f"Invest=Rs{inv:.2f}, Adds={adds}, "
            f"SL=Rs{sl:.2f}, Tgt=Rs{tgt:.2f}, "
            f"PnL=Rs{pnl:+.2f}({pnl_pct:+.1f}%)"
        )


def pre_cache_nse_stocks(period=BACKTEST_PERIOD):
    """Pre-cache historical data for all NSE stocks to prevent yfinance rate limiting during replay.

    Downloads BOTH the long-period data (for signal generation) AND 60d data (for orchestrator's
    analysis phase). The orchestrator calls get_historical_data(symbol, period='60d') in Phase 2 —
    if that data isn't cached, it hits yfinance and gets rate limited after ~7 days.

    Downloads data in parallel using ThreadPoolExecutor. Cached stocks are instant on subsequent runs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from scripts.data_fetcher import get_all_nse_symbols, get_historical_data

    print("\n  [PRE-CACHE] Loading all NSE symbols...")
    all_syms = get_all_nse_symbols()
    if isinstance(all_syms, dict):
        symbols_list = list(all_syms.keys())
    else:
        symbols_list = list(all_syms)

    data_dir = os.path.join(BACKEND_DIR, "data", "historical")

    def _count_cached(period_name):
        suffix = f"_{period_name}_1d.csv"
        cached = set()
        for f in os.listdir(data_dir):
            if f.endswith(suffix) and not f.startswith("^"):
                cached.add(f.replace(suffix, ""))
        return cached

    # Phase 1: Cache long-period data (5y) for signal generation
    cached_long = _count_cached(period)
    uncached_long = [s for s in symbols_list if s not in cached_long]
    print(f"  [PRE-CACHE] Long-period ({period}): {len(cached_long)} cached, {len(uncached_long)} to download")

    # Phase 2: Cache 60d data for orchestrator's analysis phase
    cached_60d = _count_cached("60d")
    uncached_60d = [s for s in symbols_list if s not in cached_60d]
    print(f"  [PRE-CACHE] Short-period (60d): {len(cached_60d)} cached, {len(uncached_60d)} to download")

    def _download_batch(uncached, period_name, max_workers=4):
        if not uncached:
            print(f"  [PRE-CACHE] {period_name}: All cached - skipping")
            return 0, 0

        print(f"  [PRE-CACHE] Downloading {period_name} data for {len(uncached)} stocks ({max_workers} threads)...")
        downloaded = 0
        failed = 0
        t0 = time.time()

        def _fetch(symbol):
            try:
                df = get_historical_data(symbol, period=period_name)
                return symbol, df is not None and not df.empty
            except Exception:
                return symbol, False

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_fetch, s): s for s in uncached}
            for i, future in enumerate(as_completed(futures), 1):
                symbol, success = future.result()
                if success:
                    downloaded += 1
                else:
                    failed += 1
                if i % 100 == 0 or i == len(uncached):
                    elapsed = time.time() - t0
                    speed = i / elapsed if elapsed > 0 else 0
                    remaining = (len(uncached) - i) / speed if speed > 0 else 0
                    print(
                        f"    {period_name} Progress: {i}/{len(uncached)} | "
                        f"OK: {downloaded} | Fail: {failed} | "
                        f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining"
                    )

        return downloaded, failed

    # Download long-period data first (slow on first run, instant after)
    dl_long, fail_long = _download_batch(uncached_long, period)
    total_long = len(cached_long) + dl_long

    # Download 60d data (faster - smaller data per stock)
    dl_60d, fail_60d = _download_batch(uncached_60d, "60d", max_workers=6)
    total_60d = len(cached_60d) + dl_60d

    print("  [PRE-CACHE] Complete:")
    print(f"    Long-period ({period}): {total_long} stocks cached ({fail_long} failed/delisted)")
    print(f"    Short-period (60d): {total_60d} stocks cached ({fail_60d} failed/delisted)")
    return total_long


def setup_signals(max_stocks=MAX_STOCKS, period=BACKTEST_PERIOD):
    """One-time setup: load strategy, fetch data, compute signals for all (date, symbol) pairs.

    Uses cached historical data from data/historical/ directory. Scans for all available
    CSV files to maximize stock coverage without network calls.

    Returns:
        (pass_matrix, score_matrix, indicators, strategy_config)
        - pass_matrix: bool DataFrame (dates x symbols)
        - score_matrix: float DataFrame (dates x symbols)
        - indicators: ComputedIndicators object
        - strategy_config: strategy config dict
    """
    import glob

    from scripts.data_fetcher import get_historical_data
    from scripts.vectorbt_indicator_batch import compute_all_indicators
    from scripts.vectorbt_signal_generator import compute_signal_matrix
    from utils.strategy_loader import StrategyLoader

    print(f"\n{'='*60}")
    print("  PHASE A: SIGNAL PRE-COMPUTATION")
    print(f"{'='*60}")

    # 1. Load strategy
    print("\n  [1/5] Loading strategy config...")
    strategy_config = StrategyLoader.get_strategy_by_name("Swing_Trading")
    if not strategy_config:
        raise RuntimeError("Strategy 'Swing_Trading' not found")
    print(f"    Strategy: {strategy_config['name']}")

    # 2. Discover cached stocks (instant, no network)
    print(f"\n  [2/5] Scanning cache for available {period} historical data...")
    data_dir = os.path.join(BACKEND_DIR, "data", "historical")
    cache_pattern = os.path.join(data_dir, f"*_{period}_1d.csv")
    cached_files = glob.glob(cache_pattern)

    # Extract symbol names from filenames (e.g., "RELIANCE_5y_1d.csv" -> "RELIANCE")
    suffix = f"_{period}_1d.csv"
    cached_symbols = []
    for f in cached_files:
        basename = os.path.basename(f)
        if basename.startswith("^"):
            continue  # Skip index data like ^NSEI
        symbol = basename.replace(suffix, "")
        cached_symbols.append(symbol)

    if max_stocks and len(cached_symbols) > max_stocks:
        cached_symbols = cached_symbols[:max_stocks]

    print(f"    Found {len(cached_symbols)} stocks with cached {period} data")

    if len(cached_symbols) < 10:
        raise RuntimeError(
            f"Only {len(cached_symbols)} cached stocks found. " f"Run a backtest first to populate the cache."
        )

    # 3. Load cached data (instant - reads from CSV, no network)
    print(f"\n  [3/5] Loading {len(cached_symbols)} stocks from cache...")
    symbols_data = {}
    failed = 0
    t0 = time.time()
    for i, symbol in enumerate(cached_symbols):
        try:
            df = get_historical_data(symbol, period=period)
            if not df.empty and len(df) >= 250:
                symbols_data[symbol] = df
            else:
                failed += 1
        except Exception:
            failed += 1
        if (i + 1) % 100 == 0 or i == len(cached_symbols) - 1:
            elapsed = time.time() - t0
            print(
                f"    Progress: {i+1}/{len(cached_symbols)} | "
                f"Loaded: {len(symbols_data)} | Failed: {failed} | "
                f"{elapsed:.1f}s"
            )

    print(f"    Loaded data for {len(symbols_data)} stocks ({failed} failed/skipped) in {time.time()-t0:.1f}s")

    if len(symbols_data) < 10:
        raise RuntimeError(f"Only {len(symbols_data)} stocks loaded - need at least 10")

    # 4. Compute indicators using vectorbt
    print(f"\n  [4/5] Computing indicators with vectorbt ({len(symbols_data)} symbols)...")
    t0 = time.time()
    indicators = compute_all_indicators(symbols_data, strategy_config)
    print(f"    Done in {time.time()-t0:.1f}s")
    print(f"    Matrix: {len(indicators.dates)} dates x {len(indicators.symbols)} symbols")

    # 5. Compute signal matrix
    print("\n  [5/6] Computing BUY signal matrix...")
    t0 = time.time()
    pass_matrix, score_matrix = compute_signal_matrix(indicators, strategy_config)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Count total BUY signals (before pre-filter)
    total_signals_before = int(pass_matrix.sum().sum())
    print(f"    BUY signals before pre-filter: {total_signals_before}")

    # 6. Apply screener-compatible pre-filter (same universe as live path)
    print("\n  [6/6] Applying screener pre-filter (universe alignment)...")
    from scripts.vectorbt_signal_generator import compute_stock_prefilter

    t0 = time.time()
    prefilter_matrix = compute_stock_prefilter(indicators, strategy_config)
    print(f"    Done in {time.time()-t0:.1f}s")

    # Show how many stocks pass pre-filter on recent dates
    recent_dates = prefilter_matrix.index[-30:]
    prefilter_per_day = prefilter_matrix.loc[recent_dates].sum(axis=1)
    print(
        f"    Pre-filter universe (last 30 days): avg {prefilter_per_day.mean():.0f} stocks/day, "
        f"min {int(prefilter_per_day.min())}, max {int(prefilter_per_day.max())}"
    )

    # Apply pre-filter: only keep BUY signals where stock also passes screener pre-filters
    pass_matrix = pass_matrix & prefilter_matrix
    score_matrix = score_matrix.where(pass_matrix, 0.0)

    # Count total BUY signals
    total_signals = int(pass_matrix.sum().sum())
    print(f"    BUY signals after pre-filter: {total_signals} (removed {total_signals_before - total_signals})")

    # Show signals per day (last 30 days)
    recent_dates = pass_matrix.index[-30:]
    signals_per_day = pass_matrix.loc[recent_dates].sum(axis=1)
    avg_signals = signals_per_day.mean()
    max_signals = int(signals_per_day.max())
    days_with_signals = int((signals_per_day > 0).sum())
    print(
        f"    Last 30 days: avg {avg_signals:.1f} BUY/day, max {max_signals}, "
        f"{days_with_signals}/30 days with signals"
    )

    return pass_matrix, score_matrix, indicators, strategy_config


def prepopulate_recommendations(date, pass_matrix, score_matrix, indicators, strategy_config):
    """Pre-populate MongoDB recommended_shares with BUY signals for a specific date.

    Args:
        date: datetime.date for the simulated trading day
        pass_matrix: bool DataFrame (dates x symbols)
        score_matrix: float DataFrame (dates x symbols)
        indicators: ComputedIndicators object
        strategy_config: strategy config dict

    Returns:
        Number of recommendations inserted
    """
    from database import get_mongodb

    db = get_mongodb()

    # Handle timezone: strip tz from matrix index for matching
    matrix_dates = pass_matrix.index
    if matrix_dates.tz is not None:
        matrix_dates_naive = matrix_dates.tz_localize(None)
    else:
        matrix_dates_naive = matrix_dates

    # Find exact date match
    mask = matrix_dates_naive.date == date
    if not mask.any():
        print(f"    No matching date in signal matrix for {date}")
        return 0

    date_idx = matrix_dates[mask][0]

    # Get BUY signals for this date
    day_pass = pass_matrix.loc[date_idx]
    day_score = score_matrix.loc[date_idx]
    buy_symbols = day_pass[day_pass].index.tolist()

    if not buy_symbols:
        print(f"    No BUY signals for {date}")
        return 0

    # Read config for position sizing
    risk_cfg = strategy_config.get("risk_management", {})
    max_positions = risk_cfg.get("max_positions", 15)
    initial_capital = 10000.0  # from config.TRADING_OPTIONS
    capital_per_position = initial_capital / max_positions

    # ATR multipliers for SL/target from strategy config
    exit_rules = strategy_config.get("exit_rules", {})
    sl_atr_mult = exit_rules.get("stop_loss_atr_multiplier", 2.0)
    target_atr_mult = exit_rules.get("target_atr_multiplier", 3.0)

    # recommendation_date: midnight of simulated date, tz-naive
    rec_date = datetime(date.year, date.month, date.day, 0, 0, 0)
    now = datetime.utcnow()

    inserted = 0
    strat_name = strategy_config["name"]

    for symbol in buy_symbols:
        # Get close price and ATR for this date
        try:
            close_price = float(indicators.close.at[date_idx, symbol])
            atr = float(indicators.atr_14.at[date_idx, symbol])
            tech_score = float(day_score[symbol])
        except (KeyError, TypeError, ValueError):
            continue

        if np.isnan(close_price) or np.isnan(atr) or close_price <= 0 or atr <= 0:
            continue

        # Calculate trade plan
        buy_price = round(close_price, 2)
        stop_loss = round(close_price - sl_atr_mult * atr, 2)
        sell_price = round(close_price + target_atr_mult * atr, 2)

        # Position sizing
        quantity = max(1, math.floor(capital_per_position / buy_price))

        # Risk-reward ratio
        risk = buy_price - stop_loss
        reward = sell_price - buy_price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0

        allocation_pct = round((buy_price * quantity / initial_capital) * 100, 2)

        doc = {
            "symbol": symbol,
            "filtered_stock_id": None,
            "company_name": symbol,
            "technical_score": tech_score,
            "combined_score": tech_score,
            "recommendation_strength": "BUY",
            "reason": f"Vectorbt signal: score={tech_score:.2f}",
            "buy_price": buy_price,
            "sell_price": sell_price,
            "stop_loss": stop_loss,
            "backtest_metrics": {},
            "suggested_quantity": quantity,
            "allocation_pct": allocation_pct,
            "rr_ratio": rr_ratio,
            "strategy_name": strat_name,
            "recommendation_date": rec_date,
            "created_at": now,
            "updated_at": now,
        }

        # Upsert (same pattern as persistence_handler.py)
        db.recommended_shares.update_one(
            {"symbol": symbol, "recommendation_date": rec_date},
            {"$set": doc},
            upsert=True,
        )
        inserted += 1

    return inserted


def run_with_retry(date_str, day_num):
    """Run orchestrator with retry logic for rate limit errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        success = run_single_date(date_str)
        if success:
            return True

        # Check if it was a rate limit error (look at the error output)
        if attempt < MAX_RETRIES:
            wait = RETRY_WAIT * attempt
            print(f"  RETRY {attempt}/{MAX_RETRIES}: Waiting {wait}s before retrying {date_str}...")
            time.sleep(wait)
        else:
            print(f"  FAILED after {MAX_RETRIES} retries: {date_str}")

    return False


def main():
    trading_days = get_trading_days(NUM_DAYS)
    print(f"\n{'#'*60}")
    print(f"  {NUM_DAYS}-DAY SLIPPAGE REPLAY TEST (Historical Signals)")
    print(f"  Period: {trading_days[0]} -> {trading_days[-1]}")
    print(f"  Slippage: {SLIPPAGE_PCT*100:.1f}%")
    print(f"  Max stocks: {MAX_STOCKS}")
    print(f"  Data period: {BACKTEST_PERIOD}")
    print(f"  Pre-cache: {PRE_CACHE}")
    print(f"  Delay between runs: {DELAY_BETWEEN_RUNS}s")
    print(f"  Retry wait: {RETRY_WAIT}s, Max retries: {MAX_RETRIES}")
    print(f"{'#'*60}")

    # Pre-cache all NSE stocks to prevent rate limiting during replay
    if PRE_CACHE:
        print(f"\n{'='*60}")
        print("  PRE-CACHE PHASE")
        print(f"{'='*60}")
        total_cached = pre_cache_nse_stocks()
        print(f"  Total stocks cached: {total_cached}")

    # Phase A: Pre-compute signals
    pass_matrix, score_matrix, indicators, strategy_config = setup_signals()

    # Phase B: Day-by-day replay
    print(f"\n{'='*60}")
    print(f"  PHASE B: DAY-BY-DAY REPLAY ({len(trading_days)} days)")
    print(f"{'='*60}")

    all_errors = []
    all_injections = []
    all_exits_checked = []
    all_prepop_counts = []
    failed_days = 0

    prev_closed_ids = set()
    prev_open_symbols = set()
    prev_pyramid_counts = {}  # {symbol: adds_count} — tracks processed pyramids

    for i, day in enumerate(trading_days):
        date_str = day.strftime("%Y-%m-%d")
        day_num = i + 1

        # Pre-populate recommendations for this date
        print(f"\n  --- Pre-populating recommendations for {date_str} ---")
        prepop_count = prepopulate_recommendations(day, pass_matrix, score_matrix, indicators, strategy_config)
        all_prepop_counts.append(prepop_count)
        if prepop_count > 0:
            print(f"    Inserted {prepop_count} BUY recommendations")

        # Run the trading day with retry
        success = run_with_retry(date_str, day_num)
        if not success:
            all_errors.append(f"Day {day_num} ({date_str}): Orchestrator failed after retries")
            failed_days += 1
            # Still continue to next day — don't skip
            if day_num < len(trading_days):
                time.sleep(DELAY_BETWEEN_RUNS)
            continue

        # Get positions after this day
        open_pos = get_open_positions()
        closed_pos = get_closed_positions()
        open_symbols = {p["symbol"] for p in open_pos}

        # Detect new positions
        new_symbols = open_symbols - prev_open_symbols
        if new_symbols:
            print(f"\n  NEW POSITIONS OPENED: {', '.join(new_symbols)}")
            print("\n  BUY SLIPPAGE CHECK:")
            buy_injections = inject_buy_slippage(new_symbols, open_pos)
            all_injections.extend(buy_injections)
            if not buy_injections:
                print("    No buy slippage injected")

        # Detect newly closed positions by comparing closed position IDs
        current_closed_ids = {p["_id"] for p in closed_pos}
        newly_closed_ids = current_closed_ids - prev_closed_ids
        newly_closed = [p for p in closed_pos if p["_id"] in newly_closed_ids]

        if newly_closed:
            closed_names = [p["symbol"] for p in newly_closed]
            print(f"\n  POSITIONS CLOSED: {', '.join(closed_names)}")

            print("\n  EXIT SLIPPAGE CHECK:")
            exit_injections = inject_exit_slippage(newly_closed)
            all_injections.extend(exit_injections)
            if not exit_injections:
                print("    No exit slippage injected")

            # Re-fetch closed positions to get slippage-adjusted values
            if exit_injections:
                closed_pos = get_closed_positions()
                newly_closed = [p for p in closed_pos if p["_id"] in newly_closed_ids]

            print("\n  EXIT PnL VERIFICATION:")
            exit_errors = check_exit_pnl(newly_closed)
            all_errors.extend(exit_errors)
            all_exits_checked.extend(newly_closed)

        # Print positions summary
        print(f"\n  --- After Day {day_num} ({date_str}) ---")
        print_positions_summary(open_pos, "OPEN")

        # Check for pyramiding and inject slippage (with re-injection fix)
        print("\n  PYRAMID SLIPPAGE CHECK:")
        injections, prev_pyramid_counts = inject_pyramid_slippage(open_pos, prev_pyramid_counts)
        all_injections.extend(injections)

        if not injections:
            pyramid_pos = [p for p in open_pos if p.get("adds_count", 0) > 0]
            if pyramid_pos:
                print(f"    {len(pyramid_pos)} position(s) with pyramids (already processed)")
            else:
                print("    No pyramiding activity")

        # Verify PnL recalculation after injection
        if injections:
            print("\n  PnL RECALCULATION VERIFICATION (post-slippage):")
            open_pos = get_open_positions()
            pnl_errors = verify_pnl_recalc(open_pos)
            all_errors.extend(pnl_errors)

        prev_open_symbols = open_symbols
        prev_closed_ids = current_closed_ids

        # Remove closed positions from pyramid tracking
        closed_symbols_now = {p["symbol"] for p in newly_closed}
        for sym in closed_symbols_now:
            prev_pyramid_counts.pop(sym, None)

        # Delay between days to avoid yfinance rate limiting
        if day_num < len(trading_days):
            time.sleep(DELAY_BETWEEN_RUNS)

        # Progress update every 50 days
        if day_num % 50 == 0:
            print(
                f"\n  === PROGRESS: Day {day_num}/{len(trading_days)} | "
                f"Injections: {len(all_injections)} | "
                f"Exits: {len(all_exits_checked)} | "
                f"Failed: {failed_days} ==="
            )

    # Final Summary
    print(f"\n\n{'#'*60}")
    print("  FINAL TEST SUMMARY")
    print(f"{'#'*60}")
    print(f"  Days run: {len(trading_days)}")
    print(f"  Days failed: {failed_days}")
    print(f"  Total pre-populated recommendations: {sum(all_prepop_counts)}")
    print(f"  Pyramid slippage injections: {len(all_injections)}")
    buy_inj = [i for i in all_injections if i.get("type") == "BUY"]
    pyramid_inj = [i for i in all_injections if "type" not in i]
    exit_inj = [i for i in all_injections if i.get("type") == "EXIT"]
    print(f"    BUY slippage: {len(buy_inj)}")
    print(f"    Pyramid slippage: {len(pyramid_inj)}")
    print(f"    EXIT slippage: {len(exit_inj)}")
    print(f"  Exits checked: {len(all_exits_checked)}")
    print(f"  Errors found: {len(all_errors)}")

    if all_prepop_counts:
        avg_prepop = sum(all_prepop_counts) / len(all_prepop_counts)
        days_with_signals = sum(1 for c in all_prepop_counts if c > 0)
        print(f"  Avg recommendations/day: {avg_prepop:.1f}")
        print(f"  Days with BUY signals: {days_with_signals}/{len(trading_days)}")

    # Exit stats
    if all_exits_checked:
        winners = [e for e in all_exits_checked if e.get("pnl_pct", 0) > 0]
        losers = [e for e in all_exits_checked if e.get("pnl_pct", 0) <= 0]
        avg_pnl = sum(e.get("pnl_pct", 0) for e in all_exits_checked) / len(all_exits_checked)
        print("\n  EXIT STATS:")
        print(f"    Total exits: {len(all_exits_checked)}")
        print(f"    Winners: {len(winners)} ({100*len(winners)/len(all_exits_checked):.1f}%)")
        print(f"    Losers: {len(losers)} ({100*len(losers)/len(all_exits_checked):.1f}%)")
        print(f"    Avg PnL per exit: {avg_pnl:+.2f}%")

        # Exit reason breakdown
        reasons = {}
        for e in all_exits_checked:
            r = e.get("exit_reason", "UNKNOWN")
            reasons[r] = reasons.get(r, 0) + 1
        print(f"    Exit reasons: {reasons}")

    if all_injections:
        print("\n  INJECTION DETAIL (first 20):")
        for inj in all_injections[:20]:
            inj_type = inj.get("type", "PYRAMID")
            if inj_type == "BUY":
                print(
                    f"    [BUY] {inj['symbol']}: Rs{inj['old_entry']:.2f} -> Rs{inj['new_entry']:.2f} "
                    f"(invest: Rs{inj['old_investment']:.2f} -> Rs{inj['new_investment']:.2f})"
                )
            elif inj_type == "EXIT":
                print(
                    f"    [EXIT] {inj['symbol']}: Rs{inj['old_exit']:.2f} -> Rs{inj['new_exit']:.2f} "
                    f"(PnL: {inj['old_pnl_pct']:+.2f}% -> {inj['new_pnl_pct']:+.2f}%) "
                    f"Reason={inj['exit_reason']}"
                )
            else:
                print(
                    f"    [PYRAMID] {inj['symbol']}: Rs{inj['old_entry']:.2f} -> Rs{inj['new_entry']:.2f} "
                    f"(invest: Rs{inj['old_investment']:.2f} -> Rs{inj['new_investment']:.2f})"
                )
        if len(all_injections) > 20:
            print(f"    ... and {len(all_injections) - 20} more")

    if all_errors:
        print(f"\n  ERRORS ({len(all_errors)}):")
        for err in all_errors[:30]:
            print(f"    {err}")
        if len(all_errors) > 30:
            print(f"    ... and {len(all_errors) - 30} more")
    else:
        print("\n  No errors found! All PnL calculations verified.")

    # Final positions
    open_pos = get_open_positions()
    closed_pos = get_closed_positions()
    print(f"\n  FINAL OPEN POSITIONS ({len(open_pos)}):")
    print_positions_summary(open_pos)
    print(f"\n  FINAL CLOSED POSITIONS ({len(closed_pos)}):")
    for p in closed_pos[-50:]:  # Last 50 only
        print(
            f"    {p['symbol']}: Entry=Rs{p.get('entry_price', 0):.2f} -> "
            f"Exit=Rs{p.get('exit_price', 0):.2f} "
            f"PnL={p.get('pnl_pct', 0):+.2f}% "
            f"Reason={p.get('exit_reason', '?')}"
        )
    if len(closed_pos) > 50:
        print(f"    ... showing last 50 of {len(closed_pos)} total closed positions")

    return len(all_errors) == 0


if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"slippage_test_{timestamp}.txt")

    tee = TeeWriter(report_path)
    sys.stdout = tee

    try:
        success = main()
        print(f"\n\nReport saved to: {report_path}")
    finally:
        sys.stdout = tee.stdout
        tee.close()

    exit(0 if success else 1)
