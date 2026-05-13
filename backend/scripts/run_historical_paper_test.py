#!/usr/bin/env python3
"""
Historical Paper Trading Test
=============================

Runs PortfolioBacktestSession over a past date window to verify
paper trading profitability without affecting live DB state.

Usage:
    cd backend
    python scripts/run_historical_paper_test.py
    python scripts/run_historical_paper_test.py --end-date 2026-01-13 --months 4
    python scripts/run_historical_paper_test.py --months 6 --telegram
    python scripts/run_historical_paper_test.py --months 4 --max-stocks 30 --telegram
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from scripts.run_portfolio_backtest import (
    _prepare_index_data,
    fetch_symbols_data,
)
from utils.logger import setup_logging
from utils.stock_scanner import StockScanner
from utils.strategy_loader import StrategyLoader


def _send_telegram(message: str):
    """Send a message to Telegram. Reuses pattern from main_orchestrator.py."""
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
                print("Telegram message sent (chat_id=%s)" % chat_id)
        except Exception:
            pass


def _print_summary(results: dict):
    """Print console summary table."""
    r = results
    print("")
    print("=" * 60)
    print("HISTORICAL PAPER TRADING TEST RESULTS")
    print("=" * 60)
    print("Date Range:        %s -> %s" % (r["date_range"]["start_date"], r["date_range"]["end_date"]))
    print("Initial Capital:   ₹{:,.0f}".format(r["initial_capital"]))
    print("Final Value:       ₹{:,.0f}".format(r["final_portfolio_value"]))
    print("Total Return:      {:+.2f}%".format(r["total_return_pct"]))
    print("CAGR:              {:.2f}%".format(r["cagr"]))
    print("Max Drawdown:      {:.2f}%".format(r["max_drawdown_pct"]))
    print("Sharpe Ratio:      {:.2f}".format(r["sharpe_ratio"]))
    print("Total Trades:      {}".format(r["total_trades"]))
    print("Win Rate:          {:.1f}%".format(r["win_rate"]))
    print("Profit Factor:     {:.2f}".format(r["profit_factor"]))
    print("Expectancy:        {:.2f}".format(r["expectancy"]))
    print("=" * 60)
    print("")


def _send_telegram_summary(results: dict, strategy_name: str):
    """Send formatted Telegram summary matching orchestrator format."""
    r = results
    initial_cap = r["initial_capital"]
    final_value = r["final_portfolio_value"]
    cash_left = r.get("cash_remaining", 0)
    total_pnl = final_value - initial_cap
    pnl_pct = (total_pnl / initial_cap) * 100

    # Top trades
    sells = [t for t in r.get("trades", []) if t.get("action") == "SELL"]
    win_count = sum(1 for s in sells if s.get("pnl", 0) > 0)
    loss_count = len(sells) - win_count

    trades_text = ""
    # Show last 5 trades
    for t in sells[-5:]:
        pnl = t.get("pnl", 0)
        emoji = "🟢" if pnl >= 0 else "🔴"
        trades_text += "{} {} | PnL: ₹{:+,.0f} ({:+.1f}%)\n".format(
            emoji,
            t.get("symbol", "?"),
            pnl,
            t.get("pnl_pct", 0),
        )

    header = "📝 Historical Paper Trading"
    summary = (
        "{} <b>{} — {} Month Test</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "{}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "💵 Initial Capital: ₹{:,.2f}\n"
        "📈 Final Value: ₹{:,.2f}\n"
        "🏦 Cash Balance: ₹{:,.2f}\n"
        "💰 Net Equity: ₹{:,.2f}\n"
        "📊 Total PnL: ₹{:+,.2f} ({:+.2f}%)\n"
        "📈 CAGR: {:.2f}% | Max DD: {:.2f}%\n"
        "📊 Trades: {} | Win Rate: {:.1f}% ({}W / {}L)\n\n"
        "📋 <b>Recent Trades</b>:\n"
        "{}\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        "📅 Period: {} → {}".format(
            "✅",
            strategy_name,
            args.months,
            header,
            initial_cap,
            final_value,
            cash_left,
            final_value,
            total_pnl,
            pnl_pct,
            r["cagr"],
            r["max_drawdown_pct"],
            len(sells),
            r["win_rate"],
            win_count,
            loss_count,
            trades_text or "None\n",
            r["date_range"]["start_date"],
            r["date_range"]["end_date"],
        )
    )
    _send_telegram(summary)


def run_historical_test(
    strategy_name: str,
    end_date: str,
    months: int,
    max_stocks: int = 50,
    send_telegram: bool = False,
    fresh: bool = False,
):
    setup_logging(verbose=False)

    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    if not strategy:
        print("ERROR: Strategy '%s' not found" % strategy_name)
        sys.exit(1)

    # PortfolioBacktestSession requires market_regime_detection enabled.
    # Force it on for the historical test (does not modify the JSON file).
    strategy.setdefault("analysis_config", {})["market_regime_detection"] = True

    print("Loading stock universe for %s..." % strategy_name)
    symbols = StockScanner().get_symbols(strategy_config=strategy)
    symbols = dict(list(symbols.items())[:max_stocks])
    print("  Selected %d stocks" % len(symbols))

    print("Fetching 10y historical data...")
    symbols_data = fetch_symbols_data(symbols, period="10y", verbose=False)

    if fresh and "^NSEI" in strategy.get("market_regime_config", {}).get("index", ""):
        # Re-fetch index data if regime detection is enabled
        pass

    index_data = _prepare_index_data(strategy, symbols_data, "10y")

    # Build tz-aware timestamps matching the engine's Asia/Kolkata index
    sim_end = pd.Timestamp(end_date, tz="Asia/Kolkata")
    sim_start = sim_end - pd.DateOffset(months=months)

    print("Running simulation: %s -> %s (%d months)" % (sim_start.date(), sim_end.date(), months))
    print("")

    engine = PortfolioBacktestSession(strategy_config=strategy)
    if index_data is not None:
        engine._index_data_override = index_data

    results = engine.run(symbols_data, sim_start_date=sim_start, sim_end_date=sim_end)

    _print_summary(results)

    if send_telegram:
        print("Sending Telegram summary...")
        _send_telegram_summary(results, strategy_name)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Historical Paper Trading Test")
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument("--months", type=int, default=4, help="Lookback months (default: 4)")
    parser.add_argument("--strategy", type=str, default="Swing_Trading", help="Strategy name (default: Swing_Trading)")
    parser.add_argument("--max-stocks", type=int, default=50, help="Max stocks to test (default: 50)")
    parser.add_argument("--telegram", action="store_true", help="Send summary to Telegram")
    parser.add_argument("--fresh", action="store_true", help="Force re-fetch data, bypass cache")
    args = parser.parse_args()

    run_historical_test(
        strategy_name=args.strategy,
        end_date=args.end_date,
        months=args.months,
        max_stocks=args.max_stocks,
        send_telegram=args.telegram,
        fresh=args.fresh,
    )
