#!/usr/bin/env python3
"""Diagnostic script to find why portfolio backtest CAGR is so low."""

import logging
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.data_fetcher import get_historical_data
from scripts.portfolio_backtest_engine import PortfolioBacktestSession
from utils.stock_scanner import StockScanner
from utils.strategy_loader import StrategyLoader

logging.basicConfig(level=logging.WARNING)


def run_diagnosis(strategy_name: str, max_stocks: int = 20):
    strategy = StrategyLoader.get_strategy_by_name(strategy_name)
    symbols = StockScanner.get_symbols(strategy_config=strategy)
    symbols = dict(list(symbols.items())[:max_stocks])

    print(f"Fetching data for {len(symbols)} symbols...")
    symbols_data = {}
    for sym, name in symbols.items():
        df = get_historical_data(sym, period="5y")
        if df is not None and len(df) > 100:
            symbols_data[sym] = df

    print("Running backtest...")
    engine = PortfolioBacktestSession(strategy_config=strategy)
    results = engine.run(symbols_data)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC REPORT")
    print("=" * 60)

    # 1. Overall stats
    print("\n1. OVERALL")
    print(f"   Final Value: ₹{results['final_portfolio_value']:,.0f}")
    print(f"   CAGR: {results['cagr']:.2f}%")
    print(f"   Total SELL trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1f}%")

    # 2. Trade type breakdown
    trades = results["trades"]
    sells = [t for t in trades if t.trade_type == "SELL"]
    partials = [t for t in trades if t.trade_type == "PARTIAL_SELL"]
    pyramids = [t for t in trades if t.trade_type == "PYRAMID_ADD"]
    buys = [t for t in trades if t.trade_type == "BUY"]

    print("\n2. TRADE COUNT BREAKDOWN")
    print(f"   BUYs: {len(buys)}")
    print(f"   PYRAMID_ADDs: {len(pyramids)}")
    print(f"   PARTIAL_SELLs: {len(partials)}")
    print(f"   SELLs (full exits): {len(sells)}")

    # 3. Exit reason analysis
    reasons = Counter(t.exit_reason for t in sells)
    print("\n3. EXIT REASONS (full exits only)")
    for reason, count in reasons.most_common():
        rt = [t for t in sells if t.exit_reason == reason]
        wins = [t for t in rt if t.pnl > 0]
        total_pnl = sum(t.pnl for t in rt)
        avg_pnl = total_pnl / len(rt) if rt else 0
        win_rate = len(wins) / len(rt) * 100 if rt else 0
        print(
            f"   {reason:20s}: {count:3d} trades | WinRate: {win_rate:5.1f}% | Avg PnL: ₹{avg_pnl:+8,.0f} | Total: ₹{total_pnl:+10,.0f}"
        )

    # 4. Winner vs Loser analysis
    winners = [t for t in sells if t.pnl > 0]
    losers = [t for t in sells if t.pnl <= 0]

    print("\n4. WINNER/LOSER STATS")
    if winners:
        avg_win = sum(t.pnl for t in winners) / len(winners)
        avg_win_pct = sum(t.pnl_pct for t in winners) / len(winners)
        print(f"   Winners: {len(winners)}")
        print(f"   Avg Win ₹: {avg_win:,.0f} | Avg Win %: {avg_win_pct:.2f}%")
        print(f"   Median Win ₹: {sorted(t.pnl for t in winners)[len(winners)//2]:,.0f}")
    if losers:
        avg_loss = sum(t.pnl for t in losers) / len(losers)
        avg_loss_pct = sum(t.pnl_pct for t in losers) / len(losers)
        print(f"   Losers: {len(losers)}")
        print(f"   Avg Loss ₹: {avg_loss:,.0f} | Avg Loss %: {avg_loss_pct:.2f}%")
        print(f"   Median Loss ₹: {sorted(t.pnl for t in losers)[len(losers)//2]:,.0f}")

    # 5. Realized R:R
    print("\n5. REALIZED R:R (based on SL distance vs exit distance)")
    r_values = []
    for t in sells:
        risk = t.entry_price - t.stop_loss
        if risk <= 0:
            continue
        reward = t.exit_price - t.entry_price if t.exit_price else 0
        r_values.append(reward / risk)

    if r_values:
        wins_r = [r for r in r_values if r > 0]
        losses_r = [r for r in r_values if r <= 0]
        if wins_r:
            print(f"   Avg R on winners: {sum(wins_r)/len(wins_r):.2f}")
            print(f"   Median R on winners: {sorted(wins_r)[len(wins_r)//2]:.2f}")
        if losses_r:
            print(f"   Avg R on losers: {sum(losses_r)/len(losses_r):.2f}")
            print(f"   Median R on losers: {sorted(losses_r)[len(losses_r)//2]:.2f}")

    # 6. Position sizing analysis
    print("\n6. POSITION SIZING")
    if buys:
        avg_alloc = sum(t.allocation_pct for t in buys) / len(buys)
        max_alloc = max(t.allocation_pct for t in buys)
        min_alloc = min(t.allocation_pct for t in buys)
        print(f"   Avg allocation: {avg_alloc:.1f}% of portfolio")
        print(f"   Max allocation: {max_alloc:.1f}%")
        print(f"   Min allocation: {min_alloc:.1f}%")

    # 7. Capital utilization over time
    snapshots = results.get("daily_snapshots", [])
    if snapshots:
        avg_invested = sum(s["market_value"] / s["portfolio_value"] * 100 for s in snapshots) / len(snapshots)
        print("\n7. CAPITAL UTILIZATION")
        print(f"   Avg % invested: {avg_invested:.1f}%")
        print(f"   Avg cash %: {100 - avg_invested:.1f}%")

    # 8. Theoretical vs Actual
    total_pnl = sum(t.pnl for t in sells)
    n = len(sells)
    w = len(winners)
    num_losers = len(losers)

    avg_win_r = sum(t.pnl for t in winners) / len(winners) / 10000 if winners else 0
    avg_loss_r = abs(sum(t.pnl for t in losers) / len(losers) / 10000) if losers else 1

    effective_expectancy = (w / n * avg_win_r) - (num_losers / n * avg_loss_r)
    theoretical_expectancy = 0.453 * 2.5 - 0.547 * 1.0

    print("\n8. EXPECTANCY ANALYSIS")
    print(f"   Theoretical expectancy: {theoretical_expectancy:.3f}R per trade")
    print(f"   Effective expectancy:   {effective_expectancy:.3f}R per trade")
    print(
        f"   Edge leakage:           {(theoretical_expectancy - effective_expectancy):.3f}R ({(theoretical_expectancy - effective_expectancy)/theoretical_expectancy*100:.0f}%)"
    )
    print(f"\n   If we had {theoretical_expectancy:.3f}R expectancy:")
    print(f"     Expected final value: ₹{1000000 * (1 + theoretical_expectancy * 0.01) ** n:,.0f}")
    print(
        f"     Expected CAGR: {((1000000 * (1 + theoretical_expectancy * 0.01) ** n / 1000000) ** (1/5) - 1) * 100:.1f}%"
    )


if __name__ == "__main__":
    run_diagnosis("Delayed_EP", max_stocks=20)
