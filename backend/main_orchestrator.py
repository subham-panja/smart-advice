import logging
from datetime import datetime, timezone

import requests

import config
from database import get_mongodb, get_open_positions
from run_analysis import AutomatedStockAnalysis
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.strategy_loader import StrategyLoader

setup_logging(verbose=True)
logger = logging.getLogger("Orchestrator")


def _send_telegram(message: str):
    """Send a message to Telegram chat. Fails silently if config is missing."""
    tg = getattr(config, "TELEGRAM_CONFIG", {})
    if not tg.get("enabled", False):
        logger.warning("Telegram: not enabled")
        return
    token = tg.get("bot_token", "")
    chat_ids = tg.get("allowed_user_ids", [])
    if not token or not chat_ids:
        logger.warning("Telegram: missing token or chat_ids")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram message sent (chat_id=%s)" % chat_id)
            else:
                logger.error("Telegram API error: %s %s" % (resp.status_code, resp.text))
        except Exception as e:
            logger.error("Telegram send failed: %s" % e)


def run_trading_cycle():
    """Main entry point for the unified trading cycle."""
    print("")
    print("=" * 50)
    print("STARTING UNIFIED TRADING CYCLE")
    print("=" * 50)
    print("")
    logger.warning("=== STARTING UNIFIED TRADING CYCLE ===")

    _send_telegram("🔄 <b>Trading Cycle Started</b>\nRunning analysis and execution...")

    # App-level trading config
    trading_opts = config.TRADING_OPTIONS

    # Circuit breaker check
    if trading_opts.get("circuit_breaker"):
        logger.warning("CIRCUIT BREAKER ACTIVE. Stopping.")
        print("CIRCUIT BREAKER ACTIVE. Stopping.")
        return

    # Pre-Cycle Cleanup
    PersistenceHandler().clear_old_data(config.DATA_PURGE_DAYS)

    # Load All Strategies
    all_strategies = StrategyLoader.load_all_strategies()
    if not all_strategies:
        raise RuntimeError("No enabled strategies found. Check your JSON configuration.")

    logger.info("Running trading cycle for %d enabled strategy(ies)" % len(all_strategies))

    strategies = all_strategies

    is_paper = trading_opts.get("is_paper_trading", True)

    if is_paper:
        from scripts.execution_engine_paper import ExecutionEngine
        from scripts.portfolio_monitor_paper import PortfolioMonitor
    else:
        from scripts.execution_engine import ExecutionEngine
        from scripts.portfolio_monitor import PortfolioMonitor

    total_executed = 0
    analysis_failed = False

    for strategy in strategies:
        strat_name = strategy["name"]

        print("")
        print("Processing Strategy: " + strat_name)
        logger.info("Processing Strategy: " + strat_name)

        # Phase 1: Monitor Existing Portfolio
        print("Phase 1: Monitoring existing positions...")
        logger.info("Phase 1: Monitoring existing positions...")
        PortfolioMonitor().monitor_all_positions()

        # Phase 2: Run Analysis for this strategy
        print("Phase 2: Running analysis for " + strat_name + "...")
        logger.info("Phase 2: Running analysis for " + strat_name + "...")
        try:
            analyzer = AutomatedStockAnalysis(verbose=True)
            analyzer.run(strategy_config=strategy)
            # Check if the scanner produced any candidates
            if not analyzer.scanned_symbols_count or analyzer.scanned_symbols_count == 0:
                logger.warning("Analysis scan produced 0 candidates for %s" % strat_name)
                analysis_failed = True
        except Exception as e:
            logger.error("Analysis failed for %s: %s" % (strat_name, e))
            analysis_failed = True

        # Phase 3: Execute Recommendations for this strategy
        print("Phase 3: Executing recommendations for " + strat_name + "...")
        logger.info("Phase 3: Executing recommendations for " + strat_name + "...")

        engine = ExecutionEngine(strategy_config=strategy)
        db = get_mongodb()
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)

        open_positions = get_open_positions()

        # Max positions from strategy risk_management
        risk_cfg = strategy.get("risk_management", {})
        max_pos = risk_cfg.get("max_positions", 15)

        if len(open_positions) >= max_pos:
            logger.warning("Portfolio Full: %d/%d positions. Skipping %s" % (len(open_positions), max_pos, strat_name))
            print("   Portfolio Full: %d/%d positions" % (len(open_positions), max_pos))
            continue

        recs = list(
            db.recommended_shares.find(
                {
                    "recommendation_date": {"$gte": today_start},
                    "strategy_name": strat_name,
                    "recommendation_strength": "BUY",
                }
            )
        )

        if not recs:
            logger.info("No new recommendations for " + strat_name + " today.")
            print("   No new recommendations for " + strat_name)
            continue

        # Market Breadth Filter — block new buys during broad market sell-offs
        if not engine._check_market_breadth_paper(strategy):
            logger.warning("MARKET BREADTH WEAK: Skipping new buys for %s" % strat_name)
            print("   Market Breadth Weak: Skipping new buys for " + strat_name)
            continue

        slots_left = max_pos - len(open_positions)
        executed_count = 0

        # Calculate remaining capital to enforce hard cap
        total_invested = sum(p.get("total_investment", 0) for p in open_positions)
        initial_capital = trading_opts.get("initial_capital", 100000.0)
        remaining_capital = initial_capital - total_invested
        logger.info(
            "Capital Check: ₹{:.2f} invested / ₹{:.2f} total | ₹{:.2f} remaining".format(
                total_invested, initial_capital, remaining_capital
            )
        )

        for r in recs:
            if executed_count >= slots_left:
                break

            symbol = r["symbol"]
            trade_cost = r["buy_price"] * r["suggested_quantity"]

            # Hard cap check: skip if trade would exceed remaining capital
            if trade_cost > remaining_capital:
                logger.warning(
                    "Insufficient capital for {}: need ₹{:.2f}, have ₹{:.2f} remaining".format(
                        symbol, trade_cost, remaining_capital
                    )
                )
                continue

            if trading_opts.get("auto_execute", True) or is_paper:
                success = engine.execute_buy(
                    symbol,
                    quantity=r["suggested_quantity"],
                    price=r["buy_price"],
                    stop_loss=r["stop_loss"],
                    target=r["sell_price"],
                    recomm_id=r["_id"],
                    strategy_name=strat_name,
                )
                if success:
                    executed_count += 1
                    remaining_capital -= trade_cost
                    logger.info(
                        "Successfully executed BUY for {} ({}) | Remaining capital: ₹{:.2f}".format(
                            symbol, strat_name, remaining_capital
                        )
                    )

        print("   Executed %d trades for %s" % (executed_count, strat_name))
        total_executed += executed_count

    # Final summary — match telegram_bot.py format exactly
    open_positions = get_open_positions()
    db = get_mongodb()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    buy_recs = list(
        db.recommended_shares.find({"recommendation_date": {"$gte": today_start}, "recommendation_strength": "BUY"})
    )
    initial_cap = trading_opts.get("initial_capital", 100000.0)
    is_paper = trading_opts.get("is_paper_trading", True)

    if analysis_failed:
        summary = (
            "⚠️ <b>Trading Cycle — Scan Failed</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "Chartink API was unreachable. Stock scan could not complete.\n"
            "Try again in a few minutes."
        )
        _send_telegram(summary)
    else:
        # Send recommendations (one message per rec, matching view_recs format)
        for r in buy_recs:
            bt = r.get("backtest_metrics", {})
            score = r.get("combined_score", 0)
            quantity = r.get("suggested_quantity", 1)
            buy_price = r.get("buy_price", 0)
            sell_price = r.get("sell_price", 0)
            stop_loss = r.get("stop_loss", 0)
            total_cost = quantity * buy_price
            cap_pct = r.get("allocation_pct", (total_cost / initial_cap) * 100)
            rr = r.get("rr_ratio", 0)
            rec_msg = (
                f"📈 <b>{r['symbol']}</b> | <b>{r.get('strategy_name', 'Swing_Trading')}</b>\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"🎯 Score: <b>{score:.1f}/100</b>\n"
                f"💰 <b>Trade Plan</b>:\n"
                f"• Entry: ₹{buy_price:.2f}\n"
                f"• Target: ₹{sell_price:.2f}\n"
                f"• Stop Loss: ₹{stop_loss:.2f}\n"
                f"• RR Ratio: <b>{rr:.2f}</b>\n\n"
                f"🔢 <b>Sizing (₹{initial_cap/100000:.1f}L Cap)</b>:\n"
                f"• Quantity: <b>{quantity}</b>\n"
                f"• Allocation: <b>₹{total_cost:,.2f} ({cap_pct:.1f}%)</b>\n\n"
                f"📊 <b>Backtest Stats</b>:\n"
                f"• Trades: <b>{bt.get('total_trades', 0)}</b>\n"
                f"• Win Rate: {bt.get('avg_win_rate', 0):.1f}%\n"
                f"• Avg CAGR: {bt.get('avg_cagr', 0):.1f}%\n"
                f"• Expectancy: {bt.get('avg_expectancy', 0.0):.2f}\n\n"
                f"📝 <b>Analysis</b>: {r.get('reason', 'Technical Momentum Breakout')}"
            )
            _send_telegram(rec_msg)

        # Send positions summary (matching view_positions format)
        total_mkt_val = 0
        total_pnl_val = 0
        positions_msg = ""
        for p in open_positions:
            current_p = p.get("current_price", p["entry_price"])
            pnl_val = (current_p - p["entry_price"]) * p["quantity"]
            pnl_pct = ((current_p - p["entry_price"]) / p["entry_price"]) * 100
            total_cost = p.get("total_investment", p["quantity"] * p["entry_price"])
            total_mkt_val += current_p * p["quantity"]
            total_pnl_val += pnl_val
            status_emoji = "🟢" if pnl_pct >= 0 else "🔴"
            allocation = p.get("allocation_pct", (total_cost / initial_cap) * 100)
            positions_msg += (
                f"{status_emoji} <b>{p['symbol']}</b> | {p.get('strategy_name', 'Swing_Trading')}\n"
                f"📅 Entered: {p['entry_date'].strftime('%Y-%m-%d %H:%M') if p.get('entry_date') else 'N/A'}\n"
                f"🔢 Qty: {p['quantity']} @ ₹{p['entry_price']:.2f}\n"
                f"💰 Cost: ₹{total_cost:,.2f} ({allocation:.1f}% Cap)\n"
                f"🎯 Target: ₹{p['target']:.2f} | 🛑 SL: ₹{p['stop_loss']:.2f}\n"
                f"💸 PnL: ₹{pnl_val:+,.2f} ({pnl_pct:+.2f}%)\n\n"
            )

        cash_left = initial_cap - sum(
            p.get("total_investment", p["quantity"] * p["entry_price"]) for p in open_positions
        )
        total_equity = total_mkt_val + cash_left
        overall_pnl_pct = ((total_equity - initial_cap) / initial_cap) * 100
        header = "📝 Active Paper Positions" if is_paper else "💼 Active Live Positions"

        summary = (
            f"✅ <b>Trading Cycle Complete!</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"{header} ({len(open_positions)}):\n"
            f"{'━━━━━━━━━━━━━━━━━━━━\n' + positions_msg if positions_msg else '📭 No open positions found.\n\n'}"
            f"📊 <b>PORTFOLIO SUMMARY</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💵 Initial Capital: ₹{initial_cap:,.2f}\n"
            f"📈 Market Value: ₹{total_mkt_val:,.2f}\n"
            f"🏦 Cash Balance: ₹{cash_left:,.2f}\n"
            f"💰 Net Equity: ₹{total_equity:,.2f}\n"
            f"📊 Total PnL: ₹{total_pnl_val:+,.2f} ({overall_pnl_pct:+.2f}%)\n\n"
            f"💰 Trades Executed: {total_executed}"
        )
        _send_telegram(summary)

    print("")
    print("=" * 50)
    print("UNIFIED TRADING CYCLE COMPLETE")
    print("=" * 50)
    print("")


if __name__ == "__main__":
    run_trading_cycle()
