import logging
import requests
from datetime import datetime, timezone

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

    # Final summary
    open_positions = get_open_positions()
    db = get_mongodb()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
    buy_recs = list(
        db.recommended_shares.find(
            {"recommendation_date": {"$gte": today_start}, "recommendation_strength": "BUY"}
        )
    )

    positions_text = ""
    for p in open_positions:
        pnl = p.get("unrealized_pnl", 0)
        pnl_pct = p.get("unrealized_pnl_pct", 0)
        positions_text += f"• {p['symbol']} | Qty: {p['quantity']} | PnL: ₹{pnl:+,.0f} ({pnl_pct:+.1f}%)\n"

    recs_text = ""
    for r in buy_recs:
        recs_text += f"• {r['symbol']} | Score: {r.get('combined_score', 0):.2f} | ₹{r.get('buy_price', 0):.2f}\n"

    if analysis_failed:
        summary = (
            f"⚠️ <b>Trading Cycle — Scan Failed</b>\n\n"
            f"Chartink API was unreachable. Stock scan could not complete.\n"
            f"Try again in a few minutes.\n\n"
            f"📈 <b>Open Positions</b> ({len(open_positions)}):\n{positions_text or 'None\n'}"
            f"💰 <b>Trades Executed</b>: {total_executed}"
        )
    else:
        summary = (
            f"✅ <b>Trading Cycle Complete</b>\n\n"
            f"📈 <b>Open Positions</b> ({len(open_positions)}):\n{positions_text or 'None\n'}"
            f"📋 <b>Today's Recommendations</b> ({len(buy_recs)}):\n{recs_text or 'None\n'}"
            f"💰 <b>Trades Executed</b>: {total_executed}"
        )
    _send_telegram(summary)

    print("")
    print("=" * 50)
    print("UNIFIED TRADING CYCLE COMPLETE")
    print("=" * 50)
    print("")


if __name__ == "__main__":
    run_trading_cycle()
