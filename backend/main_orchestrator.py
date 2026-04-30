import logging
from datetime import datetime, timezone

import config
from config import TRADING_OPTIONS
from database import get_mongodb, get_open_positions
from run_analysis import AutomatedStockAnalysis
from scripts.execution_engine_paper import ExecutionEngine
from scripts.portfolio_monitor_paper import PortfolioMonitor
from utils.logger import setup_logging

setup_logging(verbose=True)
logger = logging.getLogger("Orchestrator")

IS_PAPER = TRADING_OPTIONS.get("is_paper_trading", True)


def run_trading_cycle():
    print("\n" + "=" * 50)
    print("🚀 STARTING UNIFIED TRADING CYCLE")
    print("=" * 50 + "\n")
    logger.warning("=== STARTING UNIFIED TRADING CYCLE ===")

    # Phase 1: Monitor Existing Portfolio
    if PortfolioMonitor:
        print("📍 Phase 1: Monitoring existing positions...")
        logger.info("Phase 1: Monitoring existing positions...")
        PortfolioMonitor().monitor_all_positions()

    # Phase 2: Run Full Market Analysis (Unified with run_analysis.py)
    print("🔍 Phase 2: Running full market analysis (CACHED)...")
    logger.info("Phase 2: Running full market analysis (CACHED)...")
    analysis_engine = AutomatedStockAnalysis(verbose=config.VERBOSE_LOGGING, fresh=False)
    analysis_engine.run()

    # Phase 3: Execute New Recommendations from DB
    print("💰 Phase 3: Executing new recommendations...")
    logger.info("Phase 3: Executing new recommendations...")
    db = get_mongodb()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Portfolio Constraints Check
    constraints = config.RISK_MANAGEMENT.get("portfolio_constraints", {})
    max_pos = constraints.get("max_concurrent_positions", 5)
    loss_limit_pct = constraints.get("daily_loss_limit", 0.03)

    open_positions = get_open_positions()
    open_pos_symbols = {p["symbol"] for p in open_positions}

    # 1. Max Positions Check
    if len(open_positions) >= max_pos:
        logger.warning(f"⚠️ Portfolio Full: {len(open_positions)}/{max_pos} positions open. Skipping new entries.")
        return

    # 2. Daily Loss Limit Check (Simplified)
    # In a mature system, this would sum realized and unrealized PnL for today
    initial_cap = TRADING_OPTIONS.get("initial_capital", 1000000.0)
    # For now, let's just check unrealized loss of open positions
    total_unrealized_pnl = 0
    for p in open_positions:
        curr = p.get("current_price", p["entry_price"])
        total_unrealized_pnl += (curr - p["entry_price"]) * p["quantity"]

    if total_unrealized_pnl < 0 and abs(total_unrealized_pnl) >= (initial_cap * loss_limit_pct):
        logger.warning(f"🛑 Daily Loss Limit Reached: ₹{abs(total_unrealized_pnl):,.2f} lost. Halting execution.")
        return

    # Get all recommendations generated today
    recs = list(
        db[config.MONGODB_COLLECTIONS.get("recommended_shares", "recommended_shares")].find(
            {"recommendation_date": {"$gte": today_start}}
        )
    )

    if not recs:
        logger.info("No new recommendations found for today.")
        return

    engine = ExecutionEngine() if ExecutionEngine else None

    slots_left = max_pos - len(open_positions)
    executed_count = 0

    for r in recs:
        if executed_count >= slots_left:
            logger.info("Reached max concurrent positions limit during execution loop.")
            break

        symbol = r["symbol"]

        # Check if already in portfolio
        if symbol in open_pos_symbols:
            can_pyramid = config.RISK_MANAGEMENT.get("pyramiding", {}).get("enabled", False)
            if not can_pyramid:
                logger.info(f"⏭️ Skipping {symbol}: Already in portfolio and pyramiding disabled.")
                continue
            else:
                logger.info(f"🔼 {symbol} already in portfolio: Checking for Pyramiding (Add) opportunity...")

        if TRADING_OPTIONS.get("auto_execute", False) or IS_PAPER:
            if engine is None:
                logger.error(f"❌ Cannot execute {symbol}: Live engine not implemented.")
                continue

            # In unified mode, the DB already has the correct sizing/targets
            quantity = r.get("suggested_quantity", 1)
            if quantity <= 0:
                quantity = 1

            logger.warning(f"🚀 Executing BUY for {symbol} | Qty: {quantity}")
            success = engine.execute_buy(
                symbol,
                quantity=quantity,
                price=r["buy_price"],
                stop_loss=r["stop_loss"],
                target=r["sell_price"],
                recomm_id=r["_id"],
            )
            if success:
                executed_count += 1

    logger.warning("=== UNIFIED CYCLE COMPLETE ===")


if __name__ == "__main__":
    run_trading_cycle()
