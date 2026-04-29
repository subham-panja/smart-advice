import logging
from datetime import datetime, timezone

from config import TRADING_OPTIONS
from database import get_mongodb, get_open_positions
from run_analysis import AutomatedStockAnalysis

# Dynamic Engine Loading
IS_PAPER = TRADING_OPTIONS.get("is_paper_trading", True)
if IS_PAPER:
    from scripts.execution_engine_paper import ExecutionEngine
    from scripts.portfolio_monitor_paper import PortfolioMonitor
else:
    ExecutionEngine = None
    PortfolioMonitor = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")


def run_trading_cycle():
    logger.info("=== STARTING UNIFIED TRADING CYCLE ===")

    # Phase 1: Monitor Existing Portfolio
    if PortfolioMonitor:
        logger.info("Phase 1: Monitoring existing positions...")
        PortfolioMonitor().monitor_all_positions()

    # Phase 2: Run Full Market Analysis (Unified with run_analysis.py)
    logger.info("Phase 2: Running full market analysis...")
    analysis_engine = AutomatedStockAnalysis(verbose=False)
    analysis_engine.run()

    # Phase 3: Execute New Recommendations from DB
    logger.info("Phase 3: Executing new recommendations...")
    db = get_mongodb()
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Get all recommendations generated today
    recs = list(db.recommended_shares.find({"recommendation_date": {"$gte": today_start}}))

    if not recs:
        logger.info("No new recommendations found for today.")
        return

    engine = ExecutionEngine() if ExecutionEngine else None
    open_pos_symbols = {p["symbol"] for p in get_open_positions()}

    for r in recs:
        symbol = r["symbol"]

        if symbol in open_pos_symbols:
            logger.info(f"⏭️ Skipping {symbol}: Already in portfolio.")
            continue

        if TRADING_OPTIONS.get("auto_execute", False) or IS_PAPER:
            if engine is None:
                logger.error(f"❌ Cannot execute {symbol}: Live engine not implemented.")
                continue

            # In unified mode, the DB already has the correct sizing/targets
            quantity = r.get("suggested_quantity", 1)
            if quantity <= 0:
                quantity = 1

            logger.info(f"🚀 Executing BUY for {symbol} | Qty: {quantity}")
            engine.execute_buy(
                symbol,
                quantity=quantity,
                price=r["buy_price"],
                stop_loss=r["stop_loss"],
                target=r["sell_price"],
                recomm_id=r["_id"],
            )

    logger.info("=== UNIFIED CYCLE COMPLETE ===")


if __name__ == "__main__":
    run_trading_cycle()
