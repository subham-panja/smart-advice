import logging
from datetime import datetime, timezone

import config
from config import TRADING_OPTIONS
from database import get_mongodb, get_open_positions
from run_analysis import AutomatedStockAnalysis
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.strategy_loader import StrategyLoader

# Dynamic Engine Selection based on Config
IS_PAPER = TRADING_OPTIONS.get("is_paper_trading", True)

if IS_PAPER:
    from scripts.execution_engine_paper import ExecutionEngine
    from scripts.portfolio_monitor_paper import PortfolioMonitor
else:
    # Fallback for production when implemented
    try:
        from scripts.execution_engine import ExecutionEngine
        from scripts.portfolio_monitor import PortfolioMonitor
    except ImportError:
        ExecutionEngine = None
        PortfolioMonitor = None

setup_logging(verbose=True)
logger = logging.getLogger("Orchestrator")


def run_trading_cycle():
    print("\n" + "=" * 50)
    print("🚀 STARTING UNIFIED TRADING CYCLE")
    print("=" * 50 + "\n")
    logger.warning("=== STARTING UNIFIED TRADING CYCLE ===")

    # Pre-Cycle Cleanup
    PersistenceHandler().clear_old_data(7)

    # Phase 1: Monitor Existing Portfolio
    if PortfolioMonitor:
        print("📍 Phase 1: Monitoring existing positions...")
        logger.info("Phase 1: Monitoring existing positions...")
        PortfolioMonitor().monitor_all_positions()

    # Load All Strategies
    strategies = StrategyLoader.load_all_strategies()
    if not strategies:
        logger.error("No enabled strategies found. Skipping analysis/execution phases.")
        return

    # Global Portfolio Constraints
    constraints = config.RISK_MANAGEMENT.get("portfolio_constraints", {})
    max_pos = constraints.get("max_concurrent_positions", 10)

    for strategy in strategies:
        strat_name = strategy.get("name", "Unknown")
        print(f"\n📊 Processing Strategy: {strat_name}")

        # Phase 2: Run Analysis for this strategy
        analyzer = AutomatedStockAnalysis(verbose=True)
        analyzer.run(strategy_config=strategy)

        # Phase 3: Execute Recommendations for this strategy
        if ExecutionEngine:
            print(f"💰 Phase 3: Executing recommendations for {strat_name}...")
            logger.info(f"Phase 3: Executing recommendations for {strat_name}...")

            engine = ExecutionEngine()
            db = get_mongodb()
            today_start = (
                datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)
            )

            # Fetch current state
            open_positions = get_open_positions()
            open_pos_symbols = {p["symbol"] for p in open_positions}

            if len(open_positions) >= max_pos:
                logger.warning(
                    f"⚠️ Portfolio Full: {len(open_positions)}/{max_pos} positions. Skipping {strat_name} entries."
                )
                continue

            # Fetch today's "BUY" recommendations for THIS strategy
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
                logger.info(f"No new recommendations for {strat_name} today.")
                continue

            slots_left = max_pos - len(open_positions)
            executed_count = 0

            for r in recs:
                if executed_count >= slots_left:
                    break

                symbol = r["symbol"]

                # Check if already in portfolio
                if symbol in open_pos_symbols:
                    # Pyramiding logic is handled inside execute_buy
                    pass

                if TRADING_OPTIONS.get("auto_execute", False) or IS_PAPER:
                    success = engine.execute_buy(
                        symbol,
                        quantity=r.get("suggested_quantity", 1),
                        price=r["buy_price"],
                        stop_loss=r.get("stop_loss", 0),
                        target=r.get("sell_price", 0),
                        recomm_id=r["_id"],
                        strategy_name=strat_name,
                    )
                    if success:
                        executed_count += 1
                        logger.info(f"Successfully executed BUY for {symbol} ({strat_name})")

    print("\n" + "=" * 50)
    print("🏁 UNIFIED TRADING CYCLE COMPLETE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_trading_cycle()
