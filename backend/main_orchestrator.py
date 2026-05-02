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
IS_PAPER = TRADING_OPTIONS["is_paper_trading"]

if IS_PAPER:
    from scripts.execution_engine_paper import ExecutionEngine
    from scripts.portfolio_monitor_paper import PortfolioMonitor
else:
    from scripts.execution_engine import ExecutionEngine
    from scripts.portfolio_monitor import PortfolioMonitor

setup_logging(verbose=True)
logger = logging.getLogger("Orchestrator")


def run_trading_cycle():
    """Main entry point for the unified trading cycle."""
    if config.TRADING_OPTIONS.get("circuit_breaker"):
        msg = "🛑 CIRCUIT BREAKER ACTIVE: Trading cycle stopped from configuration."
        print(f"\n{msg}")
        logger.warning(msg)
        return

    print("\n" + "=" * 50)
    print("🚀 STARTING UNIFIED TRADING CYCLE")
    print("=" * 50 + "\n")
    logger.warning("=== STARTING UNIFIED TRADING CYCLE ===")

    # Pre-Cycle Cleanup
    PersistenceHandler().clear_old_data(config.DATA_PURGE_DAYS)

    # Phase 1: Monitor Existing Portfolio
    print("📍 Phase 1: Monitoring existing positions...")
    logger.info("Phase 1: Monitoring existing positions...")
    PortfolioMonitor().monitor_all_positions()

    # Load All Strategies
    strategies = StrategyLoader.load_all_strategies()
    if not strategies:
        raise RuntimeError("No enabled strategies found. Check your JSON configuration.")

    # Global Portfolio Constraints
    constraints = config.RISK_MANAGEMENT["portfolio_constraints"]
    max_pos = constraints["max_concurrent_positions"]

    for strategy in strategies:
        strat_name = strategy["name"]
        print(f"\n📊 Processing Strategy: {strat_name}")

        # Phase 2: Run Analysis for this strategy
        analyzer = AutomatedStockAnalysis(verbose=True)
        analyzer.run(strategy_config=strategy)

        # Phase 2.5: Portfolio Backtest (auto-runs per config)
        if config.PORTFOLIO_BACKTEST_CONFIG.get("enabled") and config.PORTFOLIO_BACKTEST_CONFIG.get(
            "auto_run_on_cycle"
        ):
            print(f"📈 Phase 2.5: Running portfolio backtest for {strat_name}...")
            logger.info(f"Phase 2.5: Running portfolio backtest for {strat_name}...")
            try:
                from scripts.run_portfolio_backtest import run_portfolio_backtest

                pbt_cfg = config.PORTFOLIO_BACKTEST_CONFIG
                results = run_portfolio_backtest(
                    strategy_name=strat_name,
                    max_stocks=pbt_cfg.get("auto_run_max_stocks", 20),
                    period=config.HISTORICAL_DATA_PERIOD,
                    save_to_db=True,
                    verbose=False,
                )
                logger.info(
                    f"Portfolio backtest for {strat_name}: "
                    f"CAGR {results['cagr']:.1f}% | Final ₹{results['final_portfolio_value']:,.0f} | "
                    f"Trades: {results['total_trades']} | Win Rate: {results['win_rate']:.1f}%"
                )
                print(
                    f"   ✅ Portfolio Backtest Complete: CAGR {results['cagr']:.1f}% | {results['total_trades']} trades"
                )
            except Exception as e:
                logger.error(f"Portfolio backtest failed for {strat_name}: {e}")
                print(f"   ⚠️ Portfolio Backtest Error: {e}")

        # Phase 3: Execute Recommendations for this strategy
        print(f"💰 Phase 3: Executing recommendations for {strat_name}...")
        logger.info(f"Phase 3: Executing recommendations for {strat_name}...")

        engine = ExecutionEngine()
        db = get_mongodb()
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)

        # Fetch current state
        open_positions = get_open_positions()

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

            if TRADING_OPTIONS["auto_execute"] or IS_PAPER:
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
                    logger.info(f"Successfully executed BUY for {symbol} ({strat_name})")

    print("\n" + "=" * 50)
    print("🏁 UNIFIED TRADING CYCLE COMPLETE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_trading_cycle()
