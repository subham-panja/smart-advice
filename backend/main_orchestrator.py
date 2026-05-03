import logging
from datetime import datetime, timezone

import config
from database import get_mongodb, get_open_positions
from run_analysis import AutomatedStockAnalysis
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.strategy_loader import StrategyLoader

setup_logging(verbose=True)
logger = logging.getLogger("Orchestrator")


def run_trading_cycle():
    """Main entry point for the unified trading cycle."""
    print("")
    print("=" * 50)
    print("STARTING UNIFIED TRADING CYCLE")
    print("=" * 50)
    print("")
    logger.warning("=== STARTING UNIFIED TRADING CYCLE ===")

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
    strategies = StrategyLoader.load_all_strategies()
    if not strategies:
        raise RuntimeError("No enabled strategies found. Check your JSON configuration.")

    is_paper = trading_opts.get("is_paper_trading", True)

    if is_paper:
        from scripts.execution_engine_paper import ExecutionEngine
        from scripts.portfolio_monitor_paper import PortfolioMonitor
    else:
        from scripts.execution_engine import ExecutionEngine
        from scripts.portfolio_monitor import PortfolioMonitor

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
        analyzer = AutomatedStockAnalysis(verbose=True)
        analyzer.run(strategy_config=strategy)

        # Phase 2.5: Portfolio Backtest (auto-runs per config.py)
        bt_cfg = config.PORTFOLIO_BACKTEST_CONFIG
        if bt_cfg.get("enabled") and bt_cfg.get("auto_run_on_cycle"):
            print("Phase 2.5: Running portfolio backtest for " + strat_name + "...")
            logger.info("Phase 2.5: Running portfolio backtest for " + strat_name + "...")
            try:
                from scripts.run_portfolio_backtest import run_portfolio_backtest

                period = config.DATA_CACHE_CONFIG["periods"].get("portfolio_backtest", "5y")
                max_stocks = bt_cfg.get("auto_run_max_stocks", 1000)

                results = run_portfolio_backtest(
                    strategy_name=strat_name,
                    max_stocks=max_stocks,
                    period=period,
                    save_to_db=True,
                    verbose=False,
                )
                logger.info(
                    "Portfolio backtest for %s: CAGR %.1f%% | Trades: %d | Win Rate: %.1f%%"
                    % (strat_name, results["cagr"], results["total_trades"], results["win_rate"])
                )
                print(
                    "   Portfolio Backtest Complete: CAGR %.1f%% | %d trades"
                    % (results["cagr"], results["total_trades"])
                )
            except Exception as e:
                logger.error("Portfolio backtest failed for %s: %s" % (strat_name, e))
                print("   Portfolio Backtest Error: " + str(e))

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

        slots_left = max_pos - len(open_positions)
        executed_count = 0

        for r in recs:
            if executed_count >= slots_left:
                break

            symbol = r["symbol"]

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
                    logger.info("Successfully executed BUY for %s (%s)" % (symbol, strat_name))

        print("   Executed %d trades for %s" % (executed_count, strat_name))

    print("")
    print("=" * 50)
    print("UNIFIED TRADING CYCLE COMPLETE")
    print("=" * 50)
    print("")


if __name__ == "__main__":
    run_trading_cycle()
