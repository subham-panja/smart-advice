import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ANALYSIS_CONFIG, TRADING_OPTIONS
from database import get_open_positions
from scripts.analyzer import StockAnalyzer
from scripts.data_fetcher import get_all_nse_symbols
from scripts.execution_engine import ExecutionEngine
from scripts.portfolio_monitor import PortfolioMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")


def analyze_and_execute(symbol, analyzer, engine, open_positions):
    """Analyzes a stock and executes a trade if recommended and not already held."""
    if symbol in open_positions:
        return f"Skipped {symbol}: Held"

    res = analyzer.analyze_stock(symbol, {"ANALYSIS_CONFIG": ANALYSIS_CONFIG, "FRESH_DATA": True})
    if res.get("is_recommended"):
        logger.info(f"🔥 RECOMMENDATION: {symbol}")
        if TRADING_OPTIONS.get("auto_execute", False) or TRADING_OPTIONS.get("is_paper_trading", True):
            tp = res.get("trade_plan", {})
            rm = res.get("risk_management", {})
            quantity = rm.get("position_size", 1)

            engine.execute_buy(
                symbol,
                quantity=quantity,
                price=tp.get("buy_price", res.get("buy_price")),
                stop_loss=tp.get("stop_loss", rm.get("stop_loss")),
                target=tp.get("sell_price", tp.get("target", rm.get("targets", {}).get("T1"))),
            )
            return f"Executed {symbol}"
    return f"Done {symbol}"


def run_trading_cycle():
    logger.info("=== STARTING TRADING CYCLE ===")
    PortfolioMonitor().monitor_all_positions()

    analyzer, engine = StockAnalyzer(), ExecutionEngine()
    symbols = list(get_all_nse_symbols().keys())[:50]
    open_pos = {p["symbol"] for p in get_open_positions()}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(analyze_and_execute, s, analyzer, engine, open_pos) for s in symbols]
        for f in as_completed(futures):
            logger.debug(f.result())

    logger.info("=== CYCLE COMPLETE ===")


if __name__ == "__main__":
    run_trading_cycle()
