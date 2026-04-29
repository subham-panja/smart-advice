"""
Main Orchestrator
File: main_orchestrator.py

The master script that:
1. Runs the Portfolio Monitor to update existing positions.
2. Runs the Stock Analyzer to find new opportunities.
3. Automatically executes buys (if enabled) or flags them, preventing duplicates.
"""

import logging
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from scripts.analyzer import StockAnalyzer
from scripts.portfolio_monitor import PortfolioMonitor
from scripts.execution_engine import ExecutionEngine
from scripts.data_fetcher import get_all_nse_symbols
from database import get_open_positions, get_mongodb
from config import TRADING_OPTIONS, ANALYSIS_CONFIG

from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orchestrator")

def analyze_and_execute(symbol, analyzer, engine, app_config, open_positions):
    """Worker function for parallel analysis and execution."""
    try:
        # DUPLICATE PREVENTION: Skip if we already own this stock
        if symbol in open_positions:
            return f"Skipped {symbol}: Position open"
            
        result = analyzer.analyze_stock(symbol, app_config)
        
        if result.get('is_recommended'):
            logger.info(f"🔥 NEW RECOMMENDATION: {symbol}")
            
            # 3. AUTO-EXECUTION (if enabled)
            if TRADING_OPTIONS.get('auto_execute', False) or TRADING_OPTIONS.get('is_paper_trading', True):
                rm = result.get('risk_management', {})
                pos_size = rm.get('position_sizing', {}).get('position_size', 0)
                
                if pos_size > 0:
                    engine.execute_buy(
                        symbol=symbol,
                        quantity=pos_size,
                        price=result['trade_plan']['buy_price'],
                        stop_loss=result['trade_plan']['stop_loss'],
                        target=result['trade_plan']['sell_price'],
                        recomm_id=result.get('_id')
                    )
                    return f"Executed {symbol}"
                else:
                    return f"Recommended {symbol} but invalid size"
        return f"Analyzed {symbol}"
    except Exception as e:
        return f"Error {symbol}: {e}"

def run_trading_cycle():
    logger.info("=== STARTING TRADING CYCLE ===")
    
    # 1. RUN PORTFOLIO MONITOR (Manage existing trades)
    logger.info("Phase 1: Portfolio Monitoring")
    monitor = PortfolioMonitor()
    monitor.monitor_all_positions()
    
    # 2. RUN STOCK ANALYZER (Find new trades)
    logger.info("Phase 2: Scanning for New Opportunities")
    analyzer = StockAnalyzer()
    engine = ExecutionEngine()
    
    # Get all symbols (scanning top 50 for speed)
    symbols = list(get_all_nse_symbols().keys())[:50] 
    open_positions = {p['symbol'] for p in get_open_positions()}
    
    app_config = {
        'ANALYSIS_CONFIG': ANALYSIS_CONFIG,
        'FRESH_DATA': True
    }
    
    recommendations_found = 0
    # Use ThreadPoolExecutor for parallel scanning
    max_workers = 10 # 10 threads for I/O and CPU mix
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_and_execute, sym, analyzer, engine, app_config, open_positions): sym for sym in symbols}
        
        for future in as_completed(futures):
            res = future.result()
            if "NEW RECOMMENDATION" in res or "Executed" in res:
                recommendations_found += 1
            logger.debug(res)

    logger.info(f"=== CYCLE COMPLETE: Recommendations processed. ===")

if __name__ == "__main__":
    run_trading_cycle()
