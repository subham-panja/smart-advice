#!/usr/bin/env python3
"""
Automated Stock Analysis Script
File: run_analysis.py

This script automatically analyzes all NSE stocks and saves recommendations to the database.
Designed to be run via cron job every hour.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load config to apply OpenMP/threading limits before importing other libraries
import config

import gc
import time
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from app import create_app
from scripts.analyzer import StockAnalyzer
from scripts.data_fetcher import get_historical_data, get_all_nse_symbols
from utils.logger import setup_logging
from utils.cache_manager import get_cache_manager
from utils.persistence_handler import PersistenceHandler
from utils.stock_scanner import StockScanner
from config import (MAX_WORKER_THREADS, BATCH_SIZE, REQUEST_DELAY,
                    USE_MULTIPROCESSING_PIPELINE, NUM_WORKER_PROCESSES,
                    DATA_FETCH_THREADS, HISTORICAL_DATA_PERIOD)

# Initialize logger variable (will be configured based on verbose flag in AutomatedStockAnalysis)
logger = None

class AutomatedStockAnalysis:
    """Main class for automated stock analysis."""
    
    def __init__(self, verbose=False):
        """Initialize the analyzer."""
        self.app = create_app()
        global logger
        logger = setup_logging(verbose=verbose)
        
        self.analyzer = StockAnalyzer()
        self.persistence = PersistenceHandler(self.app)
        self.start_time = datetime.now()
        self.verbose = verbose
        self.progress_callback = None
        
    def clear_old_data(self, days_old: int = 7):
        """Clear old data older than specified days."""
        self.persistence.clear_old_data(days_old)
    
    def save_recommendation(self, analysis_result: Dict[str, Any]) -> bool:
        """Save analysis result to the database (only BUY recommendations, not HOLD)."""
        return self.persistence.save_recommendation(analysis_result)
    
    def save_backtest_results(self, analysis_result: Dict[str, Any]) -> bool:
        """Save backtest results to the database."""
        return self.persistence.save_backtest_results(analysis_result)
        
    def check_macro_regime(self) -> bool:
        """Check NIFTY 50 (^NSEI) trend to protect capital during bearish regimes. Returns True if safe to trade."""
        try:
            import yfinance as yf
            logger.info("Evaluating Macroeconomic Environmental Gate (NIFTY 50)...")
            
            # Fetch last 3 months of Nifty 50 data
            nifty = yf.Ticker('^NSEI')
            hist = nifty.history(period='3mo')
            
            if len(hist) < 30:
                logger.warning("Insufficient NIFTY 50 data fetched. Defaulting to safe regime.")
                return True
                
            close = hist['Close']
            
            # Calculate 20-day EMA
            ema20 = close.ewm(span=20, adjust=False).mean()
            current_close = close.iloc[-1]
            current_ema20 = ema20.iloc[-1]
            
            # Calculate MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]
            
            is_safe = True
            reasons = []
            
            if current_close < current_ema20:
                is_safe = False
                reasons.append("Price is BELOW 20-Day EMA")
                
            if current_macd < current_signal:
                is_safe = False
                reasons.append("MACD is Negative/Bearish")
                
            if not is_safe:
                logger.warning(f"🚨 MACRO BEAR MARKET DETECTED: {', '.join(reasons)}")
                logger.warning("Activating defensive sub-routine. Halting all new long position analysis.")
                return False
                
            logger.info("NIFTY 50 environmental gate passed. Macro regime is favorable.")
            return True
            
        except Exception as e:
            logger.error(f"Error checking NIFTY 50 regime: {e}")
            return True  # Fail-safe pass if API fails
    
    def analyze_single_stock(self, symbol: str, total_stocks: int, current_index: int) -> Dict[str, Any]:
        """
        Analyze a single stock (thread-safe).
        
        Args:
            symbol: Stock symbol to analyze
            total_stocks: Total number of stocks being processed
            current_index: Current stock index for progress tracking
            
        Returns:
            Dictionary containing analysis result and metadata
        """
        try:
            logger.info(f"Analyzing {symbol} ({current_index}/{total_stocks})")
            
            # Perform analysis with optimized configuration
            try:
                # Create fast mode configuration to skip heavy operations
                fast_config = dict(self.app.config)
                # fast_config['SKIP_SENTIMENT'] = True  # Skip sentiment for speed
                # fast_config['SKIP_FUNDAMENTAL'] = True  # Skip fundamental for speed
                # fast_config['FAST_MODE'] = True  # Enable fast mode
                
                analysis_result = self.analyzer.analyze_stock(symbol, fast_config)
                
                logger.debug(f"Analysis result for {symbol}: {analysis_result}")
            except Exception as e:
                logger.exception(f"Error in analyzing stock {symbol}: {e}")
                raise
            
            # Minimal delay only for very large datasets
            if total_stocks > 500:
                time.sleep(REQUEST_DELAY / 10)  # Very minimal delay for large batches
            
            # Less frequent garbage collection for better performance
            if current_index % 10 == 0:  # Every 10 stocks instead of each
                gc.collect()
            
            return {
                'success': True,
                'symbol': symbol,
                'result': analysis_result,
                'recommended': analysis_result.get('is_recommended', False)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'recommended': False
            }
    
    def analyze_all_stocks(self, max_stocks: int = None, batch_size: int = None, use_all_symbols: bool = False, single_threaded: bool = False, group_name: str = None):
        """
        Analyze all NSE stocks using multithreading and save recommendations.
        
        Args:
            max_stocks: Maximum number of stocks to analyze (for testing)
            batch_size: Number of stocks to process in each batch (from config if None)
            use_all_symbols: If True, use all NSE symbols instead of filtered ones
            single_threaded: If True, process stocks one by one without threading (for debugging)
            group_name: Optional name of symbol group to scan (e.g. 'nifty50')
        """
        mode_str = "single-threaded" if single_threaded else "multithreading"
        log_ctx = f"group={group_name}" if group_name else f"use_all_symbols={use_all_symbols}"
        if self.verbose:
            logger.info(f"Starting automated stock analysis with {mode_str} (max_stocks={max_stocks}, {log_ctx})")
        
        # Fetch symbols using the new StockScanner
        if not hasattr(self, '_cached_symbols') or group_name:
            self._cached_symbols = StockScanner.get_symbols(max_stocks=max_stocks, use_all_symbols=use_all_symbols, group_name=group_name)
        
        filtered_symbols = self._cached_symbols
        if not filtered_symbols:
            logger.error("No NSE symbols found. Exiting analysis.")
            return
            
        symbols_list = list(filtered_symbols.keys())
        logger.info(f"DEBUG: Created symbols list with {len(symbols_list)} symbols")
        symbol_type = "all NSE" if use_all_symbols else "actively traded"
        total_stocks = len(symbols_list)
        
        # Always show stock count regardless of verbose mode
        if self.verbose:
            logger.info(f"Found {total_stocks} {symbol_type} stocks to analyze")
        else:
            print(f"\rAnalyzing {total_stocks} {symbol_type} stocks...", flush=True)
        
        # Branch: Multiprocessing pipeline vs legacy threading
        use_mp = config.USE_MULTIPROCESSING_PIPELINE and not single_threaded
        if use_mp:
            self._run_multiprocessing_pipeline(symbols_list, filtered_symbols, total_stocks)
        else:
            self._run_legacy_threaded(symbols_list, total_stocks, batch_size, single_threaded)
    
    def _fetch_single_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Fetch historical data for one stock (used by Phase 1 threads)."""
        try:
            fresh = self.app.config.get('FRESH_DATA', False)
            period = self.app.config.get('HISTORICAL_DATA_PERIOD', HISTORICAL_DATA_PERIOD)
            data = get_historical_data(symbol, period, fresh=fresh)
            return {'symbol': symbol, 'data': data, 'success': True}
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return {'symbol': symbol, 'data': None, 'success': False}
    
    def _run_multiprocessing_pipeline(self, symbols_list, symbol_names, total_stocks):
        """
        Two-phase pipeline:
          Phase 1: Threaded I/O to fetch all historical data concurrently
          Phase 2: Multiprocessing pool for CPU-heavy TA-Lib / backtest analysis
          Phase 3: Main process saves results to MongoDB
        """
        import multiprocessing
        from utils.parallel_worker import init_worker, analyze_stock_worker
        
        num_processes = config.NUM_WORKER_PROCESSES
        fetch_threads = config.DATA_FETCH_THREADS
        
        logger.info(f"=== MULTIPROCESSING PIPELINE ===")
        logger.info(f"Phase 1: {fetch_threads} threads for data fetching")
        logger.info(f"Phase 2: {num_processes} processes for analysis")
        
        # ---- PHASE 1: Threaded data fetch ----
        phase1_start = datetime.now()
        fetched_data = {}  # symbol -> DataFrame
        fetch_failed = 0
        
        if self.verbose:
            logger.info(f"Phase 1: Fetching data for {total_stocks} stocks...")
        else:
            print(f"\rPhase 1: Fetching data for {total_stocks} stocks...", flush=True)
        
        with ThreadPoolExecutor(max_workers=fetch_threads) as executor:
            future_map = {
                executor.submit(self._fetch_single_stock_data, sym): sym 
                for sym in symbols_list
            }
            for future in as_completed(future_map):
                result = future.result()
                sym = result['symbol']
                if result['success'] and result['data'] is not None and not result['data'].empty:
                    fetched_data[sym] = result['data']
                else:
                    fetch_failed += 1
        
        phase1_time = (datetime.now() - phase1_start).total_seconds()
        logger.info(f"Phase 1 complete: {len(fetched_data)} fetched, {fetch_failed} failed in {phase1_time:.1f}s")
        if not self.verbose:
            print(f"\rPhase 1 done: {len(fetched_data)} stocks fetched in {phase1_time:.1f}s", flush=True)
        
        if not fetched_data:
            logger.error("No data fetched. Aborting analysis.")
            return
        
        # ---- PHASE 2: Multiprocessing analysis ----
        phase2_start = datetime.now()
        
        # Build serializable config (exclude non-picklable Flask objects)
        serializable_config = {
            'ANALYSIS_CONFIG': dict(config.ANALYSIS_CONFIG),
            'HISTORICAL_DATA_PERIOD': config.HISTORICAL_DATA_PERIOD,
            'FRESH_DATA': self.app.config.get('FRESH_DATA', False),
        }
        
        # Prepare work items: convert DataFrames to dicts for pickling
        work_items = []
        for sym, df in fetched_data.items():
            company = symbol_names.get(sym, sym)
            work_items.append((
                sym,
                company,
                df.to_dict(),           # Serializable dict
                df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),  # Index as strings
                serializable_config
            ))
        
        if self.verbose:
            logger.info(f"Phase 2: Processing {len(work_items)} stocks across {num_processes} processes...")
        else:
            print(f"\rPhase 2: Analyzing {len(work_items)} stocks across {num_processes} processes...", flush=True)
        
        # Use 'spawn' to avoid fork-safety issues with TA-Lib/numpy on macOS
        ctx = multiprocessing.get_context('spawn')
        results = []
        
        with ctx.Pool(processes=num_processes, initializer=init_worker) as pool:
            for i, result in enumerate(pool.imap_unordered(analyze_stock_worker, work_items)):
                results.append(result)
                if not self.verbose:
                    pct = ((i + 1) / len(work_items)) * 100
                    sym_name = result.get('symbol', '')
                    bar_len = 30
                    filled = int(bar_len * (i + 1) // len(work_items))
                    bar = '█' * filled + '-' * (bar_len - filled)
                    print(f"\rPhase 2: |{bar}| {pct:.1f}% ({i+1}/{len(work_items)}) | {sym_name}", end='', flush=True)
        
        phase2_time = (datetime.now() - phase2_start).total_seconds()
        if not self.verbose:
            print()  # Newline after progress bar
        logger.info(f"Phase 2 complete: {len(results)} analyzed in {phase2_time:.1f}s")
        
        # ---- PHASE 3: Save results (main process) ----
        phase3_start = datetime.now()
        recommended_count = 0
        failed_count = 0
        
        for r in results:
            if r['success']:
                self.save_backtest_results(r['result'])
                if self.save_recommendation(r['result']):
                    if r['recommended']:
                        recommended_count += 1
                else:
                    failed_count += 1
            else:
                failed_count += 1
        
        phase3_time = (datetime.now() - phase3_start).total_seconds()
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        # Final summary
        logger.info(f"=== PIPELINE COMPLETE ===")
        logger.info(f"Phase 1 (fetch): {phase1_time:.1f}s | Phase 2 (analyze): {phase2_time:.1f}s | Phase 3 (save): {phase3_time:.1f}s")
        logger.info(f"Total: {total_time/60:.1f} min | {len(results)} processed | {recommended_count} recommended | {failed_count} failed")
        logger.info(f"Avg per stock: {total_time/len(results):.1f}s")
        
        try:
            from database import get_mongodb
            db = get_mongodb()
            total_recs = db.recommended_shares.count_documents({})
            logger.info(f"Total recommendations in MongoDB: {total_recs}")
        except Exception as e:
            logger.error(f"Error getting total recommendations count: {e}")
    
    def _run_legacy_threaded(self, symbols_list, total_stocks, batch_size, single_threaded):
        """Original threaded approach (fallback when USE_MULTIPROCESSING_PIPELINE = False)."""
        logger.info(f"Using LEGACY threaded pipeline")
        
        processed_count = 0
        recommended_count = 0
        not_recommended_count = 0
        failed_count = 0
        
        if batch_size is None:
            batch_size = BATCH_SIZE
        
        effective_threads = MAX_WORKER_THREADS
        
        if self.verbose:
            logger.info(f"Using {effective_threads} threads for processing {total_stocks} stocks")
        
        # Process stocks in batches
        for i in range(0, total_stocks, batch_size):
            batch = symbols_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}: stocks {i+1}-{min(i+batch_size, total_stocks)}")
            
            if single_threaded:
                # Single-threaded mode for debugging
                logger.info("Using SINGLE-THREADED mode for debugging")
                for j, symbol in enumerate(batch):
                    try:
                        logger.info(f"Processing {symbol} in single-threaded mode...")
                        result = self.analyze_single_stock(symbol, total_stocks, i + j + 1)
                        logger.debug(f"Received result for {symbol}: {result}")
                        processed_count += 1
                        
                        if result['success']:
                            # Save backtest results regardless of recommendation
                            self.save_backtest_results(result['result'])
                            
                            # Save recommendation only if it's a BUY/STRONG_BUY
                            if self.save_recommendation(result['result']):
                                if result['recommended']:
                                    recommended_count += 1
                                else:
                                    not_recommended_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.exception(f"Error in single-threaded processing for {symbol}: {e}")
                        failed_count += 1
                        processed_count += 1
            else:
                # Multi-threaded mode with timeout handling
                with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                    # Submit all tasks for this batch
                    future_to_symbol = {
                        executor.submit(self.analyze_single_stock, symbol, total_stocks, i + j + 1): symbol
                        for j, symbol in enumerate(batch)
                    }
                    
                    # Process completed tasks with reduced timeout for faster processing
                    for future in as_completed(future_to_symbol, timeout=120):  # 2 minute timeout per stock
                        symbol = future_to_symbol[future]
                        try:
                            result = future.result(timeout=30)  # 30 second timeout to get result
                            logger.debug(f"Received result for {symbol}: {result}")
                            processed_count += 1
                            
                            if result['success']:
                                # Save backtest results regardless of recommendation
                                self.save_backtest_results(result['result'])
                                
                                # Save recommendation only if it's a BUY/STRONG_BUY
                                if self.save_recommendation(result['result']):
                                    if result['recommended']:
                                        recommended_count += 1
                                    else:
                                        not_recommended_count += 1
                                else:
                                    failed_count += 1
                            else:
                                failed_count += 1
                                
                        except TimeoutError:
                            logger.error(f"Timeout processing {symbol} - skipping")
                            failed_count += 1
                            processed_count += 1
                        except Exception as e:
                            logger.exception(f"Error in ThreadPoolExecutor for {symbol}: {e}")
                            failed_count += 1
                            processed_count += 1
            
            # Log progress and trigger optimized garbage collection after each batch
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            avg_time_per_stock = elapsed_time / processed_count if processed_count > 0 else 0
            estimated_remaining = (total_stocks - processed_count) * avg_time_per_stock

            # Optimized garbage collection - only when needed
            if batch_num % 2 == 0:  # Every 2 batches instead of every batch
                gc.collect()

            if self.verbose:
                logger.info(f"Progress: {processed_count}/{total_stocks} stocks processed, "
                           f"{recommended_count} recommendations, {not_recommended_count} not recommended, {failed_count} failed, "
                           f"~{estimated_remaining/60:.1f} minutes remaining")
            elif self.progress_callback:
                # Call progress callback for non-verbose mode
                current_stock_symbol = batch[-1] if batch else ''
                self.progress_callback(processed_count, total_stocks, recommended_count, current_stock_symbol)
            else:
                # Fallback progress display if no callback is set
                progress_percent = (processed_count / total_stocks) * 100
                current_stock_symbol = batch[-1] if batch else ''
                print(f"\rProgress: {progress_percent:.1f}% ({processed_count}/{total_stocks}) | {current_stock_symbol} | {recommended_count} recs", end='', flush=True)
        
        # Final summary
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"Analysis complete!")
        logger.info(f"Total stocks processed: {processed_count}")
        logger.info(f"Recommendations generated: {recommended_count}")
        logger.info(f"Stocks not recommended: {not_recommended_count}")
        logger.info(f"Analysis failures: {failed_count}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average time per stock: {total_time/processed_count:.1f} seconds")
        
        # Log current recommendations count
        try:
            from database import get_mongodb
            db = get_mongodb()
            total_recommendations = db.recommended_shares.count_documents({})
            logger.info(f"Total recommendations in MongoDB: {total_recommendations}")
        except Exception as e:
            logger.error(f"Error getting total recommendations count: {e}")
    
    def run_analysis(self, max_stocks: int = None, use_all_symbols: bool = False, fast_mode: bool = False):
        """
        Run the complete analysis process.
        
        Args:
            max_stocks: Maximum number of stocks to analyze (for testing)
            use_all_symbols: If True, use all NSE symbols instead of filtered ones
            fast_mode: If True, skip database purge and cache cleaning for speed
        """
        with self.app.app_context():
            try:
                logger.info("Starting run_analysis method")
                
                if not fast_mode:
                    # Clean corrupted cache files first
                    logger.info("Cleaning corrupted cache files...")
                    cache_manager = get_cache_manager()
                    cleaned_files = cache_manager.clean_corrupted_cache_files()
                    if cleaned_files > 0:
                        logger.info(f"Cleaned {cleaned_files} corrupted cache files")
                    logger.info("Cache cleaning completed")
                    
                    # Step 2: Purge old data if configured
                    if self.app.config.get('REMOVE_OLD_DATA_ON_EACH_RUN', False):
                        logger.info("Auto-purge enabled: clearing ALL old data before run")
                        self.clear_old_data(days_old=0)
                    else:
                        days_old = self.app.config.get('DATA_PURGE_DAYS', 7)
                        logger.info(f"Auto-purge disabled: cleaning data older than {days_old} days")
                        self.clear_old_data(days_old=days_old)
                else:
                    logger.info("Fast mode enabled - skipping cache cleaning and database purge")
                
                # Analyze all stocks
                logger.info("Starting stock analysis...")
                
                self.analyze_all_stocks(
                    max_stocks=max_stocks, 
                    use_all_symbols=use_all_symbols, 
                    single_threaded=getattr(self, 'single_threaded', False),
                    group_name=getattr(self, 'group_name', None)
                )
                logger.info("Stock analysis completed")
                
                logger.info("Automated analysis completed successfully")
                
            except Exception as e:
                logger.error(f"Error in automated analysis: {e}")
                raise


def main():
    """Main entry point for the script."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description='Automated NSE Stock Analysis')
    parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to analyze (for testing)')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited stocks')
    parser.add_argument('--all', action='store_true', help='Analyze all NSE stocks (not just filtered/actively traded ones)')
    parser.add_argument('--purge-days', type=int, help='Number of days to keep old data (overrides config). Use 0 to remove ALL data.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging with detailed output')
    parser.add_argument('--single-threaded', action='store_true', help='Use single-threaded mode for debugging (slower but more stable)')
    parser.add_argument('--disable-volume-filter', action='store_true', help='Disable volume-based filtering for analysis')
    parser.add_argument('--fresh-data', action='store_true', help='Force fresh data fetch (no cache for today)')
    parser.add_argument('--fast', action='store_true', help='Enable fast mode - skip cache cleaning and database purge for maximum speed')
    parser.add_argument('--group', type=str, help='Analyze aspecific group of stocks from symbol_groups.json (e.g. nifty50)')
    
    args = parser.parse_args()
    
    # Configure logging IMMEDIATELY based on verbose flag
    from utils.logger import setup_logging
    global logger
    logger = setup_logging(verbose=args.verbose)
    
    # Set test mode parameters
    if args.test:
        max_stocks = 2
        if args.verbose:
            logger.info("Running in TEST mode with limited stocks")
    else:
        max_stocks = args.max_stocks
        if args.verbose:
            logger.info("Running in PRODUCTION mode with all stocks")
    
    # Log symbol selection mode
    if args.all:
        if args.verbose:
            logger.info("Using ALL NSE symbols (including inactive/low-volume stocks)")
    else:
        if args.verbose:
            logger.info("Using FILTERED NSE symbols (only actively traded stocks)")
    
    try:
        # Create analyzer with correct verbose setting from the start
        analyzer = AutomatedStockAnalysis(verbose=args.verbose)
        
        # Set flags
        analyzer.single_threaded = args.single_threaded
        analyzer.group_name = args.group
        if args.single_threaded and args.verbose:
            logger.info("Single-threaded mode enabled for debugging")
        
        # Override config if CLI argument provided
        if args.purge_days is not None:
            analyzer.app.config['DATA_PURGE_DAYS'] = args.purge_days
            if args.verbose:
                logger.info(f"Data purge days set to {args.purge_days} (from CLI argument)")
        
        # Override config if disable-volume-filter provided
        if args.disable_volume_filter:
            analyzer.app.config['ENABLE_VOLUME_FILTER'] = False
            config.ENABLE_VOLUME_FILTER = False
            if args.verbose:
                logger.info("Volume filter disabled via CLI argument")
        
        # Set fresh data flag
        if args.fresh_data:
            analyzer.app.config['FRESH_DATA'] = True
            if args.verbose:
                logger.info("Fresh data mode enabled - bypassing cache")
        
        if args.verbose:
            # Verbose mode - logging already configured in constructor
            analyzer.run_analysis(max_stocks=max_stocks, use_all_symbols=args.all, fast_mode=args.fast)
            logger.info("Script completed successfully")
        else:
            # Non-verbose mode - logging already configured in constructor
            
            # Setup progress callback for non-verbose mode
            last_progress_update = [0]  # Use list to allow modification in nested function
            
            def progress_callback(processed, total, recommendations, current_stock=''):
                progress_percent = (processed / total) * 100
                # Only update every 2% or when complete to show more frequent updates
                if progress_percent - last_progress_update[0] >= 2 or processed == total:
                    bar_length = 30
                    filled_length = int(bar_length * processed // total)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f"\rProgress: |{bar}| {progress_percent:.1f}% ({processed}/{total}) | {current_stock} | {recommendations} recs", end='', flush=True)
                    last_progress_update[0] = progress_percent
            
            analyzer.progress_callback = progress_callback
            
            # Show initial message (we'll update this after getting the actual stock count)
            print(f"Initializing analysis...")
            analyzer.run_analysis(max_stocks=max_stocks, use_all_symbols=args.all, fast_mode=args.fast)
            print("\n")
            
            # Show final summary in non-verbose mode
            try:
                from database import get_mongodb
                db = get_mongodb()
                total_recommendations = db.recommended_shares.count_documents({})
                print(f"Analysis completed. Total recommendations in database: {total_recommendations}")
            except Exception:
                print("Analysis completed.")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
