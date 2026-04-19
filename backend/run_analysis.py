#!/usr/bin/env python3
"""
Automated Stock Analysis Script
File: run_analysis.py

This script automatically analyzes all NSE stocks and saves recommendations to the database.
Designed to be run via cron job every hour.
"""

# Fix OpenMP/threading issues on macOS - MUST be set before importing numpy/scipy/sklearn
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import gc
import sys
import config
import time
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from scripts.analyzer import StockAnalyzer
from utils.logger import setup_logging
from utils.cache_manager import get_cache_manager
from utils.persistence_handler import PersistenceHandler
from utils.stock_scanner import StockScanner
from config import MAX_WORKER_THREADS, BATCH_SIZE, REQUEST_DELAY

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
    
    def analyze_all_stocks(self, max_stocks: int = None, batch_size: int = None, use_all_symbols: bool = False, single_threaded: bool = False):
        """
        Analyze all NSE stocks using multithreading and save recommendations.
        
        Args:
            max_stocks: Maximum number of stocks to analyze (for testing)
            batch_size: Number of stocks to process in each batch (from config if None)
            use_all_symbols: If True, use all NSE symbols instead of filtered ones
            single_threaded: If True, process stocks one by one without threading (for debugging)
        """
        mode_str = "single-threaded" if single_threaded else "multithreading"
        if self.verbose:
            logger.info(f"Starting automated stock analysis with {mode_str} (max_stocks={max_stocks}, use_all_symbols={use_all_symbols})")
        
        # Fetch symbols using the new StockScanner
        if not hasattr(self, '_cached_symbols'):
            self._cached_symbols = StockScanner.get_symbols(max_stocks=max_stocks, use_all_symbols=use_all_symbols)
        
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
            
        processed_count = 0
        recommended_count = 0
        not_recommended_count = 0
        failed_count = 0
        
        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = BATCH_SIZE
        
        # Use full thread pool for better performance
        effective_threads = MAX_WORKER_THREADS
        
        if self.verbose:
            logger.info(f"Using {effective_threads} threads for processing {total_stocks} stocks")
            logger.info(f"Processing {total_stocks} stocks in batches of {batch_size} using {effective_threads} threads")
        
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
                print(f"\rProgress: {progress_percent:.1f}% ({processed_count}/{total_stocks}) - {recommended_count} recommendations", end='', flush=True)
        
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
                    
                    # Get configurable threshold for data purge
                    days_old = self.app.config.get('DATA_PURGE_DAYS', 7)
                    logger.info(f"Data purge threshold: {days_old} days")
                    
                    # Clear old data (recommendations and backtest results) at the start
                    logger.info("SKIPPING database purge operation temporarily for debugging...")
                    # self.clear_old_data(days_old=days_old)
                    logger.info("Database purge operation skipped")
                else:
                    logger.info("Fast mode enabled - skipping cache cleaning and database purge")
                
                # Analyze all stocks
                logger.info("Starting stock analysis...")
                self.analyze_all_stocks(max_stocks=max_stocks, use_all_symbols=use_all_symbols, single_threaded=getattr(self, 'single_threaded', False))
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
        
        # Set single_threaded flag
        analyzer.single_threaded = args.single_threaded
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
            
            def progress_callback(processed, total, recommendations):
                progress_percent = (processed / total) * 100
                # Only update every 2% or when complete to show more frequent updates
                if progress_percent - last_progress_update[0] >= 2 or processed == total:
                    bar_length = 30
                    filled_length = int(bar_length * processed // total)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f"\rProgress: |{bar}| {progress_percent:.1f}% ({processed}/{total}) - {recommendations} recommendations", end='', flush=True)
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
