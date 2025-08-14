#!/usr/bin/env python3
"""
Test script to mirror the exact initialization sequence in run_analysis
"""

import os
import sys
import signal
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix OpenMP/threading issues on macOS - MUST be set BEFORE importing any numeric libraries
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_with_timeout(operation_name, operation_func, timeout_seconds=30):
    print(f"Testing {operation_name}...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        start_time = time.time()
        result = operation_func()
        end_time = time.time()
        
        signal.alarm(0)  # Cancel the alarm
        print(f"✓ {operation_name} completed successfully in {end_time - start_time:.2f}s")
        return result
    except TimeoutError:
        print(f"✗ {operation_name} TIMED OUT after {timeout_seconds} seconds")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"✗ {operation_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

print("Starting full initialization sequence test...")

try:
    # Step 1: Import and setup logging (what the main script does first)
    def setup_logging():
        from utils.logger import setup_logging
        return setup_logging(verbose=True)
    
    logger = test_with_timeout("Setup logging", setup_logging)
    if not logger:
        sys.exit(1)
    
    # Step 2: Create Flask app (what AutomatedStockAnalysis.__init__ does first)
    def create_app():
        from app import create_app
        return create_app()
    
    app = test_with_timeout("Create Flask app", create_app)
    if not app:
        sys.exit(1)
    
    # Step 3: Create StockAnalyzer (what AutomatedStockAnalysis.__init__ does)
    def create_stock_analyzer():
        from scripts.analyzer import StockAnalyzer
        return StockAnalyzer()
    
    analyzer = test_with_timeout("Create StockAnalyzer", create_stock_analyzer, 60)  # Longer timeout
    if not analyzer:
        sys.exit(1)
    
    # Step 4: Full AutomatedStockAnalysis initialization
    def create_automated_stock_analysis():
        from run_analysis import AutomatedStockAnalysis
        return AutomatedStockAnalysis(verbose=True)
    
    analysis = test_with_timeout("Create AutomatedStockAnalysis", create_automated_stock_analysis, 60)
    if not analysis:
        sys.exit(1)
    
    # Step 5: Test getting cache manager (what run_analysis does next)
    def get_cache_manager():
        from utils.cache_manager import get_cache_manager
        return get_cache_manager()
    
    cache_manager = test_with_timeout("Get cache manager", get_cache_manager)
    if not cache_manager:
        sys.exit(1)
    
    # Step 6: Test cache cleaning (what run_analysis does next)
    def clean_cache():
        return cache_manager.clean_corrupted_cache_files()
    
    cleaned_files = test_with_timeout("Clean cache files", clean_cache)
    print(f"Cache cleaning result: {cleaned_files}")
    
    # Step 7: Test get_filtered_nse_symbols with offline mode (what analyze_all_stocks does)
    def get_filtered_symbols():
        from scripts.data_fetcher import get_filtered_nse_symbols
        return get_filtered_nse_symbols(2)  # Limit to 2 for testing
    
    symbols = test_with_timeout("Get filtered NSE symbols", get_filtered_symbols, 120)  # Very long timeout
    if symbols:
        print(f"Found {len(symbols)} symbols: {list(symbols.keys())[:5]}")
    
    print("\n✓ All initialization steps completed successfully!")
    print("The hang is likely not in the initialization sequence itself.")
    
except KeyboardInterrupt:
    print("\nTest interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
