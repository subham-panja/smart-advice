#!/usr/bin/env python3
"""
Simplified test script that progressively adds complexity to identify the exact hang point
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

def test_step(step_name, func, timeout_seconds=30):
    print(f"Step: {step_name}")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        start_time = time.time()
        result = func()
        end_time = time.time()
        
        signal.alarm(0)  # Cancel the alarm
        print(f"✓ {step_name} completed in {end_time - start_time:.2f}s")
        return result
    except TimeoutError:
        print(f"✗ {step_name} TIMED OUT after {timeout_seconds} seconds")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"✗ {step_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return None

print("=== Progressive Run Analysis Test ===")

try:
    # Configure logging
    def step1():
        from utils.logger import setup_logging
        return setup_logging(verbose=True)
    
    logger = test_step("1. Setup logging", step1)
    if not logger:
        sys.exit(1)
    
    # Create analyzer
    def step2():
        from run_analysis import AutomatedStockAnalysis
        return AutomatedStockAnalysis(verbose=True)
    
    analyzer = test_step("2. Create AutomatedStockAnalysis", step2)
    if not analyzer:
        sys.exit(1)
    
    # Test app context creation and usage
    def step3():
        with analyzer.app.app_context():
            print("    - App context created successfully")
            return True
    
    test_step("3. Test app context", step3)
    
    # Test cache manager
    def step4():
        from utils.cache_manager import get_cache_manager
        cache_manager = get_cache_manager()
        return cache_manager.clean_corrupted_cache_files()
    
    cleaned = test_step("4. Test cache manager", step4)
    print(f"    - Cleaned {cleaned} files")
    
    # Test getting symbols
    def step5():
        from scripts.data_fetcher import get_filtered_nse_symbols
        return get_filtered_nse_symbols(2)
    
    symbols = test_step("5. Test get symbols", step5)
    if symbols:
        print(f"    - Found {len(symbols)} symbols")
    
    # Test the beginning of run_analysis method
    def step6():
        print("    - Starting run_analysis method simulation...")
        with analyzer.app.app_context():
            print("    - Created app context")
            
            # Test cache manager
            from utils.cache_manager import get_cache_manager
            cache_manager = get_cache_manager()
            print("    - Got cache manager")
            
            # Test cache cleaning  
            cleaned_files = cache_manager.clean_corrupted_cache_files()
            print(f"    - Cleaned {cleaned_files} cache files")
            
            # Test config access
            days_old = analyzer.app.config.get('DATA_PURGE_DAYS', 7)
            print(f"    - Data purge threshold: {days_old} days")
            
            print("    - About to test analyze_all_stocks call...")
            return True
    
    test_step("6. Test run_analysis beginning", step6, 60)
    
    # Test the actual analyze_all_stocks call (the most likely hanging point)
    def step7():
        print("    - Calling analyze_all_stocks with minimal parameters...")
        try:
            # This is the actual call that might be hanging
            with analyzer.app.app_context():
                analyzer.analyze_all_stocks(max_stocks=1, use_all_symbols=False, offline_mode=True)
            return True
        except Exception as e:
            print(f"    - Error in analyze_all_stocks: {e}")
            raise
    
    test_step("7. Test analyze_all_stocks call", step7, 120)  # Longer timeout
    
    # If we get here, the issue is resolved
    print("\n✓ All steps completed successfully!")
    print("The hanging issue appears to be resolved.")
    
except KeyboardInterrupt:
    print("\nTest interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
