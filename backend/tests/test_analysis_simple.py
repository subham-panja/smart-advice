#!/usr/bin/env python3
import os
import sys
import gc

# Fix OpenMP/threading issues on macOS - MUST be set before importing numpy/scipy/sklearn
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports."""
    try:
        print("Testing imports...")
        
        print("  - app...")
        from app import create_app
        
        print("  - data_fetcher...")
        from scripts.data_fetcher import get_filtered_nse_symbols
        
        print("  - analyzer...")
        from scripts.analyzer import StockAnalyzer
        
        print("  - database...")
        from database import get_mongodb
        
        print("  - logger...")
        from utils.logger import setup_logging
        
        print("✓ All imports successful")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality step by step."""
    try:
        print("\nTesting basic functionality...")
        
        # Test app creation
        print("  - Creating Flask app...")
        from app import create_app
        app = create_app()
        print("  ✓ Flask app created")
        
        # Test database in app context
        with app.app_context():
            print("  - Testing database connection...")
            from database import get_mongodb
            db = get_mongodb()
            collections = db.list_collection_names()
            print(f"  ✓ Database connected, collections: {collections}")
            
            # Test data fetcher
            print("  - Testing data fetcher...")
            from scripts.data_fetcher import get_filtered_nse_symbols
            symbols = get_filtered_nse_symbols(2)
            print(f"  ✓ Got {len(symbols)} symbols: {list(symbols.keys())[:2]}")
            
            # Test analyzer creation
            print("  - Creating analyzer...")
            from scripts.analyzer import StockAnalyzer
            analyzer = StockAnalyzer()
            print("  ✓ Analyzer created")
            
            print("✓ Basic functionality test passed")
            return True
            
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_single_analysis():
    """Test analyzing a single stock."""
    try:
        print("\nTesting single stock analysis...")
        
        from app import create_app
        from scripts.analyzer import StockAnalyzer
        
        app = create_app()
        with app.app_context():
            analyzer = StockAnalyzer()
            
            print("  - Analyzing RELIANCE...")
            # Set a timeout to prevent hanging
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Analysis timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)  # 60 second timeout
            
            try:
                result = analyzer.analyze_stock('RELIANCE', app.config)
                signal.alarm(0)  # Cancel timeout
                
                print(f"  ✓ Analysis completed for RELIANCE")
                print(f"  - Is recommended: {result.get('is_recommended', False)}")
                print(f"  - Recommendation strength: {result.get('recommendation_strength', 'N/A')}")
                print(f"  - Combined score: {result.get('combined_score', 0.0):.4f}")
                
                # Check for important fields
                required_fields = ['symbol', 'company_name', 'technical_score', 'fundamental_score', 'sentiment_score']
                missing_fields = [field for field in required_fields if field not in result]
                if missing_fields:
                    print(f"  ⚠ Missing fields: {missing_fields}")
                else:
                    print("  ✓ All required fields present")
                
                return True
                
            except TimeoutError:
                signal.alarm(0)
                print("  ✗ Analysis timed out after 60 seconds")
                return False
                
    except Exception as e:
        print(f"✗ Single analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=== Minimal Analysis Test ===")
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality),
        ("Single Analysis Test", test_single_analysis)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} failed - stopping here")
            break
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
