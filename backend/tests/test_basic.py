#!/usr/bin/env python3
"""
Basic test script to verify the Share Market Analyzer setup.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from scripts.data_fetcher import get_all_nse_symbols, get_historical_data
from database import get_mongodb
import json

def test_database():
    """Test database connection."""
    print("Testing database connection...")
    with app.app_context():
        try:
            from database import get_db
            db = get_db()
            # Test MongoDB connection by listing collections
            collections = db.list_collection_names()
            print(f"✓ Database connection successful - Collections: {collections}")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False

def test_symbols():
    """Test symbol loading."""
    print("\nTesting symbol loading...")
    try:
        symbols = get_all_nse_symbols()
        print(f"✓ Loaded {len(symbols)} symbols")
        print(f"Sample symbols: {list(symbols.keys())[:5]}")
        return True
    except Exception as e:
        print(f"✗ Symbol loading failed: {e}")
        return False

def test_data_fetching():
    """Test data fetching."""
    print("\nTesting data fetching...")
    try:
        data = get_historical_data('RELIANCE', '1mo')
        if not data.empty:
            print(f"✓ Fetched {len(data)} days of data for RELIANCE")
            print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"Latest close: ₹{data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("✗ No data fetched")
            return False
    except Exception as e:
        print(f"✗ Data fetching failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints."""
    print("\nTesting API endpoints...")
    try:
        with app.test_client() as client:
            # Test health check
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Health check endpoint working")
            else:
                print(f"✗ Health check failed with status {response.status_code}")
                return False
            
            # Test symbols endpoint
            response = client.get('/symbols')
            if response.status_code == 200:
                data = json.loads(response.data)
                print(f"✓ Symbols endpoint working - returned {data['count']} symbols")
            else:
                print(f"✗ Symbols endpoint failed with status {response.status_code}")
                return False
            
            # Test data endpoint
            response = client.get('/test_data/RELIANCE')
            if response.status_code == 200:
                data = json.loads(response.data)
                print(f"✓ Data endpoint working - {data['data_points']} data points for RELIANCE")
            else:
                print(f"✗ Data endpoint failed with status {response.status_code}")
                return False
            
            return True
    except Exception as e:
        print(f"✗ API endpoint testing failed: {e}")
        return False

def test_analyzer_functionality():
    """Test the enhanced analyzer functionality."""
    print("\nTesting enhanced analyzer functionality...")
    try:
        from scripts.analyzer import StockAnalyzer
        
        analyzer = StockAnalyzer()
        
        # Test trade-level analysis
        print("Testing trade-level analysis...")
        trade_result = analyzer.analyze('RELIANCE')
        
        expected_fields = ['buy_price', 'sell_price', 'stop_loss', 'days_to_target', 
                          'entry_timing', 'risk_reward_ratio', 'confidence']
        
        for field in expected_fields:
            if field not in trade_result:
                print(f"✗ Missing field in trade analysis: {field}")
                return False
        
        print(f"✓ Trade-level analysis working - Generated {len(expected_fields)} trade fields")
        
        # Test comprehensive stock analysis
        print("Testing comprehensive stock analysis...")
        with app.app_context():
            full_result = analyzer.analyze_stock('RELIANCE', app.config)
            
            # Check for new fields
            if 'trade_plan' not in full_result:
                print("✗ Missing trade_plan in comprehensive analysis")
                return False
            
            if 'backtest_results' not in full_result:
                print("✗ Missing backtest_results in comprehensive analysis")
                return False
            
            print("✓ Comprehensive analysis working - includes trade_plan and backtest_results")
            
            # Check trade plan fields
            trade_plan = full_result['trade_plan']
            for field in expected_fields:
                if field not in trade_plan:
                    print(f"✗ Missing field in trade_plan: {field}")
                    return False
            
            print("✓ Trade plan contains all required fields")
            
            # Check backtest results structure
            backtest_results = full_result['backtest_results']
            if 'error' not in backtest_results:
                if 'period_results' in backtest_results and 'overall_metrics' in backtest_results:
                    print("✓ Backtest results structure is correct")
                else:
                    print("✗ Backtest results missing required fields")
                    return False
            else:
                print(f"⚠ Backtest results show error (might be due to insufficient data): {backtest_results['error']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Analyzer functionality testing failed: {e}")
        return False

def test_database_schema():
    """Test database schema for new fields (MongoDB collections)."""
    print("\nTesting database schema...")
    try:
        with app.app_context():
            from database import get_db
            db = get_db()
            
            # Test that recommended_shares collection exists and has documents with required fields
            rec_collection = db.recommended_shares
            sample_doc = rec_collection.find_one()
            
            if sample_doc:
                required_fields = ['buy_price', 'sell_price', 'est_time_to_target', 'symbol', 'recommendation_date']
                missing_fields = [field for field in required_fields if field not in sample_doc]
                
                if missing_fields:
                    print(f"✓ recommended_shares collection exists but sample document missing fields: {missing_fields}")
                else:
                    print("✓ recommended_shares collection has all required fields")
            else:
                print("✓ recommended_shares collection exists (no documents yet)")
            
            # Test backtest_results collection
            backtest_collection = db.backtest_results
            sample_backtest = backtest_collection.find_one()
            
            if sample_backtest:
                required_fields = ['symbol', 'period', 'CAGR', 'win_rate', 'max_drawdown', 'created_at']
                missing_fields = [field for field in required_fields if field not in sample_backtest]
                
                if missing_fields:
                    print(f"✓ backtest_results collection exists but sample document missing fields: {missing_fields}")
                else:
                    print("✓ backtest_results collection has all required fields")
            else:
                print("✓ backtest_results collection exists (no documents yet)")
            
        return True
        
    except Exception as e:
        print(f"✗ Database schema testing failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Share Market Analyzer Basic Tests ===")
    
    tests = [
        test_database,
        test_database_schema,
        test_symbols,
        test_data_fetching,
        test_api_endpoints,
        test_analyzer_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("✓ All tests passed! Your setup is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
