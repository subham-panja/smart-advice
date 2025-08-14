#!/usr/bin/env python3
"""
Simple test script to verify that data fetching works correctly.
"""

import os
import sys

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.data_fetcher import get_historical_data, get_filtered_nse_symbols
from utils.logger import setup_logging

logger = setup_logging()

def test_data_fetching():
    """Test basic data fetching functionality."""
    
    print("=== Testing Data Fetching ===")
    
    # Test 1: Get filtered symbols for 2 stocks
    print("\n1. Testing symbol filtering...")
    try:
        symbols = get_filtered_nse_symbols(max_stocks=2)
        print(f"   Found {len(symbols)} filtered symbols: {list(symbols.keys())}")
        
        if len(symbols) == 0:
            print("   ERROR: No symbols found after filtering!")
            return False
            
    except Exception as e:
        print(f"   ERROR: Failed to get filtered symbols: {e}")
        return False
    
    # Test 2: Get historical data for first symbol
    print("\n2. Testing historical data fetching...")
    if symbols:
        first_symbol = list(symbols.keys())[0]
        print(f"   Testing with symbol: {first_symbol}")
        
        try:
            data = get_historical_data(first_symbol, period='2y', interval='1d')
            
            if not data.empty:
                print(f"   SUCCESS: Got {len(data)} data points")
                print(f"   Columns: {list(data.columns)}")
                print(f"   Date range: {data.index[0]} to {data.index[-1]}")
                print(f"   Sample data:")
                print(data.head(3))
                return True
            else:
                print(f"   ERROR: No historical data returned for {first_symbol}")
                return False
                
        except Exception as e:
            print(f"   ERROR: Failed to get historical data for {first_symbol}: {e}")
            return False
    
    return False

if __name__ == "__main__":
    success = test_data_fetching()
    
    if success:
        print("\n=== SUCCESS: All tests passed! ===")
        sys.exit(0)
    else:
        print("\n=== FAILED: Some tests failed! ===")
        sys.exit(1)
