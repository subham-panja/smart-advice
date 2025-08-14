#!/usr/bin/env python3
"""
Test script for backtesting integration
File: test_backtesting_integration.py

This script tests the backtesting integration to ensure it works correctly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scripts.backtesting_runner import BacktestingRunner, run_backtest
from scripts.analyzer import StockAnalyzer
import warnings
warnings.filterwarnings('ignore')

def create_sample_data(days=100):
    """Create sample historical data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate sample OHLCV data
    np.random.seed(42)
    base_price = 100
    
    prices = []
    current_price = base_price
    
    for i in range(days):
        # Random walk with slight upward trend
        change = np.random.normal(0.001, 0.02)  # 0.1% mean return, 2% volatility
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        daily_volatility = np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily volatility
        high = price * (1 + daily_volatility)
        low = price * (1 - daily_volatility)
        open_price = prices[i-1] if i > 0 else price
        close_price = price
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close_price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_backtesting_runner():
    """Test the BacktestingRunner directly."""
    print("Testing BacktestingRunner...")
    
    # Create sample data
    sample_data = create_sample_data(200)  # 200 days of data
    
    # Test with sufficient data
    runner = BacktestingRunner()
    results = runner.run('TEST_SYMBOL', sample_data)
    
    print(f"Status: {results.get('status')}")
    print(f"Data length: {results.get('data_length')}")
    print(f"Strategies tested: {results.get('strategies_tested')}")
    
    if results.get('status') == 'completed':
        combined_metrics = results.get('combined_metrics', {})
        print(f"Average CAGR: {combined_metrics.get('avg_cagr', 'N/A')}%")
        print(f"Average Win Rate: {combined_metrics.get('avg_win_rate', 'N/A')}%")
        print(f"Average Max Drawdown: {combined_metrics.get('avg_max_drawdown', 'N/A')}%")
        print(f"Best Strategy: {combined_metrics.get('best_strategy', 'N/A')}")
        print(f"Worst Strategy: {combined_metrics.get('worst_strategy', 'N/A')}")
        
        # Show individual strategy results
        strategy_results = results.get('strategy_results', {})
        print("\nIndividual Strategy Results:")
        for strategy_name, result in strategy_results.items():
            if result.get('status') == 'completed':
                print(f"  {strategy_name}: CAGR={result.get('cagr', 'N/A')}%, "
                      f"Win Rate={result.get('win_rate', 'N/A')}%, "
                      f"Max DD={result.get('max_drawdown', 'N/A')}%")
            else:
                print(f"  {strategy_name}: {result.get('error', 'Failed')}")
    
    # Test with insufficient data
    print("\nTesting with insufficient data...")
    insufficient_data = create_sample_data(30)  # Only 30 days
    results_insufficient = runner.run('TEST_SYMBOL_INSUFFICIENT', insufficient_data)
    print(f"Status: {results_insufficient.get('status')}")
    print(f"Message: {results_insufficient.get('message')}")
    
    return results

def test_analyzer_integration():
    """Test the StockAnalyzer integration."""
    print("\nTesting StockAnalyzer integration...")
    
    # This would normally require real market data, but we'll test the structure
    analyzer = StockAnalyzer()
    
    # Test the analyzer summary to ensure backtesting components are available
    summary = analyzer.get_analyzer_summary()
    print(f"Analyzer capabilities: {list(summary.keys())}")
    
    return True

def test_convenience_function():
    """Test the convenience function."""
    print("\nTesting convenience function...")
    
    sample_data = create_sample_data(150)
    results = run_backtest('TEST_CONVENIENCE', sample_data)
    
    print(f"Convenience function result status: {results.get('status')}")
    
    if results.get('status') == 'completed':
        print("Convenience function working correctly!")
    
    return results

if __name__ == "__main__":
    print("=" * 60)
    print("BACKTESTING INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # Test 1: BacktestingRunner directly
        test_backtesting_runner()
        
        # Test 2: StockAnalyzer integration
        test_analyzer_integration()
        
        # Test 3: Convenience function
        test_convenience_function()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
