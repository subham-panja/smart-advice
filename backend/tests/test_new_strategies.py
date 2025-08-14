#!/usr/bin/env python3
"""
Test script for new advanced pattern recognition strategies
File: test_new_strategies.py

This script tests the newly implemented strategies:
- Chart Patterns
- Volume Profile
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.strategies.chart_patterns import ChartPatterns
from scripts.strategies.volume_profile import VolumeProfile
from utils.logger import setup_logging

logger = setup_logging()

def create_sample_data():
    """Create sample OHLCV data for testing."""
    
    # Create 100 days of sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Start with base price of 100
    base_price = 100
    prices = [base_price]
    
    # Generate realistic price movements
    for i in range(99):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(1, new_price))  # Prevent negative prices
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Create realistic OHLC from close price
        volatility = np.random.uniform(0.01, 0.03)  # 1-3% intraday range
        high = price * (1 + volatility/2)
        low = price * (1 - volatility/2)
        open_price = prices[i-1] if i > 0 else price
        
        # Generate volume (higher volume on larger moves)
        price_change = abs(price - (prices[i-1] if i > 0 else price))
        base_volume = 10000
        volume = base_volume * (1 + price_change * 10)
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': price,
            'Volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    return df

def create_pattern_data():
    """Create data with specific patterns for testing."""
    
    # Create data with inside bar pattern
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    
    data = []
    base_price = 100
    
    for i in range(50):
        if i == 48:  # Create inside bar pattern at the end
            # Previous bar (larger range)
            open_price = base_price
            high = base_price + 2
            low = base_price - 2
            close = base_price + 1
        elif i == 49:  # Inside bar (contained within previous)
            open_price = base_price + 0.5
            high = base_price + 1.5  # Within previous bar's range
            low = base_price - 1.5   # Within previous bar's range
            close = base_price + 1
        else:
            # Normal price action
            change = np.random.normal(0, 0.01)
            close = base_price * (1 + change)
            open_price = base_price
            high = close * 1.01
            low = close * 0.99
        
        volume = 10000 * (1 + abs(np.random.normal(0, 0.1)))
        
        data.append({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
        
        base_price = close
    
    df = pd.DataFrame(data, index=dates)
    return df

def test_chart_patterns():
    """Test the Chart Patterns strategy."""
    
    print("\n" + "="*60)
    print("TESTING CHART PATTERNS STRATEGY")
    print("="*60)
    
    try:
        # Initialize strategy
        strategy = ChartPatterns()
        print(f"✓ Chart Patterns strategy initialized")
        
        # Test with sample data
        sample_data = create_sample_data()
        print(f"✓ Sample data created: {len(sample_data)} days")
        
        # Run strategy
        signal = strategy.run_strategy(sample_data)
        print(f"✓ Strategy executed successfully")
        print(f"  Signal: {signal} ({'BUY' if signal == 1 else 'SELL/HOLD'})")
        
        # Test with pattern data
        pattern_data = create_pattern_data()
        print(f"✓ Pattern data created: {len(pattern_data)} days")
        
        signal_pattern = strategy.run_strategy(pattern_data)
        print(f"✓ Strategy executed on pattern data")
        print(f"  Signal: {signal_pattern} ({'BUY' if signal_pattern == 1 else 'SELL/HOLD'})")
        
        print("✅ Chart Patterns strategy test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Chart Patterns strategy test failed: {e}")
        logger.error(f"Chart Patterns test error: {e}")
        return False

def test_volume_profile():
    """Test the Volume Profile strategy."""
    
    print("\n" + "="*60)
    print("TESTING VOLUME PROFILE STRATEGY")
    print("="*60)
    
    try:
        # Initialize strategy
        strategy = VolumeProfile()
        print(f"✓ Volume Profile strategy initialized")
        
        # Test with sample data
        sample_data = create_sample_data()
        print(f"✓ Sample data created: {len(sample_data)} days")
        
        # Run strategy
        signal = strategy.run_strategy(sample_data)
        print(f"✓ Strategy executed successfully")
        print(f"  Signal: {signal} ({'BUY' if signal == 1 else 'SELL/HOLD'})")
        
        # Create high volume data to test volume profile
        high_volume_data = sample_data.copy()
        # Add some high volume spikes
        high_volume_data.loc[high_volume_data.index[-10:], 'Volume'] *= 3
        
        signal_hv = strategy.run_strategy(high_volume_data)
        print(f"✓ Strategy executed on high volume data")
        print(f"  Signal: {signal_hv} ({'BUY' if signal_hv == 1 else 'SELL/HOLD'})")
        
        print("✅ Volume Profile strategy test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Volume Profile strategy test failed: {e}")
        logger.error(f"Volume Profile test error: {e}")
        return False

def test_strategy_integration():
    """Test integration with strategy evaluator."""
    
    print("\n" + "="*60)
    print("TESTING STRATEGY INTEGRATION")
    print("="*60)
    
    try:
        from scripts.strategy_evaluator import StrategyEvaluator
        
        # Create a minimal config for testing
        test_config = {
            'Chart_Patterns': True,
            'Volume_Profile': True,
            'MA_Crossover_50_200': True  # Include one existing strategy
        }
        
        evaluator = StrategyEvaluator(test_config)
        print(f"✓ Strategy Evaluator initialized")
        
        # Get summary
        summary = evaluator.get_strategy_summary()
        print(f"✓ Strategy Summary:")
        print(f"  Total Configured: {summary['total_configured']}")
        print(f"  Total Enabled: {summary['total_enabled']}")
        print(f"  Total Loaded: {summary['total_loaded']}")
        print(f"  Loaded Strategies: {summary['loaded_strategies']}")
        print(f"  Failed Strategies: {summary['failed_strategies']}")
        
        # Test evaluation
        sample_data = create_sample_data()
        results = evaluator.evaluate_strategies('TEST', sample_data)
        
        print(f"✓ Strategy evaluation completed:")
        print(f"  Technical Score: {results['technical_score']:.3f}")
        print(f"  Positive Signals: {results['positive_signals']}/{results['total_strategies']}")
        print(f"  Recommendation: {results['recommendation']}")
        
        print("✅ Strategy integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Strategy integration test failed: {e}")
        logger.error(f"Strategy integration test error: {e}")
        return False

def main():
    """Main test function."""
    
    print("🚀 STARTING ADVANCED STRATEGY TESTS")
    print("="*80)
    
    results = []
    
    # Test individual strategies
    results.append(test_chart_patterns())
    results.append(test_volume_profile())
    results.append(test_strategy_integration())
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Advanced strategies are working correctly.")
        return True
    else:
        print("⚠️  SOME TESTS FAILED. Please check the error messages above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
