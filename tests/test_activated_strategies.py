"""
Unit Tests for Activated Advanced Technical Strategies
File: tests/test_activated_strategies.py

This module contains comprehensive unit tests for the strategies activated in Phase 1.1.
"""

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.strategies.fibonacci_retracement import FibonacciRetracementStrategy
from scripts.strategies.chart_patterns import ChartPatterns
from scripts.strategies.volume_profile import VolumeProfile
from scripts.strategies.gap_trading import Gap_Trading
from scripts.strategies.channel_trading import Channel_Trading
from scripts.strategies.ichimoku_cloud_breakout import Ichimoku_Cloud_Breakout
from scripts.strategies.volume_breakout import VolumeBreakoutStrategy
from scripts.strategies.bollinger_band_breakout import Bollinger_Band_Breakout
from scripts.strategies.macd_signal_crossover import MACD_Signal_Crossover
from utils.volume_analysis import VolumeAnalyzer, get_enhanced_volume_confirmation


class TestActivatedStrategies(unittest.TestCase):
    """Test suite for activated advanced technical strategies."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all tests."""
        # Create sample OHLCV data
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic price data
        base_price = 100
        price_changes = np.random.normal(0, 0.02, len(dates))
        prices = [base_price]
        
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # Ensure price stays positive
        
        # Create OHLCV data
        cls.test_data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(10000, 100000, len(dates))
        }, index=dates)
        
        # Ensure High >= Open, Close and Low <= Open, Close
        cls.test_data['High'] = np.maximum(cls.test_data['High'], 
                                          np.maximum(cls.test_data['Open'], cls.test_data['Close']))
        cls.test_data['Low'] = np.minimum(cls.test_data['Low'], 
                                         np.minimum(cls.test_data['Open'], cls.test_data['Close']))
    
    def test_fibonacci_retracement_strategy(self):
        """Test Fibonacci Retracement strategy."""
        strategy = FibonacciRetracementStrategy()
        
        # Test with sufficient data
        signal = strategy.run_strategy(self.test_data)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")
        
        # Test swing point detection
        swing_info = strategy.find_swing_points(self.test_data)
        if swing_info:
            self.assertIn('trend_direction', swing_info)
            self.assertIn('fib_levels', swing_info)
            self.assertIn(swing_info['trend_direction'], ['uptrend', 'downtrend'])
        
        # Test signal strength calculation
        strength = strategy.get_signal_strength(self.test_data)
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
    
    def test_chart_patterns_strategy(self):
        """Test Chart Patterns strategy."""
        strategy = ChartPatterns()
        
        # Test with sufficient data
        signal = strategy.run_strategy(self.test_data)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")
        
        # Test individual pattern detection methods
        inside_bar = strategy._detect_inside_bars(self.test_data)
        # Inside bar detection should return None or dict
        if inside_bar:
            self.assertIn('name', inside_bar)
            self.assertIn('strength', inside_bar)
        
        nr7 = strategy._detect_nr7_pattern(self.test_data)
        if nr7:
            self.assertIn('name', nr7)
            self.assertIn('strength', nr7)
    
    def test_volume_profile_strategy(self):
        """Test Volume Profile strategy."""
        strategy = VolumeProfile()
        
        # Test with sufficient data
        signal = strategy.run_strategy(self.test_data)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")
        
        # Test volume profile calculation
        volume_profile = strategy._calculate_volume_profile(self.test_data.tail(50))
        if volume_profile:
            self.assertIn('poc_price', volume_profile)
            self.assertIn('value_area_high', volume_profile)
            self.assertIn('value_area_low', volume_profile)
            self.assertGreater(volume_profile['poc_price'], 0)
    
    def test_gap_trading_strategy(self):
        """Test Gap Trading strategy."""
        strategy = Gap_Trading()
        
        # Create test data with a gap
        gap_data = self.test_data.copy()
        # Create a gap-up scenario
        gap_data.iloc[-1, gap_data.columns.get_loc('Open')] = gap_data.iloc[-2, gap_data.columns.get_loc('Close')] * 1.03
        gap_data.iloc[-1, gap_data.columns.get_loc('High')] = gap_data.iloc[-1, gap_data.columns.get_loc('Open')] * 1.01
        gap_data.iloc[-1, gap_data.columns.get_loc('Close')] = gap_data.iloc[-1, gap_data.columns.get_loc('Open')] * 1.005
        gap_data.iloc[-1, gap_data.columns.get_loc('Volume')] = gap_data['Volume'].mean() * 2.5  # High volume
        
        # Test gap detection
        signal = strategy.run_strategy(gap_data)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")
    
    def test_volume_breakout_strategy(self):
        """Test Volume Breakout strategy."""
        strategy = VolumeBreakoutStrategy()
        
        # Test with sufficient data
        signal = strategy.run_strategy(self.test_data)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")
        
        # Test signal calculation
        data_with_signals = strategy.calculate_signals(self.test_data.copy())
        self.assertIn('volume_breakout_signal', data_with_signals.columns)
        
        # Test signal strength
        strength = strategy.get_signal_strength(self.test_data)
        self.assertGreaterEqual(strength, 0.0)
        self.assertLessEqual(strength, 1.0)
    
    def test_bollinger_band_breakout_strategy(self):
        """Test Bollinger Band Breakout strategy."""
        strategy = Bollinger_Band_Breakout()
        
        # Test with sufficient data
        signal = strategy.run_strategy(self.test_data)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")
        
        # Test with insufficient data
        insufficient_data = self.test_data.head(10)
        signal_insufficient = strategy.run_strategy(insufficient_data)
        self.assertEqual(signal_insufficient, -1, "Should return -1 for insufficient data")
    
    def test_macd_signal_crossover_strategy(self):
        """Test MACD Signal Crossover strategy."""
        strategy = MACD_Signal_Crossover()
        
        # Test with sufficient data
        signal = strategy.run_strategy(self.test_data)
        self.assertIn(signal, [-1, 0, 1], "Signal should be -1, 0, or 1")
        
        # Test with insufficient data
        insufficient_data = self.test_data.head(20)
        signal_insufficient = strategy.run_strategy(insufficient_data)
        self.assertEqual(signal_insufficient, -1, "Should return -1 for insufficient data")
    
    def test_enhanced_volume_confirmation(self):
        """Test enhanced volume confirmation system."""
        # Test volume analyzer
        analyzer = VolumeAnalyzer()
        
        # Test volume confirmation factor
        confirmation = analyzer.get_volume_confirmation_factor(self.test_data, 'bullish')
        self.assertIn('factor', confirmation)
        self.assertIn('strength', confirmation)
        self.assertGreater(confirmation['factor'], 0)
        
        # Test volume breakout detection
        breakout = analyzer.detect_volume_breakout(self.test_data, price_breakout=True)
        self.assertIn('detected', breakout)
        self.assertIn('strength', breakout)
        
        # Test VWAP analysis
        vwap = analyzer.get_volume_weighted_price(self.test_data)
        self.assertIn('analysis', vwap)
        
        # Test convenience function
        enhanced = get_enhanced_volume_confirmation(self.test_data, 'bullish', breakout=True)
        self.assertIn('factor', enhanced)
        self.assertIn('strength', enhanced)
    
    def test_strategy_integration(self):
        """Test strategy integration with enhanced volume confirmation."""
        strategies = [
            FibonacciRetracementStrategy(),
            ChartPatterns(),
            VolumeProfile(),
            Gap_Trading(),
            VolumeBreakoutStrategy(),
            Bollinger_Band_Breakout(),
            MACD_Signal_Crossover()
        ]
        
        for strategy in strategies:
            with self.subTest(strategy=strategy.name):
                # Test that each strategy can run without errors
                try:
                    signal = strategy.run_strategy(self.test_data)
                    self.assertIn(signal, [-1, 0, 1], f"Strategy {strategy.name} should return valid signal")
                except Exception as e:
                    self.fail(f"Strategy {strategy.name} failed with error: {e}")
                
                # Test volume confirmation methods if available
                if hasattr(strategy, 'apply_volume_filtering'):
                    try:
                        volume_result = strategy.apply_volume_filtering(1, self.test_data)
                        self.assertIn('signal', volume_result)
                        self.assertIn('volume_factor', volume_result)
                    except Exception as e:
                        self.fail(f"Volume filtering failed for {strategy.name}: {e}")


class TestRealDataIntegration(unittest.TestCase):
    """Test strategies with real market data."""
    
    def setUp(self):
        """Set up real market data for testing."""
        try:
            # Try to fetch real data
            ticker = yf.Ticker("RELIANCE.NS")
            self.real_data = ticker.history(period="6mo")
            self.has_real_data = not self.real_data.empty
        except:
            self.has_real_data = False
            self.real_data = None
    
    def test_strategies_with_real_data(self):
        """Test strategies with real market data."""
        if not self.has_real_data:
            self.skipTest("Real market data not available")
        
        strategies = [
            ('Fibonacci_Retracement', FibonacciRetracementStrategy()),
            ('Chart_Patterns', ChartPatterns()),
            ('Volume_Profile', VolumeProfile()),
            ('Volume_Breakout', VolumeBreakoutStrategy()),
            ('Bollinger_Band_Breakout', Bollinger_Band_Breakout()),
            ('MACD_Signal_Crossover', MACD_Signal_Crossover())
        ]
        
        results = {}
        for name, strategy in strategies:
            with self.subTest(strategy=name):
                try:
                    signal = strategy.run_strategy(self.real_data)
                    results[name] = signal
                    self.assertIn(signal, [-1, 0, 1], f"Strategy {name} should return valid signal")
                except Exception as e:
                    self.fail(f"Strategy {name} failed with real data: {e}")
        
        # Print results for manual verification
        print(f"\nReal Data Test Results:")
        for name, signal in results.items():
            signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
            print(f"{name}: {signal_text}")


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestActivatedStrategies))
    test_suite.addTest(unittest.makeSuite(TestRealDataIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
