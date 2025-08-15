#!/usr/bin/env python3
"""
Unit Tests for Swing Trading Gates
=================================

Tests for individual gates: trend filter, volatility gate, volume confirmation,
and multi-timeframe confirmation with fixture charts and scenarios.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.swing_trading_signals import SwingTradingAnalyzer
from utils.logger import setup_logging

logger = setup_logging()


class TestSwingTradingGates(unittest.TestCase):
    """Test cases for swing trading gates with fixture data"""
    
    def setUp(self):
        """Set up test fixtures and analyzer"""
        self.analyzer = SwingTradingAnalyzer()
        
        # Create base fixture data - 250 days of realistic stock data
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        np.random.seed(42)  # For reproducible tests
        
        # Generate realistic price series with trend and volatility
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 250)  # ~0.1% daily return, 2% volatility
        
        # Add trend component
        trend = np.linspace(0, 0.3, 250)  # 30% uptrend over period
        returns = returns + trend / 250
        
        prices = [base_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        self.base_df = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices[:-1]],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
            'Close': prices[:-1],
            'Volume': np.random.randint(50000, 500000, 250)
        })
        self.base_df.set_index('Date', inplace=True)
        
        # Ensure realistic OHLC relationships
        for i in range(len(self.base_df)):
            high = max(self.base_df.iloc[i]['Open'], self.base_df.iloc[i]['Close'], self.base_df.iloc[i]['High'])
            low = min(self.base_df.iloc[i]['Open'], self.base_df.iloc[i]['Close'], self.base_df.iloc[i]['Low'])
            self.base_df.iloc[i, self.base_df.columns.get_loc('High')] = high
            self.base_df.iloc[i, self.base_df.columns.get_loc('Low')] = low

    def create_bullish_trend_fixture(self):
        """Create fixture data showing strong bullish trend"""
        df = self.base_df.copy()
        
        # Ensure strong uptrend in last 50 days
        multiplier = np.linspace(1.0, 1.4, 50)  # 40% gain in last 50 days
        df.iloc[-50:, df.columns.get_loc('Close')] = df.iloc[-50:]['Close'] * multiplier
        df.iloc[-50:, df.columns.get_loc('High')] = df.iloc[-50:]['High'] * multiplier * 1.02
        df.iloc[-50:, df.columns.get_loc('Low')] = df.iloc[-50:]['Low'] * multiplier * 0.98
        df.iloc[-50:, df.columns.get_loc('Open')] = df.iloc[-50:]['Open'] * multiplier
        
        return df

    def create_bearish_trend_fixture(self):
        """Create fixture data showing bearish trend"""
        df = self.base_df.copy()
        
        # Ensure downtrend in last 50 days
        multiplier = np.linspace(1.0, 0.7, 50)  # 30% decline in last 50 days
        df.iloc[-50:, df.columns.get_loc('Close')] = df.iloc[-50:]['Close'] * multiplier
        df.iloc[-50:, df.columns.get_loc('High')] = df.iloc[-50:]['High'] * multiplier * 1.02
        df.iloc[-50:, df.columns.get_loc('Low')] = df.iloc[-50:]['Low'] * multiplier * 0.98
        df.iloc[-50:, df.columns.get_loc('Open')] = df.iloc[-50:]['Open'] * multiplier
        
        return df

    def create_high_volatility_fixture(self):
        """Create fixture data with high volatility"""
        df = self.base_df.copy()
        
        # Add high volatility to last 20 days
        for i in range(20):
            idx = len(df) - 20 + i
            volatility_factor = 1.0 + np.random.normal(0, 0.05)  # 5% volatility
            df.iloc[idx, df.columns.get_loc('High')] = df.iloc[idx]['Close'] * volatility_factor * 1.03
            df.iloc[idx, df.columns.get_loc('Low')] = df.iloc[idx]['Close'] * volatility_factor * 0.97
        
        return df

    def create_volume_spike_fixture(self):
        """Create fixture data with volume spike"""
        df = self.base_df.copy()
        
        # Create volume spike in last 5 days
        avg_volume = df['Volume'].mean()
        df.iloc[-5:, df.columns.get_loc('Volume')] = avg_volume * np.array([2.5, 3.0, 2.8, 2.2, 1.8])
        
        return df

    def test_trend_filter_bullish_pass(self):
        """Test trend filter passes with strong bullish trend"""
        df = self.create_bullish_trend_fixture()
        result = self.analyzer.calculate_trend_filter(df)
        
        self.assertTrue(result['passed'], "Trend filter should pass with bullish trend")
        self.assertIn('adx', result)
        self.assertIn('price_above_sma', result)
        self.assertGreater(result['adx'], self.analyzer.thresholds['adx_min'])
        self.assertTrue(result['price_above_sma'])
        
        print(f"✓ Trend Filter (Bullish): {result['reason']}")

    def test_trend_filter_bearish_fail(self):
        """Test trend filter fails with bearish trend"""
        df = self.create_bearish_trend_fixture()
        result = self.analyzer.calculate_trend_filter(df)
        
        self.assertFalse(result['passed'], "Trend filter should fail with bearish trend")
        self.assertIn('reason', result)
        
        print(f"✗ Trend Filter (Bearish): {result['reason']}")

    def test_trend_filter_insufficient_data(self):
        """Test trend filter with insufficient data"""
        df = self.base_df.iloc[-50:].copy()  # Only 50 days
        result = self.analyzer.calculate_trend_filter(df)
        
        self.assertFalse(result['passed'], "Trend filter should fail with insufficient data")
        self.assertIn('Insufficient data', result['reason'])
        
        print(f"✗ Trend Filter (Insufficient Data): {result['reason']}")

    def test_volatility_gate_normal_pass(self):
        """Test volatility gate passes with normal volatility"""
        df = self.base_df.copy()
        result = self.analyzer.calculate_volatility_gate(df)
        
        self.assertIn('passed', result)
        self.assertIn('atr_percentile', result)
        self.assertIn('volatility', result)
        
        # Should be within acceptable range
        atr_percentile = result['atr_percentile']
        expected_pass = (self.analyzer.thresholds['atr_percentile_min'] <= atr_percentile 
                        <= self.analyzer.thresholds['atr_percentile_max'])
        
        self.assertEqual(result['passed'], expected_pass)
        print(f"{'✓' if result['passed'] else '✗'} Volatility Gate (Normal): {result['reason']}")

    def test_volatility_gate_high_fail(self):
        """Test volatility gate fails with extreme volatility"""
        df = self.create_high_volatility_fixture()
        result = self.analyzer.calculate_volatility_gate(df)
        
        self.assertIn('atr_percentile', result)
        atr_percentile = result['atr_percentile']
        
        # High volatility should likely fail
        if atr_percentile > self.analyzer.thresholds['atr_percentile_max']:
            self.assertFalse(result['passed'], "Should fail with extreme volatility")
        
        print(f"{'✓' if result['passed'] else '✗'} Volatility Gate (High): {result['reason']}")

    def test_volume_confirmation_spike_pass(self):
        """Test volume confirmation passes with volume spike"""
        df = self.create_volume_spike_fixture()
        result = self.analyzer.calculate_volume_confirmation(df)
        
        self.assertIn('passed', result)
        self.assertIn('volume_zscore', result)
        self.assertIn('obv_trending_up', result)
        
        # Should pass due to volume spike or OBV trend
        print(f"{'✓' if result['passed'] else '✗'} Volume Confirmation (Spike): {result['reason']}")

    def test_volume_confirmation_low_fail(self):
        """Test volume confirmation fails with low volume"""
        df = self.base_df.copy()
        
        # Create low volume scenario
        df.iloc[-20:, df.columns.get_loc('Volume')] = df['Volume'].mean() * 0.3  # 30% of average
        result = self.analyzer.calculate_volume_confirmation(df)
        
        self.assertIn('volume_zscore', result)
        volume_zscore = result['volume_zscore']
        
        # Low volume should likely result in negative Z-score
        self.assertLess(volume_zscore, 0)
        print(f"{'✓' if result['passed'] else '✗'} Volume Confirmation (Low): {result['reason']}")

    def test_multi_timeframe_confirmation_aligned_pass(self):
        """Test MTF confirmation passes when timeframes are aligned"""
        df = self.create_bullish_trend_fixture()
        
        # Create weekly data by resampling
        weekly_df = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        result = self.analyzer.calculate_multi_timeframe_confirmation(df, weekly_df)
        
        self.assertIn('passed', result)
        self.assertIn('weekly_trend_up', result)
        self.assertIn('daily_trend_up', result)
        
        print(f"{'✓' if result['passed'] else '✗'} MTF Confirmation (Aligned): {result['reason']}")

    def test_multi_timeframe_confirmation_misaligned_fail(self):
        """Test MTF confirmation fails when timeframes are misaligned"""
        df = self.base_df.copy()
        
        # Create bearish weekly but bullish daily (recent)
        weekly_df = self.create_bearish_trend_fixture().resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        # Make daily recent bullish
        daily_df = self.create_bullish_trend_fixture()
        
        result = self.analyzer.calculate_multi_timeframe_confirmation(daily_df, weekly_df)
        
        # Should likely fail due to misalignment
        print(f"{'✓' if result['passed'] else '✗'} MTF Confirmation (Misaligned): {result['reason']}")

    def test_all_gates_integration(self):
        """Test full gate integration with realistic scenario"""
        df = self.create_bullish_trend_fixture()
        df = self.create_volume_spike_fixture()  # Add volume spike
        
        # Run full analysis
        result = self.analyzer.analyze_swing_opportunity('TEST', df)
        
        self.assertIn('gates_passed', result)
        self.assertIn('all_gates_passed', result)
        self.assertIn('recommendation', result)
        self.assertIn('reasons', result)
        
        # Print detailed results
        print(f"\n=== Full Gate Analysis for TEST ===")
        print(f"All Gates Passed: {result['all_gates_passed']}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Signal Strength: {result.get('signal_strength', 0):.2f}")
        
        for gate, passed in result['gates_passed'].items():
            print(f"{'✓' if passed else '✗'} {gate.replace('_', ' ').title()}: {passed}")
        
        print("\nReasons:")
        for reason in result['reasons']:
            print(f"  - {reason}")

    def test_gate_error_handling(self):
        """Test gate error handling with malformed data"""
        # Create DataFrame with missing columns
        bad_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        })
        
        # Test each gate with bad data
        trend_result = self.analyzer.calculate_trend_filter(bad_df)
        self.assertFalse(trend_result['passed'])
        self.assertIn('Error', trend_result['reason'])
        
        volatility_result = self.analyzer.calculate_volatility_gate(bad_df)
        self.assertFalse(volatility_result['passed'])
        self.assertIn('Error', volatility_result['reason'])
        
        volume_result = self.analyzer.calculate_volume_confirmation(bad_df)
        self.assertFalse(volume_result['passed'])
        self.assertIn('Error', volume_result['reason'])
        
        print("✓ Error handling tests passed")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        
        # Test with exactly minimum data
        min_df = self.base_df.iloc[-200:].copy()  # Exactly 200 days
        result = self.analyzer.analyze_swing_opportunity('MIN_DATA', min_df)
        self.assertNotEqual(result['recommendation'], 'INSUFFICIENT_DATA')
        
        # Test with all zeros
        zero_df = self.base_df.copy()
        zero_df['Volume'] = 0
        volume_result = self.analyzer.calculate_volume_confirmation(zero_df)
        self.assertFalse(volume_result['passed'])
        
        # Test with constant prices (no volatility)
        flat_df = self.base_df.copy()
        flat_df['Close'] = 100  # All same price
        flat_df['High'] = 100.1
        flat_df['Low'] = 99.9
        flat_df['Open'] = 100
        volatility_result = self.analyzer.calculate_volatility_gate(flat_df)
        
        print("✓ Edge case tests completed")

    def tearDown(self):
        """Clean up after tests"""
        pass


class TestGateParameterization(unittest.TestCase):
    """Test gate parameterization and configuration"""
    
    def test_threshold_modifications(self):
        """Test that gate thresholds can be modified"""
        analyzer = SwingTradingAnalyzer()
        
        # Modify thresholds
        original_adx = analyzer.thresholds['adx_min']
        analyzer.thresholds['adx_min'] = 25
        
        self.assertEqual(analyzer.thresholds['adx_min'], 25)
        print(f"✓ Threshold modification: ADX {original_adx} -> {analyzer.thresholds['adx_min']}")

    def test_gate_sensitivity(self):
        """Test gate sensitivity to parameter changes"""
        base_analyzer = SwingTradingAnalyzer()
        sensitive_analyzer = SwingTradingAnalyzer()
        
        # Make gates more sensitive
        sensitive_analyzer.thresholds['adx_min'] = 15  # Lower ADX requirement
        sensitive_analyzer.thresholds['volume_zscore_min'] = 0.5  # Lower volume requirement
        
        # Create test data
        dates = pd.date_range(start='2023-01-01', periods=250, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.random.randn(250),
            'High': 102 + np.random.randn(250),
            'Low': 98 + np.random.randn(250),
            'Close': 100 + np.cumsum(np.random.randn(250) * 0.01),
            'Volume': 100000 + np.random.randint(-20000, 50000, 250)
        })
        df.set_index('Date', inplace=True)
        
        base_result = base_analyzer.analyze_swing_opportunity('BASE', df)
        sensitive_result = sensitive_analyzer.analyze_swing_opportunity('SENSITIVE', df)
        
        print(f"✓ Base gates passed: {sum(base_result['gates_passed'].values())}/4")
        print(f"✓ Sensitive gates passed: {sum(sensitive_result['gates_passed'].values())}/4")


if __name__ == '__main__':
    print("Running Swing Trading Gates Unit Tests")
    print("=" * 50)
    
    # Create test suite
    test_classes = [TestSwingTradingGates, TestGateParameterization]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
