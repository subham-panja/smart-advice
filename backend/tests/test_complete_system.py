#!/usr/bin/env python3
"""
Complete System Test
File: test_complete_system.py

This script tests the complete Share Market Analyzer system including
technical analysis, fundamental analysis, and sentiment analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app
from scripts.analyzer import StockAnalyzer
from scripts.strategy_evaluator import StrategyEvaluator
from scripts.data_fetcher import get_all_nse_symbols, get_historical_data
from scripts.sentiment_analysis import SentimentAnalysis
from scripts.fundamental_analysis import FundamentalAnalysis
import json

def test_strategy_evaluator():
    """Test the strategy evaluator."""
    print("\\n=== Testing Strategy Evaluator ===")
    
    try:
        evaluator = StrategyEvaluator()
        summary = evaluator.get_strategy_summary()
        
        print(f"✓ Strategy Evaluator initialized")
        print(f"  - Total configured strategies: {summary['total_configured']}")
        print(f"  - Total enabled strategies: {summary['total_enabled']}")
        print(f"  - Total loaded strategies: {summary['total_loaded']}")
        print(f"  - Loaded strategies: {summary['loaded_strategies']}")
        
        if summary['failed_strategies']:
            print(f"  - Failed strategies: {summary['failed_strategies']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy Evaluator test failed: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis on a sample stock."""
    print("\\n=== Testing Technical Analysis ===")
    
    try:
        # Get sample data
        data = get_historical_data('RELIANCE', '6mo')
        if data.empty:
            print("✗ No data for technical analysis test")
            return False
        
        # Test strategy evaluator
        evaluator = StrategyEvaluator()
        result = evaluator.evaluate_strategies('RELIANCE', data)
        
        print(f"✓ Technical analysis completed for RELIANCE")
        print(f"  - Technical score: {result['technical_score']:.2f}")
        print(f"  - Positive signals: {result['positive_signals']}/{result['total_strategies']}")
        print(f"  - Recommendation: {result['recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Technical analysis test failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis."""
    print("\\n=== Testing Sentiment Analysis ===")
    
    try:
        # Test with a simple text first
        analyzer = SentimentAnalysis()
        
        # Test with mock news texts
        mock_news = [
            "The company reported strong quarterly earnings with 15% growth.",
            "Stock price is expected to rise due to positive market sentiment.",
            "New product launch shows promising results in market testing."
        ]
        
        sentiment_score = analyzer.analyze_sentiment(mock_news)
        print(f"✓ Sentiment analysis completed")
        print(f"  - Mock news sentiment score: {sentiment_score:.3f}")
        
        # Test full sentiment analysis (this might be slow)
        print("  - Testing full sentiment analysis (news fetching)...")
        full_score = analyzer.perform_sentiment_analysis("Reliance Industries Limited")
        print(f"  - Full sentiment score for Reliance: {full_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sentiment analysis test failed: {e}")
        return False

def test_fundamental_analysis():
    """Test fundamental analysis."""
    print("\\n=== Testing Fundamental Analysis ===")
    
    try:
        analyzer = FundamentalAnalysis()
        score = analyzer.perform_fundamental_analysis('RELIANCE')
        
        print(f"✓ Fundamental analysis completed")
        print(f"  - Fundamental score for RELIANCE: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Fundamental analysis test failed: {e}")
        return False

def test_complete_analyzer():
    """Test the complete stock analyzer."""
    print("\\n=== Testing Complete Stock Analyzer ===")
    
    try:
        with app.app_context():
            analyzer = StockAnalyzer()
            
            # Test analyzer summary
            summary = analyzer.get_analyzer_summary()
            print(f"✓ Analyzer summary retrieved")
            print(f"  - Technical strategies: {summary['technical_analysis']['total_strategies']}")
            print(f"  - Fundamental analysis: {summary['fundamental_analysis']['enabled']}")
            print(f"  - Sentiment analysis: {summary['sentiment_analysis']['enabled']}")
            
            # Test full analysis on a stock
            print("\\n  Running full analysis on RELIANCE...")
            result = analyzer.analyze_stock('RELIANCE', app.config)
            
            print(f"✓ Complete analysis finished")
            print(f"  - Symbol: {result['symbol']}")
            print(f"  - Company: {result['company_name']}")
            print(f"  - Technical score: {result['technical_score']:.3f}")
            print(f"  - Fundamental score: {result['fundamental_score']:.3f}")
            print(f"  - Sentiment score: {result['sentiment_score']:.3f}")
            print(f"  - Combined score: {result.get('combined_score', 'N/A')}")
            print(f"  - Recommended: {result['is_recommended']}")
            print(f"  - Strength: {result.get('recommendation_strength', 'N/A')}")
            print(f"  - Reason: {result['reason']}")
            
            return True
            
    except Exception as e:
        print(f"✗ Complete analyzer test failed: {e}")
        return False

def test_api_endpoints():
    """Test Flask API endpoints."""
    print("\\n=== Testing API Endpoints ===")
    
    try:
        with app.test_client() as client:
            # Test health check
            response = client.get('/')
            if response.status_code == 200:
                print("✓ Health check endpoint working")
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
            
            # Test stock analysis endpoint
            response = client.get('/analyze_stock/RELIANCE')
            if response.status_code == 200:
                data = json.loads(response.data)
                print(f"✓ Stock analysis endpoint working")
                print(f"  - Analysis result received for {data.get('symbol', 'N/A')}")
                print(f"  - Recommended: {data.get('is_recommended', 'N/A')}")
            else:
                print(f"✗ Stock analysis endpoint failed: {response.status_code}")
                return False
            
            # Test symbols endpoint
            response = client.get('/symbols')
            if response.status_code == 200:
                data = json.loads(response.data)
                print(f"✓ Symbols endpoint working - {data['count']} symbols")
            else:
                print(f"✗ Symbols endpoint failed: {response.status_code}")
                return False
            
            return True
            
    except Exception as e:
        print(f"✗ API endpoints test failed: {e}")
        return False

def main():
    """Run all system tests."""
    print("=== Share Market Analyzer Complete System Test ===")
    print("This test will verify all components of the system.")
    
    tests = [
        test_strategy_evaluator,
        test_technical_analysis,
        test_sentiment_analysis,
        test_fundamental_analysis,
        test_complete_analyzer,
        test_api_endpoints
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\\n=== Test Results ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\\n🎉 All tests passed! The Share Market Analyzer system is fully functional.")
        return 0
    else:
        print(f"\\n⚠️  {failed} test(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
