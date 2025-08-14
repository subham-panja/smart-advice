#!/usr/bin/env python3
"""
Test script to isolate which analyzer component is hanging
"""

import os
import sys
import signal
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix OpenMP/threading issues on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

def timeout_handler(signum, frame):
    raise TimeoutError("Component initialization timed out")

def test_component(component_name, import_func):
    print(f"Testing {component_name}...")
    try:
        # Set a 30-second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        start_time = time.time()
        result = import_func()
        end_time = time.time()
        
        signal.alarm(0)  # Cancel the alarm
        print(f"✓ {component_name} initialized successfully in {end_time - start_time:.2f}s")
        return result
    except TimeoutError:
        print(f"✗ {component_name} TIMED OUT after 30 seconds")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel the alarm
        print(f"✗ {component_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

print("Starting analyzer component tests...")

# Test each component individually
try:
    # 1. StrategyEvaluator
    def create_strategy_evaluator():
        from scripts.strategy_evaluator import StrategyEvaluator
        return StrategyEvaluator()
    
    strategy_evaluator = test_component("StrategyEvaluator", create_strategy_evaluator)
    
    # 2. FundamentalAnalysis
    def create_fundamental_analyzer():
        from scripts.fundamental_analysis import FundamentalAnalysis
        return FundamentalAnalysis()
    
    fundamental_analyzer = test_component("FundamentalAnalysis", create_fundamental_analyzer)
    
    # 3. SentimentAnalysis
    def create_sentiment_analyzer():
        from scripts.sentiment_analysis import SentimentAnalysis
        return SentimentAnalysis()
    
    sentiment_analyzer = test_component("SentimentAnalysis", create_sentiment_analyzer)
    
    # 4. RiskManager
    def create_risk_manager():
        from scripts.risk_management import RiskManager
        return RiskManager()
    
    risk_manager = test_component("RiskManager", create_risk_manager)
    
    # 5. SectorAnalyzer
    def create_sector_analyzer():
        from scripts.sector_analysis import SectorAnalyzer
        return SectorAnalyzer()
    
    sector_analyzer = test_component("SectorAnalyzer", create_sector_analyzer)
    
    # 6. MarketRegimeDetection
    def create_market_regime_detector():
        from scripts.market_regime_detection import MarketRegimeDetection
        return MarketRegimeDetection(symbol='DEFAULT', n_regimes=3, lookback_period='2y')
    
    market_regime_detector = test_component("MarketRegimeDetection", create_market_regime_detector)
    
    # 7. MarketMicrostructureAnalyzer
    def create_market_microstructure_analyzer():
        from scripts.market_microstructure import MarketMicrostructureAnalyzer
        return MarketMicrostructureAnalyzer()
    
    market_microstructure_analyzer = test_component("MarketMicrostructureAnalyzer", create_market_microstructure_analyzer)
    
    # 8. AlternativeDataAnalyzer
    def create_alternative_data_analyzer():
        from scripts.alternative_data_analyzer import AlternativeDataAnalyzer
        return AlternativeDataAnalyzer()
    
    alternative_data_analyzer = test_component("AlternativeDataAnalyzer", create_alternative_data_analyzer)
    
    # 9. PricePredictor
    def create_price_predictor():
        from scripts.predictor import PricePredictor
        return PricePredictor(symbol='DEFAULT')
    
    price_predictor = test_component("PricePredictor", create_price_predictor)
    
    # 10. RLTradingAgent
    def create_rl_trading_agent():
        from scripts.rl_trading_agent import RLTradingAgent
        return RLTradingAgent(symbol='DEFAULT')
    
    rl_trading_agent = test_component("RLTradingAgent", create_rl_trading_agent)
    
    # 11. TransactionCostAnalyzer
    def create_tca_analyzer():
        from scripts.tca_analysis import TransactionCostAnalyzer
        return TransactionCostAnalyzer()
    
    tca_analyzer = test_component("TransactionCostAnalyzer", create_tca_analyzer)
    
    print("\nAll component tests completed!")
    
except KeyboardInterrupt:
    print("\nTest interrupted by user")
    sys.exit(1)
except Exception as e:
    print(f"Test failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
