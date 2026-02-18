
import sys
import os
import logging
from pprint import pprint

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analysis.recommendation_engine import RecommendationEngine
from config import RECOMMENDATION_THRESHOLDS, ANALYSIS_WEIGHTS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_recommendation_logic():
    engine = RecommendationEngine()
    
    print("\n=== Current Configuration ===")
    print("THRESHOLDS:")
    pprint(RECOMMENDATION_THRESHOLDS)
    print("\nWEIGHTS:")
    pprint(ANALYSIS_WEIGHTS)
    
    # Test Case 1: Weak Stock (Should be HOLD)
    print("\n=== Test Case 1: Weak Stock ===")
    weak_stock = {
        'symbol': 'WEAK',
        'technical_score': -0.2,
        'fundamental_score': -0.1,
        'sentiment_score': 0.0,
        'backtest': {'combined_metrics': {'avg_cagr': -0.05}},
        'volume_analysis': {'overall_signal': 'bearish', 'confidence': 0.8},
        'sector_analysis': {'sector_score': -0.5, 'sector': 'Technology'},
        'gates_passed': {'trend': False, 'volatility': True}
    }
    
    result_weak = engine.combine_analysis_results(weak_stock.copy(), consider_backtest=True, keep_reason_as_list=True)
    print(f"Result: {result_weak.get('recommendation_strength')} (Recommended: {result_weak.get('is_recommended')})")
    print(f"Combined Score: {result_weak.get('combined_score')}")
    print(f"Reasons: {result_weak.get('reason')}")

    # Test Case 2: Strong Stock (Should be BUY/STRONG_BUY)
    print("\n=== Test Case 2: Strong Stock ===")
    strong_stock = {
        'symbol': 'STRONG',
        'technical_score': 0.6,
        'fundamental_score': 0.4,
        'sentiment_score': 0.2,
        'backtest': {'combined_metrics': {'avg_cagr': 0.15}},
        'volume_analysis': {'overall_signal': 'bullish', 'confidence': 0.7},
        'sector_analysis': {'sector_score': 0.2, 'sector': 'Technology'},
        'gates_passed': {'trend': True, 'volatility': True}
    }
    
    result_strong = engine.combine_analysis_results(strong_stock.copy(), consider_backtest=True, keep_reason_as_list=True)
    print(f"Result: {result_strong.get('recommendation_strength')} (Recommended: {result_strong.get('is_recommended')})")
    print(f"Combined Score: {result_strong.get('combined_score')}")
    print(f"Reasons: {result_strong.get('reason')}")

    # Test Case 3: Borderline Stock (Should be BUY if fundamentals good, else HOLD)
    print("\n=== Test Case 3: Borderline Stock ===")
    borderline_stock = {
        'symbol': 'BORDER',
        'technical_score': 0.15, # Slightly below buy threshold usually
        'fundamental_score': 0.3,
        'sentiment_score': 0.1,
        'backtest': {'combined_metrics': {'avg_cagr': 0.05}},
        'volume_analysis': {'overall_signal': 'neutral', 'confidence': 0.5},
        'sector_analysis': {'sector_score': 0.1, 'sector': 'Technology'},
        'gates_passed': {'trend': True, 'volatility': True},
        'fundamental_details': {
            'eps_growth': 0.15,
            'de_ratio': 0.4,
            'profit_margins': 0.12,
            'roe': 0.18
        }
    }
    
    result_borderline = engine.combine_analysis_results(borderline_stock.copy(), consider_backtest=True, keep_reason_as_list=True)
    print(f"Result: {result_borderline.get('recommendation_strength')} (Recommended: {result_borderline.get('is_recommended')})")
    print(f"Combined Score: {result_borderline.get('combined_score')}")
    print(f"Reasons: {result_borderline.get('reason')}")

if __name__ == "__main__":
    test_recommendation_logic()
