#!/usr/bin/env python3
# ruff: noqa: F841
"""
Test script to isolate which component hangs during StockAnalyzer initialization.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging

logger = setup_logging(verbose=True)


def test_components():
    """Test each component of StockAnalyzer individually."""

    print("Starting component tests...")

    # Test StrategyEvaluator
    try:
        logger.info("Testing StrategyEvaluator...")
        from scripts.strategy_evaluator import StrategyEvaluator

        _ = StrategyEvaluator()

        logger.info("✓ StrategyEvaluator initialized successfully")
    except Exception as e:
        logger.error(f"✗ StrategyEvaluator failed: {e}")
        return False

    # Test FundamentalAnalysis
    try:
        logger.info("Testing FundamentalAnalysis...")
        from scripts.fundamental_analysis import FundamentalAnalysis

        fundamental_analyzer = FundamentalAnalysis()
        logger.info("✓ FundamentalAnalysis initialized successfully")
    except Exception as e:
        logger.error(f"✗ FundamentalAnalysis failed: {e}")
        return False

    # Test SentimentAnalysis
    try:
        logger.info("Testing SentimentAnalysis...")
        from scripts.sentiment_analysis import SentimentAnalysis

        sentiment_analyzer = SentimentAnalysis()
        logger.info("✓ SentimentAnalysis initialized successfully")
    except Exception as e:
        logger.error(f"✗ SentimentAnalysis failed: {e}")
        return False

    # Test RiskManager
    try:
        logger.info("Testing RiskManager...")
        from scripts.risk_management import RiskManager

        risk_manager = RiskManager()
        logger.info("✓ RiskManager initialized successfully")
    except Exception as e:
        logger.error(f"✗ RiskManager failed: {e}")
        return False

    # Test SectorAnalyzer
    try:
        logger.info("Testing SectorAnalyzer...")
        from scripts.sector_analysis import SectorAnalyzer

        sector_analyzer = SectorAnalyzer()
        logger.info("✓ SectorAnalyzer initialized successfully")
    except Exception as e:
        logger.error(f"✗ SectorAnalyzer failed: {e}")
        return False

    # Test MarketRegimeDetection
    try:
        logger.info("Testing MarketRegimeDetection...")
        from scripts.market_regime_detection import MarketRegimeDetection

        market_regime_detector = MarketRegimeDetection(symbol="DEFAULT", n_regimes=3, lookback_period="2y")
        logger.info("✓ MarketRegimeDetection initialized successfully")
    except Exception as e:
        logger.error(f"✗ MarketRegimeDetection failed: {e}")
        return False

    # Test MarketMicrostructureAnalyzer
    try:
        logger.info("Testing MarketMicrostructureAnalyzer...")
        from scripts.market_microstructure import MarketMicrostructureAnalyzer

        market_microstructure_analyzer = MarketMicrostructureAnalyzer()
        logger.info("✓ MarketMicrostructureAnalyzer initialized successfully")
    except Exception as e:
        logger.error(f"✗ MarketMicrostructureAnalyzer failed: {e}")
        return False

    logger.info("All components initialized successfully!")
    return True


if __name__ == "__main__":
    if test_components():
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)
