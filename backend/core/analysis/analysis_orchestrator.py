"""
Analysis Orchestrator Module
============================

Coordinates various analyses and handles the overall workflow.
"""

import sys
import os

# Add parent directories to path to enable imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .technical_analyzer import TechnicalAnalyzer
from .fundamental_analyzer import FundamentalAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from .recommendation_engine import RecommendationEngine
from scripts.sector_analysis import SectorAnalyzer
from utils.logger import setup_logging

logger = setup_logging()

class AnalysisOrchestrator:
    """
    Orchestrates the entire analysis pipeline.
    """

    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sector_analyzer = SectorAnalyzer()
        self.recommendation_engine = RecommendationEngine()

    def perform_analysis(self, symbol: str):
        """
        Perform full stock analysis.
        """
        try:
            logger.info(f"Starting analysis for {symbol}")
            # Perform all analyses
            technical_result = self.technical_analyzer.calculate_technical_indicators(symbol)
            fundamental_result = self.fundamental_analyzer.perform_fundamental_analysis(symbol)
            sentiment_result = self.sentiment_analyzer.perform_sentiment_analysis(symbol)
            
            # Perform sector analysis
            sector_result = self.sector_analyzer.get_comprehensive_sector_analysis(symbol)

            # Extract fundamental score and details
            if isinstance(fundamental_result, dict):
                fundamental_score = fundamental_result.get('score', 0.1)
                fundamental_details = fundamental_result.get('details', {})
            else:
                # Backward compatibility
                fundamental_score = fundamental_result
                fundamental_details = {}

            combined_result = self.recommendation_engine.combine_analysis_results({
                'technical_score': technical_result,
                'fundamental_score': fundamental_score,
                'fundamental_details': fundamental_details,
                'sentiment_score': sentiment_result,
                'sector_analysis': sector_result
            })
            logger.info("Analysis complete")
            return combined_result
        except Exception as e:
            logger.error(f"Error in analysis: {e}")

