"""
Analysis Orchestrator Module
============================

Coordinates various analyses and handles the overall workflow.
"""

from core.analysis.technical_analyzer import TechnicalAnalyzer
from core.analysis.fundamental_analyzer import FundamentalAnalyzer
from core.analysis.sentiment_analyzer import SentimentAnalyzer
from core.analysis.recommendation_engine import RecommendationEngine
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

            combined_result = self.recommendation_engine.combine_analysis_results({
                'technical_score': technical_result,
                'fundamental_score': fundamental_result,
                'sentiment_score': sentiment_result
            })
            logger.info("Analysis complete")
            return combined_result
        except Exception as e:
            logger.error(f"Error in analysis: {e}")

