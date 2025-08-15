#!/usr/bin/env python3
"""
ML Secondary Ranker
==================

Uses ML model as secondary ranker for swing trading recommendations.
Does not override rule-based hard gates but provides additional ranking.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.classifier_trainer import SwingTradingClassifier
from ml.feature_extractor import MLFeatureExtractor
from utils.logger import setup_logging

logger = setup_logging()

class MLSecondaryRanker:
    """
    ML-based secondary ranking system for swing trading recommendations.
    Enhances rule-based recommendations without overriding hard gates.
    """
    
    def __init__(self, model_dir: str = "ml/models", target_horizon: int = 10):
        """
        Initialize the ML secondary ranker
        
        Args:
            model_dir: Directory containing trained ML models
            target_horizon: Prediction horizon in days
        """
        self.model_dir = model_dir
        self.target_horizon = target_horizon
        self.classifier = SwingTradingClassifier(target_horizon=target_horizon)
        self.feature_extractor = MLFeatureExtractor()
        self.is_loaded = False
        
        # Ranking configuration
        self.ranking_config = {
            'ml_weight': 0.3,  # Weight of ML score in final ranking
            'rule_weight': 0.7,  # Weight of rule-based score
            'min_probability_threshold': 0.4,  # Minimum ML probability for positive ranking
            'confidence_boost_threshold': 0.7,  # Threshold for confidence boost
            'max_ml_boost': 0.15,  # Maximum boost from ML score
            'penalty_threshold': 0.3  # Below this, apply penalty
        }
        
        # Load models if available
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load trained ML models"""
        try:
            if not os.path.exists(self.model_dir):
                logger.warning(f"ML model directory not found: {self.model_dir}")
                return False
            
            self.classifier.load_models(self.model_dir)
            self.is_loaded = True
            logger.info("ML models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
            self.is_loaded = False
            return False
    
    def enhance_recommendations(self, recommendations: List[Dict[str, Any]], 
                              price_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Enhance recommendations with ML-based secondary ranking
        
        Args:
            recommendations: List of rule-based recommendations
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            
        Returns:
            Enhanced recommendations with ML scoring and ranking
        """
        logger.info(f"Enhancing {len(recommendations)} recommendations with ML ranking")
        
        if not self.is_loaded:
            logger.warning("ML models not loaded, returning original recommendations")
            return self._add_ml_placeholder(recommendations)
        
        enhanced_recommendations = []
        
        for rec in recommendations:
            try:
                symbol = rec.get('symbol')
                if not symbol or symbol not in price_data:
                    logger.warning(f"No price data for symbol: {symbol}")
                    enhanced_rec = rec.copy()
                    enhanced_rec.update(self._get_default_ml_scores())
                    enhanced_recommendations.append(enhanced_rec)
                    continue
                
                # Get ML prediction
                ml_scores = self._get_ml_prediction(symbol, price_data[symbol])
                
                # Enhance recommendation
                enhanced_rec = self._enhance_single_recommendation(rec, ml_scores)
                enhanced_recommendations.append(enhanced_rec)
                
            except Exception as e:
                logger.error(f"Error enhancing recommendation for {rec.get('symbol', 'Unknown')}: {e}")
                enhanced_rec = rec.copy()
                enhanced_rec.update(self._get_default_ml_scores())
                enhanced_recommendations.append(enhanced_rec)
        
        # Re-rank recommendations
        enhanced_recommendations = self._rerank_recommendations(enhanced_recommendations)
        
        logger.info("ML enhancement completed")
        return enhanced_recommendations
    
    def _get_ml_prediction(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Get ML prediction for a single symbol"""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(df, symbol)
            
            if features.empty:
                logger.warning(f"No features extracted for {symbol}")
                return self._get_default_ml_scores()
            
            # Get latest features (most recent day)
            latest_features = features.iloc[-1:].copy()
            
            # Make prediction
            predictions = self.classifier.predict(latest_features)
            
            probability = predictions['probabilities'][0]
            prediction = predictions['predictions'][0]
            model_used = predictions['model_used']
            
            # Calculate confidence metrics
            confidence = self._calculate_confidence(probability)
            
            return {
                'ml_probability': float(probability),
                'ml_prediction': int(prediction),
                'ml_confidence': confidence,
                'ml_model_used': model_used,
                'ml_score': self._probability_to_score(probability),
                'features_count': len(latest_features.columns)
            }
            
        except Exception as e:
            logger.error(f"Error getting ML prediction for {symbol}: {e}")
            return self._get_default_ml_scores()
    
    def _probability_to_score(self, probability: float) -> float:
        """Convert ML probability to score (-1 to 1 scale)"""
        # Transform probability (0-1) to score (-1 to 1)
        # 0.5 probability = 0 score, 1.0 probability = 1 score, 0.0 probability = -1 score
        return (probability - 0.5) * 2
    
    def _calculate_confidence(self, probability: float) -> float:
        """Calculate confidence based on probability distance from 0.5"""
        # Confidence is how far the probability is from 0.5 (neutral)
        return abs(probability - 0.5) * 2
    
    def _get_default_ml_scores(self) -> Dict[str, Any]:
        """Get default ML scores when prediction fails"""
        return {
            'ml_probability': 0.5,
            'ml_prediction': 0,
            'ml_confidence': 0.0,
            'ml_model_used': 'none',
            'ml_score': 0.0,
            'features_count': 0
        }
    
    def _enhance_single_recommendation(self, rec: Dict[str, Any], ml_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single recommendation with ML scores"""
        enhanced_rec = rec.copy()
        
        # Add ML scores
        enhanced_rec.update(ml_scores)
        
        # Calculate combined score
        rule_based_score = rec.get('combined_score', 0)
        ml_score = ml_scores['ml_score']
        ml_confidence = ml_scores['ml_confidence']
        
        # Apply ML weighting
        combined_score = self._calculate_combined_score(rule_based_score, ml_score, ml_confidence)
        enhanced_rec['ml_enhanced_score'] = combined_score
        
        # Update recommendation strength if ML provides strong signal
        enhanced_rec = self._update_recommendation_strength(enhanced_rec, ml_scores)
        
        # Add ML reasoning
        enhanced_rec['ml_reasoning'] = self._generate_ml_reasoning(ml_scores)
        
        return enhanced_rec
    
    def _calculate_combined_score(self, rule_score: float, ml_score: float, ml_confidence: float) -> float:
        """Calculate combined score from rule-based and ML components"""
        try:
            # Base combination using configured weights
            base_combined = (rule_score * self.ranking_config['rule_weight'] + 
                           ml_score * self.ranking_config['ml_weight'])
            
            # Apply confidence-based adjustments
            if ml_confidence > self.ranking_config['confidence_boost_threshold']:
                # High confidence ML prediction gets a boost
                confidence_boost = (ml_confidence - self.ranking_config['confidence_boost_threshold']) * self.ranking_config['max_ml_boost']
                if ml_score > 0:  # Only boost positive ML scores
                    base_combined += confidence_boost
            
            # Apply penalty for very low ML probability
            ml_probability = (ml_score + 1) / 2  # Convert back to 0-1 scale
            if ml_probability < self.ranking_config['penalty_threshold']:
                penalty = (self.ranking_config['penalty_threshold'] - ml_probability) * 0.1
                base_combined -= penalty
            
            return base_combined
            
        except Exception as e:
            logger.error(f"Error calculating combined score: {e}")
            return rule_score
    
    def _update_recommendation_strength(self, rec: Dict[str, Any], ml_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Update recommendation strength based on ML signals"""
        try:
            current_strength = rec.get('recommendation_strength', 'HOLD')
            ml_probability = ml_scores['ml_probability']
            ml_confidence = ml_scores['ml_confidence']
            
            # Don't override STRONG_BUY or downgrade recommendations
            if current_strength == 'STRONG_BUY':
                return rec
            
            # Upgrade BUY to STRONG_BUY if ML is very confident and positive
            if (current_strength == 'BUY' and 
                ml_probability > 0.8 and 
                ml_confidence > 0.6):
                rec['recommendation_strength'] = 'STRONG_BUY'
                rec['ml_upgrade_reason'] = f"ML model very confident (p={ml_probability:.3f}, conf={ml_confidence:.3f})"
            
            # Add ML confidence flag
            if ml_confidence > 0.7:
                rec['ml_high_confidence'] = True
            else:
                rec['ml_high_confidence'] = False
            
            return rec
            
        except Exception as e:
            logger.error(f"Error updating recommendation strength: {e}")
            return rec
    
    def _generate_ml_reasoning(self, ml_scores: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for ML prediction"""
        try:
            probability = ml_scores['ml_probability']
            confidence = ml_scores['ml_confidence']
            model_used = ml_scores['ml_model_used']
            
            if probability > 0.7:
                strength = "high"
            elif probability > 0.6:
                strength = "moderate"
            elif probability > 0.4:
                strength = "weak"
            else:
                strength = "negative"
            
            confidence_desc = "high" if confidence > 0.6 else "moderate" if confidence > 0.3 else "low"
            
            return f"ML model ({model_used}) shows {strength} positive signal (p={probability:.3f}) with {confidence_desc} confidence"
            
        except Exception as e:
            logger.error(f"Error generating ML reasoning: {e}")
            return "ML analysis unavailable"
    
    def _rerank_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-rank recommendations based on ML-enhanced scores"""
        try:
            # Sort by ML-enhanced score (descending)
            sorted_recommendations = sorted(
                recommendations,
                key=lambda x: x.get('ml_enhanced_score', x.get('combined_score', 0)),
                reverse=True
            )
            
            # Add ranking information
            for i, rec in enumerate(sorted_recommendations):
                rec['ml_ranking'] = i + 1
                rec['ml_rank_change'] = self._calculate_rank_change(rec, recommendations, i)
            
            return sorted_recommendations
            
        except Exception as e:
            logger.error(f"Error re-ranking recommendations: {e}")
            return recommendations
    
    def _calculate_rank_change(self, rec: Dict[str, Any], original_list: List[Dict[str, Any]], new_rank: int) -> int:
        """Calculate how much the ranking changed due to ML enhancement"""
        try:
            symbol = rec.get('symbol')
            
            # Find original position
            original_rank = None
            for i, orig_rec in enumerate(original_list):
                if orig_rec.get('symbol') == symbol:
                    original_rank = i
                    break
            
            if original_rank is not None:
                return original_rank - new_rank  # Positive means moved up
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating rank change: {e}")
            return 0
    
    def _add_ml_placeholder(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add placeholder ML fields when models are not available"""
        enhanced_recommendations = []
        
        for rec in recommendations:
            enhanced_rec = rec.copy()
            enhanced_rec.update(self._get_default_ml_scores())
            enhanced_rec['ml_enhanced_score'] = rec.get('combined_score', 0)
            enhanced_rec['ml_reasoning'] = "ML models not available"
            enhanced_rec['ml_ranking'] = 0
            enhanced_rec['ml_rank_change'] = 0
            enhanced_rec['ml_high_confidence'] = False
            enhanced_recommendations.append(enhanced_rec)
        
        return enhanced_recommendations
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of ML models"""
        return {
            'is_loaded': self.is_loaded,
            'model_dir': self.model_dir,
            'target_horizon': self.target_horizon,
            'available_models': list(self.classifier.models.keys()) if self.is_loaded else [],
            'feature_count': len(self.classifier.feature_names) if self.is_loaded else 0,
            'training_metrics': self.classifier.training_metrics if self.is_loaded else {}
        }
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """Update ranking configuration"""
        self.ranking_config.update(new_config)
        logger.info(f"Updated ML ranking configuration: {new_config}")
    
    def analyze_ml_performance(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze ML performance on recommendations"""
        try:
            if not recommendations:
                return {'error': 'No recommendations to analyze'}
            
            ml_probabilities = [rec.get('ml_probability', 0.5) for rec in recommendations]
            ml_confidences = [rec.get('ml_confidence', 0.0) for rec in recommendations]
            ml_enhanced_scores = [rec.get('ml_enhanced_score', 0) for rec in recommendations]
            original_scores = [rec.get('combined_score', 0) for rec in recommendations]
            
            analysis = {
                'total_recommendations': len(recommendations),
                'ml_model_loaded': self.is_loaded,
                'probability_stats': {
                    'mean': np.mean(ml_probabilities),
                    'std': np.std(ml_probabilities),
                    'min': np.min(ml_probabilities),
                    'max': np.max(ml_probabilities)
                },
                'confidence_stats': {
                    'mean': np.mean(ml_confidences),
                    'high_confidence_count': sum(1 for c in ml_confidences if c > 0.6),
                    'low_confidence_count': sum(1 for c in ml_confidences if c < 0.3)
                },
                'ranking_impact': {
                    'score_changes': [ml_enhanced_scores[i] - original_scores[i] 
                                    for i in range(len(ml_enhanced_scores))],
                    'upgrades': sum(1 for rec in recommendations if rec.get('ml_upgrade_reason')),
                    'avg_rank_change': np.mean([abs(rec.get('ml_rank_change', 0)) for rec in recommendations])
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing ML performance: {e}")
            return {'error': str(e)}


def main():
    """Main function for testing the ML secondary ranker"""
    logger.info("Testing ML Secondary Ranker")
    
    # Create sample recommendations
    sample_recommendations = [
        {
            'symbol': 'STOCK_A',
            'combined_score': 0.45,
            'recommendation_strength': 'BUY',
            'technical_score': 0.6,
            'fundamental_score': 0.3
        },
        {
            'symbol': 'STOCK_B',
            'combined_score': 0.38,
            'recommendation_strength': 'BUY',
            'technical_score': 0.5,
            'fundamental_score': 0.4
        },
        {
            'symbol': 'STOCK_C',
            'combined_score': 0.42,
            'recommendation_strength': 'BUY',
            'technical_score': 0.55,
            'fundamental_score': 0.35
        }
    ]
    
    # Create sample price data
    np.random.seed(42)
    price_data = {}
    
    for i, symbol in enumerate(['STOCK_A', 'STOCK_B', 'STOCK_C']):
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        base_price = 100 + i * 10
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        df = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices[:-1]],
            'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices[:-1]],
            'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices[:-1]],
            'Close': prices[:-1],
            'Volume': np.random.randint(10000, 1000000, 100)
        }, index=dates)
        
        price_data[symbol] = df
    
    # Initialize ranker
    ranker = MLSecondaryRanker(model_dir="ml/models", target_horizon=10)
    
    # Get model status
    status = ranker.get_model_status()
    print("=== ML Model Status ===")
    print(f"Models loaded: {status['is_loaded']}")
    print(f"Available models: {status['available_models']}")
    print(f"Feature count: {status['feature_count']}")
    
    # Enhance recommendations
    enhanced_recs = ranker.enhance_recommendations(sample_recommendations, price_data)
    
    print("\n=== Enhanced Recommendations ===")
    for i, rec in enumerate(enhanced_recs):
        print(f"{i+1}. {rec['symbol']}:")
        print(f"   Original Score: {rec.get('combined_score', 0):.3f}")
        print(f"   ML Enhanced Score: {rec.get('ml_enhanced_score', 0):.3f}")
        print(f"   ML Probability: {rec.get('ml_probability', 0):.3f}")
        print(f"   ML Confidence: {rec.get('ml_confidence', 0):.3f}")
        print(f"   Recommendation: {rec.get('recommendation_strength', 'N/A')}")
        print(f"   ML Reasoning: {rec.get('ml_reasoning', 'N/A')}")
        print(f"   Rank Change: {rec.get('ml_rank_change', 0):+d}")
        print()
    
    # Analyze performance
    analysis = ranker.analyze_ml_performance(enhanced_recs)
    print("=== ML Performance Analysis ===")
    print(f"Total recommendations: {analysis.get('total_recommendations', 0)}")
    if 'probability_stats' in analysis:
        prob_stats = analysis['probability_stats']
        print(f"Probability - Mean: {prob_stats['mean']:.3f}, Range: [{prob_stats['min']:.3f}, {prob_stats['max']:.3f}]")
    if 'confidence_stats' in analysis:
        conf_stats = analysis['confidence_stats']
        print(f"High confidence predictions: {conf_stats.get('high_confidence_count', 0)}")
    if 'ranking_impact' in analysis:
        rank_stats = analysis['ranking_impact']
        print(f"Upgrades: {rank_stats.get('upgrades', 0)}")
        print(f"Average rank change: {rank_stats.get('avg_rank_change', 0):.1f}")


if __name__ == "__main__":
    main()
