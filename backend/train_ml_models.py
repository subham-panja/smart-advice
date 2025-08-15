#!/usr/bin/env python3
"""
ML Models Training Script
========================

Comprehensive script to train and deploy ML models for swing trading.
Integrates feature extraction, model training, and secondary ranking.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml.feature_extractor import MLFeatureExtractor
from ml.classifier_trainer import SwingTradingClassifier
from ml.secondary_ranker import MLSecondaryRanker
from database import DatabaseManager
from utils.logger import setup_logging

logger = setup_logging()

class MLPipeline:
    """
    Complete ML pipeline for swing trading models
    """
    
    def __init__(self, target_horizon: int = 10):
        """Initialize the ML pipeline"""
        self.target_horizon = target_horizon
        self.feature_extractor = MLFeatureExtractor()
        self.classifier = SwingTradingClassifier(target_horizon=target_horizon)
        self.db_manager = DatabaseManager()
        
        # Pipeline configuration
        self.config = {
            'train_test_split_date': '2023-06-01',  # Split date for train/test
            'min_samples_per_symbol': 100,  # Minimum samples required per symbol
            'max_symbols_for_training': 50,  # Maximum symbols to use for training
            'model_save_dir': 'ml/models',
            'feature_save_path': 'ml/features/training_features.parquet',
            'validation_split': 0.2  # Validation split ratio
        }
    
    def fetch_training_data(self) -> List[Tuple[pd.DataFrame, str]]:
        """
        Fetch training data from database or generate sample data
        
        Returns:
            List of (DataFrame, symbol) tuples
        """
        logger.info("Fetching training data...")
        
        try:
            # Try to get real data from database first
            data_sources = self._fetch_from_database()
            
            if not data_sources:
                logger.warning("No data from database, generating sample data")
                data_sources = self._generate_sample_data()
            
            logger.info(f"Loaded data for {len(data_sources)} symbols")
            return data_sources
            
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            logger.info("Falling back to sample data generation")
            return self._generate_sample_data()
    
    def _fetch_from_database(self) -> List[Tuple[pd.DataFrame, str]]:
        """Fetch actual market data from database"""
        try:
            # This is a placeholder - in a real implementation, you would
            # fetch historical price data from your database
            
            # For now, return empty list to fall back to sample data
            return []
            
        except Exception as e:
            logger.error(f"Error fetching from database: {e}")
            return []
    
    def _generate_sample_data(self, n_symbols: int = 10, n_days: int = 1000) -> List[Tuple[pd.DataFrame, str]]:
        """Generate realistic sample data for training"""
        logger.info(f"Generating sample data for {n_symbols} symbols")
        
        np.random.seed(42)
        data_sources = []
        
        for i in range(n_symbols):
            symbol = f"TRAINING_STOCK_{i+1:02d}"
            dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
            
            # Generate realistic price series with different characteristics
            base_price = 50 + np.random.uniform(10, 200)  # Random base price
            volatility = np.random.uniform(0.015, 0.035)  # Random volatility
            drift = np.random.uniform(-0.0005, 0.002)  # Random drift
            
            returns = np.random.normal(drift, volatility, n_days)
            
            # Add some market regime changes
            regime_changes = np.random.choice(n_days, size=3, replace=False)
            for change_point in regime_changes:
                regime_vol = np.random.uniform(0.02, 0.05)
                regime_drift = np.random.uniform(-0.003, 0.003)
                length = min(100, n_days - change_point)
                returns[change_point:change_point+length] = np.random.normal(regime_drift, regime_vol, length)
            
            # Generate prices
            prices = [base_price]
            for r in returns:
                prices.append(max(1.0, prices[-1] * (1 + r)))  # Ensure price stays positive
            
            # Create OHLCV data
            df = pd.DataFrame({
                'Open': [p * (1 + np.random.normal(0, 0.002)) for p in prices[:-1]],
                'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
                'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
                'Close': prices[:-1],
                'Volume': np.random.randint(10000, 2000000, n_days)
            }, index=dates)
            
            # Ensure realistic OHLC relationships
            for idx in df.index:
                o, h, l, c = df.loc[idx, ['Open', 'High', 'Low', 'Close']]
                df.loc[idx, 'High'] = max(o, h, l, c)
                df.loc[idx, 'Low'] = min(o, h, l, c)
            
            data_sources.append((df, symbol))
        
        return data_sources
    
    def extract_and_save_features(self, data_sources: List[Tuple[pd.DataFrame, str]]) -> pd.DataFrame:
        """
        Extract features from all data sources and save for future use
        
        Args:
            data_sources: List of (DataFrame, symbol) tuples
            
        Returns:
            Combined features DataFrame
        """
        logger.info("Extracting features from all data sources...")
        
        all_features = []
        
        for df, symbol in data_sources:
            try:
                logger.info(f"Extracting features for {symbol}")
                features = self.feature_extractor.extract_features(df, symbol)
                
                if not features.empty:
                    # Add metadata
                    features['symbol'] = symbol
                    features['extraction_date'] = datetime.now()
                    all_features.append(features)
                    logger.info(f"Extracted {len(features)} samples for {symbol}")
                else:
                    logger.warning(f"No features extracted for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error extracting features for {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No features extracted from any symbol")
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined features: {len(combined_features)} samples, {len(combined_features.columns)} columns")
        
        # Save features for future use
        os.makedirs(os.path.dirname(self.config['feature_save_path']), exist_ok=True)
        combined_features.to_parquet(self.config['feature_save_path'])
        logger.info(f"Features saved to {self.config['feature_save_path']}")
        
        return combined_features
    
    def train_models(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML models with proper train/validation split
        
        Args:
            features: Combined features DataFrame
            
        Returns:
            Training results
        """
        logger.info("Training ML models...")
        
        # Prepare training data
        train_features, train_targets = self.classifier.prepare_training_data(
            [(features, 'combined_training_data')]
        )
        
        # Time-based split for validation
        split_date = pd.to_datetime(self.config['train_test_split_date'])
        
        if 'extraction_date' in train_features.columns:
            train_mask = train_features['extraction_date'] < split_date
            val_mask = train_features['extraction_date'] >= split_date
            
            X_train = train_features[train_mask]
            y_train = train_targets[train_mask]
            X_val = train_features[val_mask]
            y_val = train_targets[val_mask]
            
            logger.info(f"Train set: {len(X_train)} samples")
            logger.info(f"Validation set: {len(X_val)} samples")
        else:
            # Fallback to simple split
            split_idx = int(len(train_features) * (1 - self.config['validation_split']))
            X_train = train_features.iloc[:split_idx]
            y_train = train_targets.iloc[:split_idx]
            X_val = train_features.iloc[split_idx:]
            y_val = train_targets.iloc[split_idx:]
        
        # Train models
        training_results = self.classifier.train_models(X_train, y_train)
        
        # Validate models
        if len(X_val) > 0:
            validation_results = self.classifier.evaluate_model_performance(X_val, y_val)
            training_results['validation_results'] = validation_results
        
        # Save trained models
        os.makedirs(self.config['model_save_dir'], exist_ok=True)
        self.classifier.save_models(self.config['model_save_dir'])
        
        return training_results
    
    def evaluate_pipeline(self) -> Dict[str, Any]:
        """
        Evaluate the complete ML pipeline
        
        Returns:
            Evaluation results
        """
        logger.info("Evaluating ML pipeline...")
        
        try:
            # Initialize secondary ranker
            ranker = MLSecondaryRanker(
                model_dir=self.config['model_save_dir'],
                target_horizon=self.target_horizon
            )
            
            # Get model status
            model_status = ranker.get_model_status()
            
            # Create sample recommendations for testing
            sample_recs = [
                {
                    'symbol': 'TEST_STOCK_1',
                    'combined_score': 0.45,
                    'recommendation_strength': 'BUY',
                    'technical_score': 0.6,
                    'fundamental_score': 0.3
                },
                {
                    'symbol': 'TEST_STOCK_2',
                    'combined_score': 0.38,
                    'recommendation_strength': 'BUY',
                    'technical_score': 0.5,
                    'fundamental_score': 0.4
                }
            ]
            
            # Create sample price data
            sample_price_data = {}
            for i, symbol in enumerate(['TEST_STOCK_1', 'TEST_STOCK_2']):
                dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
                base_price = 100 + i * 20
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
                
                sample_price_data[symbol] = df
            
            # Test enhancement
            enhanced_recs = ranker.enhance_recommendations(sample_recs, sample_price_data)
            
            # Analyze performance
            performance_analysis = ranker.analyze_ml_performance(enhanced_recs)
            
            return {
                'model_status': model_status,
                'enhanced_recommendations': enhanced_recs,
                'performance_analysis': performance_analysis,
                'pipeline_status': 'SUCCESS'
            }
            
        except Exception as e:
            logger.error(f"Error evaluating pipeline: {e}")
            return {
                'pipeline_status': 'ERROR',
                'error': str(e)
            }
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete ML pipeline from data fetching to model deployment
        
        Returns:
            Results from each stage
        """
        logger.info("Starting complete ML pipeline...")
        
        try:
            # Stage 1: Fetch training data
            logger.info("Stage 1: Fetching training data")
            data_sources = self.fetch_training_data()
            
            # Stage 2: Extract features
            logger.info("Stage 2: Extracting features")
            features = self.extract_and_save_features(data_sources)
            
            # Stage 3: Train models
            logger.info("Stage 3: Training models")
            training_results = self.train_models(features)
            
            # Stage 4: Evaluate pipeline
            logger.info("Stage 4: Evaluating pipeline")
            evaluation_results = self.evaluate_pipeline()
            
            # Combine results
            pipeline_results = {
                'stage_1_data_fetch': {
                    'symbols_loaded': len(data_sources),
                    'status': 'SUCCESS'
                },
                'stage_2_feature_extraction': {
                    'total_features': len(features.columns),
                    'total_samples': len(features),
                    'features_saved_to': self.config['feature_save_path'],
                    'status': 'SUCCESS'
                },
                'stage_3_model_training': {
                    'training_results': training_results,
                    'models_saved_to': self.config['model_save_dir'],
                    'status': 'SUCCESS'
                },
                'stage_4_evaluation': evaluation_results,
                'overall_status': 'SUCCESS',
                'completed_at': datetime.now().isoformat()
            }
            
            logger.info("Complete ML pipeline finished successfully!")
            return pipeline_results
            
        except Exception as e:
            logger.error(f"ML pipeline failed: {e}")
            return {
                'overall_status': 'FAILED',
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Train ML models for swing trading')
    parser.add_argument('--target-horizon', type=int, default=10, 
                        help='Target prediction horizon in days (default: 10)')
    parser.add_argument('--n-symbols', type=int, default=10,
                        help='Number of symbols for sample data (default: 10)')
    parser.add_argument('--n-days', type=int, default=1000,
                        help='Number of days for sample data (default: 1000)')
    parser.add_argument('--stage', type=str, choices=['all', 'train', 'evaluate'], default='all',
                        help='Pipeline stage to run (default: all)')
    
    args = parser.parse_args()
    
    print("=== ML Pipeline for Swing Trading ===")
    print(f"Target Horizon: {args.target_horizon} days")
    print(f"Stage: {args.stage}")
    print()
    
    # Initialize pipeline
    pipeline = MLPipeline(target_horizon=args.target_horizon)
    
    if args.stage == 'all':
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("=== Pipeline Results ===")
        print(f"Overall Status: {results.get('overall_status', 'UNKNOWN')}")
        
        if results.get('overall_status') == 'SUCCESS':
            stage_1 = results.get('stage_1_data_fetch', {})
            stage_2 = results.get('stage_2_feature_extraction', {})
            stage_3 = results.get('stage_3_model_training', {})
            stage_4 = results.get('stage_4_evaluation', {})
            
            print(f"✓ Data Loading: {stage_1.get('symbols_loaded', 0)} symbols")
            print(f"✓ Feature Extraction: {stage_2.get('total_features', 0)} features, {stage_2.get('total_samples', 0)} samples")
            print(f"✓ Model Training: Models saved to {stage_3.get('models_saved_to', 'N/A')}")
            print(f"✓ Pipeline Evaluation: {stage_4.get('pipeline_status', 'N/A')}")
            
            # Show training results summary
            training_results = stage_3.get('training_results', {})
            if training_results:
                print("\n=== Model Performance Summary ===")
                for model_name, metrics in training_results.items():
                    if isinstance(metrics, dict) and 'cv_auc_mean' in metrics:
                        print(f"{model_name}: CV AUC {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
            
            # Show evaluation summary
            model_status = stage_4.get('model_status', {})
            if model_status:
                print(f"\nML Models Status: {'✓ Loaded' if model_status.get('is_loaded') else '✗ Not loaded'}")
                print(f"Available Models: {model_status.get('available_models', [])}")
        
        else:
            print(f"✗ Pipeline failed: {results.get('error', 'Unknown error')}")
    
    elif args.stage == 'train':
        # Run only training
        data_sources = pipeline.fetch_training_data()
        features = pipeline.extract_and_save_features(data_sources)
        training_results = pipeline.train_models(features)
        
        print("=== Training Results ===")
        for model_name, metrics in training_results.items():
            if isinstance(metrics, dict):
                print(f"{model_name}:")
                print(f"  CV AUC: {metrics.get('cv_auc_mean', 0):.3f} ± {metrics.get('cv_auc_std', 0):.3f}")
                print(f"  Precision: {metrics.get('precision', 0):.3f}")
                print(f"  Recall: {metrics.get('recall', 0):.3f}")
    
    elif args.stage == 'evaluate':
        # Run only evaluation
        results = pipeline.evaluate_pipeline()
        
        print("=== Evaluation Results ===")
        print(f"Status: {results.get('pipeline_status', 'UNKNOWN')}")
        
        if results.get('model_status'):
            status = results['model_status']
            print(f"Models loaded: {status.get('is_loaded', False)}")
            print(f"Feature count: {status.get('feature_count', 0)}")
        
        if results.get('performance_analysis'):
            analysis = results['performance_analysis']
            print(f"Total recommendations: {analysis.get('total_recommendations', 0)}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
