#!/usr/bin/env python3
"""
ML Classifier Trainer
=====================

Trains classifiers to predict 10-20 bar positive R multiple for swing trading.
Uses Platt scaling for probability calibration.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.feature_extractor import MLFeatureExtractor
from utils.logger import setup_logging

logger = setup_logging()

class SwingTradingClassifier:
    """
    ML Classifier for predicting positive R multiple in swing trading
    """
    
    def __init__(self, target_horizon: int = 10):
        """
        Initialize the classifier
        
        Args:
            target_horizon: Prediction horizon in days (10 or 20)
        """
        self.target_horizon = target_horizon
        self.feature_extractor = MLFeatureExtractor()
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'needs_scaling': False
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                ),
                'needs_scaling': False
            },
            'logistic_regression': {
                'model': LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                ),
                'needs_scaling': True
            }
        }
        
        # Target configuration
        self.target_config = {
            'r_multiple_threshold': 1.0,  # Positive R multiple (1:1 minimum)
            'stop_loss_pct': 0.02,  # 2% stop loss assumption
            'min_return_threshold': 0.02  # Minimum 2% return to be considered positive
        }
    
    def prepare_training_data(self, data_sources: List[Tuple[pd.DataFrame, str]]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data from multiple data sources
        
        Args:
            data_sources: List of (DataFrame, symbol) tuples
            
        Returns:
            Features and target DataFrames
        """
        logger.info(f"Preparing training data from {len(data_sources)} sources")
        
        all_features = []
        all_targets = []
        
        for df, symbol in data_sources:
            try:
                # Extract features for this symbol
                features = self.feature_extractor.extract_features(df, symbol)
                
                if features.empty:
                    logger.warning(f"No features extracted for {symbol}")
                    continue
                
                # Get target variable
                target_col = f'target_r_positive_{self.target_horizon}d'
                if target_col not in features.columns:
                    logger.warning(f"Target column {target_col} not found for {symbol}")
                    continue
                
                # Filter out rows with NaN targets
                valid_idx = features[target_col].notna()
                if not valid_idx.any():
                    logger.warning(f"No valid targets for {symbol}")
                    continue
                
                features_clean = features[valid_idx].copy()
                targets_clean = features_clean[target_col].copy()
                
                # Remove target columns from features
                feature_cols = [col for col in features_clean.columns if not col.startswith('target_')]
                features_only = features_clean[feature_cols]
                
                # Add symbol identifier
                features_only['symbol'] = symbol
                
                all_features.append(features_only)
                all_targets.append(targets_clean)
                
                logger.info(f"Added {len(features_only)} samples from {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data found")
        
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_targets = pd.concat(all_targets, ignore_index=True)
        
        logger.info(f"Combined training data: {len(combined_features)} samples, {len(combined_features.columns)} features")
        
        return combined_features, combined_targets
    
    def train_models(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        Train multiple ML models with time series cross-validation
        
        Args:
            features: Feature DataFrame
            targets: Target Series
            
        Returns:
            Dictionary with training results
        """
        logger.info("Training ML models with time series cross-validation")
        
        # Prepare features (remove non-numeric columns)
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        if 'symbol' in feature_cols:
            feature_cols.remove('symbol')
        
        X = features[feature_cols].fillna(0)  # Fill any remaining NaN with 0
        y = targets.astype(int)
        
        self.feature_names = feature_cols
        logger.info(f"Training with {len(feature_cols)} features")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        training_results = {}
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name}")
            
            try:
                # Scale features if needed
                if config['needs_scaling']:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    self.scalers[model_name] = scaler
                else:
                    X_scaled = X.values
                    self.scalers[model_name] = None
                
                # Train base model
                base_model = config['model']
                
                # Cross-validation scores
                cv_scores = cross_val_score(base_model, X_scaled, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
                
                # Train on full dataset
                base_model.fit(X_scaled, y)
                
                # Calibrate with Platt scaling
                logger.info(f"Calibrating {model_name} with Platt scaling")
                calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
                calibrated_model.fit(X_scaled, y)
                
                self.models[model_name] = {
                    'base_model': base_model,
                    'calibrated_model': calibrated_model
                }
                
                # Evaluate on training data (for metrics)
                y_pred = calibrated_model.predict(X_scaled)
                y_prob = calibrated_model.predict_proba(X_scaled)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y, y_pred, y_prob)
                metrics['cv_auc_scores'] = cv_scores
                metrics['cv_auc_mean'] = np.mean(cv_scores)
                metrics['cv_auc_std'] = np.std(cv_scores)
                
                training_results[model_name] = metrics
                
                logger.info(f"{model_name} - CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        self.training_metrics = training_results
        
        # Select best model
        best_model = self._select_best_model(training_results)
        logger.info(f"Best model selected: {best_model}")
        
        return training_results
    
    def _calculate_metrics(self, y_true: np.array, y_pred: np.array, y_prob: np.array) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        try:
            metrics = {
                'accuracy': np.mean(y_true == y_pred),
                'precision': np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0,
                'recall': np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0,
                'roc_auc': roc_auc_score(y_true, y_prob),
                'avg_precision': average_precision_score(y_true, y_prob)
            }
            
            # F1 score
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1'] = 0
            
            # Class distribution
            metrics['positive_rate'] = np.mean(y_true)
            metrics['prediction_rate'] = np.mean(y_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _select_best_model(self, training_results: Dict[str, Any]) -> str:
        """Select best model based on cross-validation AUC"""
        try:
            best_model = None
            best_score = 0
            
            for model_name, metrics in training_results.items():
                cv_score = metrics.get('cv_auc_mean', 0)
                if cv_score > best_score:
                    best_score = cv_score
                    best_model = model_name
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error selecting best model: {e}")
            return 'random_forest'  # Default fallback
    
    def predict(self, features: pd.DataFrame, model_name: str = None) -> Dict[str, np.array]:
        """
        Make predictions using trained models
        
        Args:
            features: Feature DataFrame
            model_name: Specific model to use (None for best model)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if model_name is None:
            model_name = self._select_best_model(self.training_metrics)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Prepare features
        feature_cols = [col for col in self.feature_names if col in features.columns]
        X = features[feature_cols].fillna(0)
        
        # Scale if needed
        if self.scalers[model_name] is not None:
            X = self.scalers[model_name].transform(X)
        
        # Get model
        model = self.models[model_name]['calibrated_model']
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'model_used': model_name
        }
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model_name: Model to analyze
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name = self._select_best_model(self.training_metrics)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Get base model (before calibration)
        base_model = self.models[model_name]['base_model']
        
        # Extract feature importance
        if hasattr(base_model, 'feature_importances_'):
            importances = base_model.feature_importances_
        elif hasattr(base_model, 'coef_'):
            importances = np.abs(base_model.coef_[0])
        else:
            logger.warning(f"Model {model_name} does not have feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def save_models(self, save_dir: str) -> None:
        """Save trained models and metadata"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for model_name, model_dict in self.models.items():
            model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
            joblib.dump(model_dict, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save scalers
        for model_name, scaler in self.scalers.items():
            if scaler is not None:
                scaler_path = os.path.join(save_dir, f'{model_name}_scaler.joblib')
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved {model_name} scaler to {scaler_path}")
        
        # Save metadata (convert numpy arrays to lists for JSON serialization)
        serializable_metrics = {}
        for model_name, metrics in self.training_metrics.items():
            serializable_metrics[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[model_name][key] = value.tolist()
                elif isinstance(value, np.floating):
                    serializable_metrics[model_name][key] = float(value)
                elif isinstance(value, np.integer):
                    serializable_metrics[model_name][key] = int(value)
                else:
                    serializable_metrics[model_name][key] = value
        
        metadata = {
            'target_horizon': self.target_horizon,
            'feature_names': self.feature_names,
            'target_config': self.target_config,
            'training_metrics': serializable_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(save_dir, 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load_models(self, load_dir: str) -> None:
        """Load trained models and metadata"""
        # Load metadata
        metadata_path = os.path.join(load_dir, 'model_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.target_horizon = metadata['target_horizon']
            self.feature_names = metadata['feature_names']
            self.target_config = metadata['target_config']
            self.training_metrics = metadata['training_metrics']
            
            logger.info(f"Loaded metadata from {metadata_path}")
        else:
            logger.warning(f"Metadata file not found: {metadata_path}")
        
        # Load models
        for model_name in self.model_configs.keys():
            model_path = os.path.join(load_dir, f'{model_name}_model.joblib')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded {model_name} from {model_path}")
            
            # Load scaler
            scaler_path = os.path.join(load_dir, f'{model_name}_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scalers[model_name] = joblib.load(scaler_path)
                logger.info(f"Loaded {model_name} scaler from {scaler_path}")
            else:
                self.scalers[model_name] = None
    
    def evaluate_model_performance(self, test_features: pd.DataFrame, test_targets: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        
        Args:
            test_features: Test feature DataFrame
            test_targets: Test target Series
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("Evaluating model performance on test data")
        
        evaluation_results = {}
        
        for model_name in self.models.keys():
            try:
                predictions = self.predict(test_features, model_name)
                
                y_true = test_targets.astype(int)
                y_pred = predictions['predictions']
                y_prob = predictions['probabilities']
                
                metrics = self._calculate_metrics(y_true, y_pred, y_prob)
                evaluation_results[model_name] = metrics
                
                logger.info(f"{model_name} Test Performance:")
                logger.info(f"  AUC: {metrics['roc_auc']:.3f}")
                logger.info(f"  Precision: {metrics['precision']:.3f}")
                logger.info(f"  Recall: {metrics['recall']:.3f}")
                logger.info(f"  F1: {metrics['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
        
        return evaluation_results


def create_sample_data(n_symbols: int = 5, n_days: int = 1000) -> List[Tuple[pd.DataFrame, str]]:
    """Create sample data for testing"""
    np.random.seed(42)
    data_sources = []
    
    for i in range(n_symbols):
        symbol = f"STOCK_{i+1}"
        dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
        
        # Generate realistic price series
        base_price = 100
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = [base_price]
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices[:-1]],
            'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices[:-1]],
            'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices[:-1]],
            'Close': prices[:-1],
            'Volume': np.random.randint(10000, 1000000, n_days)
        })
        df.set_index('Date', inplace=True)
        
        data_sources.append((df, symbol))
    
    return data_sources


def main():
    """Main function for testing the classifier"""
    logger.info("Testing ML Classifier Trainer")
    
    # Create sample data
    data_sources = create_sample_data(n_symbols=3, n_days=500)
    
    # Initialize classifier
    classifier = SwingTradingClassifier(target_horizon=10)
    
    # Prepare training data
    features, targets = classifier.prepare_training_data(data_sources)
    
    # Train models
    training_results = classifier.train_models(features, targets)
    
    # Print results
    print("\n=== Training Results ===")
    for model_name, metrics in training_results.items():
        print(f"{model_name}:")
        print(f"  CV AUC: {metrics.get('cv_auc_mean', 0):.3f} ± {metrics.get('cv_auc_std', 0):.3f}")
        print(f"  Train AUC: {metrics.get('roc_auc', 0):.3f}")
        print(f"  Precision: {metrics.get('precision', 0):.3f}")
        print(f"  Recall: {metrics.get('recall', 0):.3f}")
        print()
    
    # Show feature importance
    best_model = classifier._select_best_model(training_results)
    importance_df = classifier.get_feature_importance(best_model, top_n=10)
    
    print(f"\n=== Top 10 Features ({best_model}) ===")
    print(importance_df.to_string(index=False))
    
    # Test predictions
    test_features = features.iloc[-100:].copy()  # Last 100 samples
    predictions = classifier.predict(test_features)
    
    print(f"\n=== Sample Predictions ===")
    print(f"Model used: {predictions['model_used']}")
    print(f"Positive predictions: {np.sum(predictions['predictions'])}/100")
    print(f"Average probability: {np.mean(predictions['probabilities']):.3f}")
    
    # Save models
    save_dir = "ml/models"
    os.makedirs(save_dir, exist_ok=True)
    classifier.save_models(save_dir)
    print(f"\nModels saved to {save_dir}")


if __name__ == "__main__":
    main()
