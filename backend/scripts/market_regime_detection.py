# scripts/market_regime_detection.py

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from arch import arch_model
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

from utils.logger import setup_logging
logger = setup_logging()
from scripts.data_fetcher import get_historical_data

class MarketRegimeDetection:
    """
    Advanced market regime detection using multiple methods:
    - Hidden Markov Models (HMM)
    - GARCH models for volatility regime switching
    - Advanced clustering algorithms
    """
    def __init__(self, symbol, n_regimes=3, lookback_period="2y"):
        """
        Args:
            symbol (str): The stock symbol to analyze.
            n_regimes (int): The number of hidden market regimes to detect (e.g., 3 for Bull, Bear, Neutral).
            lookback_period (str): The period to fetch historical data for (e.g., "1y", "5y").
        """
        self.symbol = symbol
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.hmm_model = None
        self.garch_model = None
        self.regimes = None
        self.volatility_regimes = None
        self.clustering_regimes = None
        self.data = None

    def _prepare_enhanced_data(self):
        """
        Fetches historical data and prepares enhanced features for regime detection.
        Returns the processed DataFrame and feature array.
        """
        logger.info(f"Fetching historical data for {self.symbol} for enhanced regime analysis...")
        data = get_historical_data(self.symbol, period=self.lookback_period)
        if data is None or data.empty:
            logger.warning(f"No historical data found for {self.symbol}.")
            return None, None

        # Enhanced feature engineering
        data['returns'] = data['Close'].pct_change().fillna(0)
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
        
        # Volatility measures
        data['volatility_21'] = data['returns'].rolling(window=21).std().fillna(0)
        data['volatility_5'] = data['returns'].rolling(window=5).std().fillna(0)
        data['volatility_63'] = data['returns'].rolling(window=63).std().fillna(0)
        
        # Volume features
        data['volume_ma'] = data['Volume'].rolling(window=20).mean().fillna(data['Volume'])
        data['volume_ratio'] = data['Volume'] / data['volume_ma']
        
        # Price momentum features
        data['price_momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        data['price_momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Correlation with market (using price changes as proxy)
        data['price_change'] = data['Close'].pct_change()
        market_correlation = data['price_change'].rolling(window=63).corr(data['price_change'].shift(1)).fillna(0)
        data['market_correlation'] = market_correlation
        
        # Select features for modeling
        feature_columns = [
            'returns', 'volatility_21', 'volume_ratio', 
            'price_momentum_5', 'price_momentum_20', 'market_correlation'
        ]
        
        # Remove rows with NaN values
        data = data.dropna()
        
        if len(data) < 100:  # Ensure sufficient data
            logger.warning(f"Insufficient data for {self.symbol} after feature engineering")
            return None, None
        
        features = data[feature_columns].values
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        self.data = data
        return data, scaled_features

    def _prepare_data(self):
        """
        Legacy method for backward compatibility.
        """
        data, features = self._prepare_enhanced_data()
        return features

    def fit_garch_volatility_regimes(self):
        """
        Fits GARCH model for volatility regime switching analysis.
        """
        try:
            if self.data is None:
                data, _ = self._prepare_enhanced_data()
                if data is None:
                    return
            
            returns = self.data['returns'].dropna() * 100  # Convert to percentage
            
            if len(returns) < 100:
                logger.warning(f"Insufficient data for GARCH model for {self.symbol}")
                return
            
            # Additional safety checks for GARCH model
            if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
                logger.warning(f"Invalid values detected in returns for GARCH model for {self.symbol}")
                return
            
            # Check for extremely small or zero returns which can cause GARCH issues
            if np.all(np.abs(returns) < 1e-8):
                logger.warning(f"Returns too small for GARCH model for {self.symbol}")
                return
            
            # Fit GARCH(1,1) model with safer parameters
            self.garch_model = arch_model(
                returns, 
                vol='Garch', 
                p=1, 
                q=1, 
                dist='normal'
            )
            garch_fit = self.garch_model.fit(
                disp='off', 
                show_warning=False,
                options={'maxiter': 1000, 'ftol': 1e-6}  # Add convergence options
            )
            
            # Get conditional volatility
            conditional_volatility = garch_fit.conditional_volatility
            
            # Additional check for conditional volatility
            if conditional_volatility is None or len(conditional_volatility) == 0:
                logger.warning(f"Empty conditional volatility from GARCH model for {self.symbol}")
                return
            
            # Classify volatility regimes using quantiles
            vol_quantiles = conditional_volatility.quantile([0.33, 0.67])
            
            # Create volatility regime labels
            self.volatility_regimes = np.zeros(len(conditional_volatility))
            self.volatility_regimes[conditional_volatility <= vol_quantiles.iloc[0]] = 0  # Low volatility
            self.volatility_regimes[(conditional_volatility > vol_quantiles.iloc[0]) & 
                                   (conditional_volatility <= vol_quantiles.iloc[1])] = 1  # Medium volatility
            self.volatility_regimes[conditional_volatility > vol_quantiles.iloc[1]] = 2  # High volatility
            
            logger.info(f"GARCH volatility regime model fitted successfully for {self.symbol}")
            
            return {
                'model': garch_fit,
                'conditional_volatility': conditional_volatility,
                'volatility_regimes': self.volatility_regimes,
                'quantiles': vol_quantiles
            }
            
        except Exception as e:
            logger.error(f"Error fitting GARCH model for {self.symbol}: {e}")
            return None
    
    def fit_clustering_regimes(self):
        """
        Fits advanced clustering algorithms for regime classification.
        """
        try:
            data, features = self._prepare_enhanced_data()
            if features is None:
                return
            
            # Determine optimal number of clusters using silhouette score
            best_score = -1
            best_k = self.n_regimes
            
            for k in range(2, min(6, len(features) // 20)):  # Test different cluster numbers
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(features)
                
                if len(np.unique(cluster_labels)) > 1:  # Ensure multiple clusters
                    score = silhouette_score(features, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
            
            # Fit final clustering model
            final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            self.clustering_regimes = final_kmeans.fit_predict(features)
            
            # Calculate cluster characteristics
            cluster_characteristics = []
            for i in range(best_k):
                cluster_mask = self.clustering_regimes == i
                cluster_data = data.iloc[cluster_mask]
                
                characteristics = {
                    'cluster': i,
                    'mean_return': cluster_data['returns'].mean(),
                    'mean_volatility': cluster_data['volatility_21'].mean(),
                    'mean_volume_ratio': cluster_data['volume_ratio'].mean(),
                    'count': np.sum(cluster_mask)
                }
                cluster_characteristics.append(characteristics)
            
            # Sort by mean return and assign labels
            cluster_characteristics.sort(key=lambda x: x['mean_return'], reverse=True)
            
            logger.info(f"Clustering regime model fitted successfully for {self.symbol} with {best_k} clusters")
            
            return {
                'model': final_kmeans,
                'regimes': self.clustering_regimes,
                'n_clusters': best_k,
                'silhouette_score': best_score,
                'characteristics': cluster_characteristics
            }
            
        except Exception as e:
            logger.error(f"Error fitting clustering model for {self.symbol}: {e}")
            return None
    
    def fit_all_models(self):
        """
        Fits all regime detection models (HMM, GARCH, Clustering).
        """
        logger.info(f"Fitting all regime detection models for {self.symbol}")
        
        # Fit HMM model
        self.fit()
        
        # Fit GARCH volatility model
        garch_results = self.fit_garch_volatility_regimes()
        
        # Fit clustering model
        clustering_results = self.fit_clustering_regimes()
        
        return {
            'hmm_fitted': self.regimes is not None,
            'garch_fitted': garch_results is not None,
            'clustering_fitted': clustering_results is not None,
            'garch_results': garch_results,
            'clustering_results': clustering_results
        }

    def fit(self):
        """
        Fits the HMM model to the historical data.
        """
        features = self._prepare_data()
        if features is None:
            return

        # Use safer parameters to avoid segmentation faults
        # Diagonal covariance is more stable than full covariance
        # Reduced iterations to prevent convergence issues
        self.hmm_model = GaussianHMM(
            n_components=self.n_regimes, 
            covariance_type="diag",  # Changed from "full" to "diag" for stability
            n_iter=100,              # Reduced from 1000 to 100
            random_state=42,
            tol=1e-2                 # Added tolerance for faster convergence
        )
        try:
            # Additional safety checks
            if len(features) < 50:  # Need minimum data for HMM
                logger.warning(f"Insufficient data for HMM model for {self.symbol}: {len(features)} samples")
                return
            
            # Check for invalid values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning(f"Invalid values detected in features for {self.symbol}")
                return
            
            self.hmm_model.fit(features)
            self.regimes = self.hmm_model.predict(features)
            logger.info(f"HMM model fitted successfully for {self.symbol}.")
        except Exception as e:
            logger.error(f"Error fitting HMM model for {self.symbol}: {e}")
            # Set regimes to None to indicate failure
            self.regimes = None

    def get_current_regime(self):
        """
        Returns the current market regime for the symbol.
        """
        if self.regimes is None:
            return None
        return self.regimes[-1]

    def get_current_volatility_regime(self):
        """
        Returns the current volatility regime from GARCH analysis.
        """
        if self.volatility_regimes is None:
            return None
        return int(self.volatility_regimes[-1])
    
    def get_current_clustering_regime(self):
        """
        Returns the current clustering regime.
        """
        if self.clustering_regimes is None:
            return None
        return int(self.clustering_regimes[-1])
    
    def get_comprehensive_regime_analysis(self):
        """
        Returns comprehensive analysis combining all regime detection methods.
        """
        analysis = {
            'symbol': self.symbol,
            'hmm_regime': self.get_current_regime(),
            'volatility_regime': self.get_current_volatility_regime(),
            'clustering_regime': self.get_current_clustering_regime(),
            'regime_details': self.get_regime_details(),
            'consensus_regime': None,
            'confidence': 0.0
        }
        
        # Calculate consensus regime
        regimes = [r for r in [analysis['hmm_regime'], analysis['volatility_regime'], analysis['clustering_regime']] if r is not None]
        
        if regimes:
            # Simple majority vote or most common regime
            from collections import Counter
            regime_counts = Counter(regimes)
            analysis['consensus_regime'] = regime_counts.most_common(1)[0][0]
            analysis['confidence'] = regime_counts.most_common(1)[0][1] / len(regimes)
        
        return analysis

    def get_regime_details(self):
        """
        Returns a summary of the detected regimes.
        """
        if self.hmm_model is None:
            return None

        regime_details = []
        for i in range(self.n_regimes):
            details = {
                'regime': i,
                'mean_return': self.hmm_model.means_[i][0],
                'mean_volatility': self.hmm_model.means_[i][1]
            }
            regime_details.append(details)
        
        # Sort regimes by mean return to identify Bull/Bear/Neutral
        regime_details.sort(key=lambda x: x['mean_return'], reverse=True)

        # Assign labels
        if self.n_regimes == 3:
            regime_details[0]['label'] = 'Bull'
            regime_details[1]['label'] = 'Neutral'
            regime_details[2]['label'] = 'Bear'
        
        return regime_details

if __name__ == '__main__':
    # Enhanced example usage
    print("=== Advanced Market Regime Detection Demo ===")
    
    symbol = 'RELIANCE.NS'
    mrd = MarketRegimeDetection(symbol, n_regimes=3, lookback_period='2y')
    
    print(f"\nAnalyzing {symbol} with enhanced regime detection...")
    
    # Fit all models
    results = mrd.fit_all_models()
    
    if results:
        print(f"\nModel Fitting Results:")
        print(f"  - HMM Model: {'✓' if results['hmm_fitted'] else '✗'}")
        print(f"  - GARCH Model: {'✓' if results['garch_fitted'] else '✗'}")
        print(f"  - Clustering Model: {'✓' if results['clustering_fitted'] else '✗'}")
        
        # Get comprehensive analysis
        analysis = mrd.get_comprehensive_regime_analysis()
        
        print(f"\nCurrent Market Regimes:")
        print(f"  - HMM Regime: {analysis['hmm_regime']}")
        print(f"  - Volatility Regime: {analysis['volatility_regime']} (0=Low, 1=Medium, 2=High)")
        print(f"  - Clustering Regime: {analysis['clustering_regime']}")
        print(f"  - Consensus Regime: {analysis['consensus_regime']} (Confidence: {analysis['confidence']:.2f})")
        
        # Show regime details
        if analysis['regime_details']:
            print(f"\nHMM Regime Details:")
            for detail in analysis['regime_details']:
                print(f"  - {detail['label']}: Mean Return={detail['mean_return']:.4f}, Mean Volatility={detail['mean_volatility']:.4f}")
        
        # Show clustering results if available
        if results['clustering_results']:
            clustering = results['clustering_results']
            print(f"\nClustering Analysis:")
            print(f"  - Optimal Clusters: {clustering['n_clusters']}")
            print(f"  - Silhouette Score: {clustering['silhouette_score']:.3f}")
            
            print(f"  - Cluster Characteristics:")
            for char in clustering['characteristics']:
                print(f"    Cluster {char['cluster']}: Return={char['mean_return']:.4f}, Vol={char['mean_volatility']:.4f}, Count={char['count']}")
    
    else:
        print("Failed to fit regime detection models. Please check data availability.")

