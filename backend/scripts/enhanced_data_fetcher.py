"""
Enhanced Data Fetcher with Multi-Provider Reconciliation
=========================================================

This module implements improved data fetching with:
1. Multi-provider data reconciliation (NSE, Yahoo, alternatives)
2. Data continuity checks
3. Corporate action normalization
4. Enhanced caching with checksums
5. Drift detection
"""

import pandas as pd
import numpy as np
import json
import os
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logging
from utils.memory_utils import optimize_dataframe_memory
import yfinance as yf
from config import (
    HISTORICAL_DATA_PERIOD, MAX_WORKER_THREADS, REQUEST_DELAY,
    TIMEOUT_SECONDS, RATE_LIMIT_DELAY, BACKOFF_MULTIPLIER
)

# Import existing fetchers
from scripts.data_fetcher import get_historical_data_with_retry
from scripts.alternative_data_fetcher import AlternativeDataFetcher

logger = setup_logging()


class EnhancedDataFetcher:
    """
    Enhanced data fetcher with multi-provider reconciliation and quality checks
    """
    
    def __init__(self, cache_dir: str = None):
        """Initialize the enhanced data fetcher"""
        self.alt_fetcher = AlternativeDataFetcher()
        
        # Set up cache directory
        if cache_dir is None:
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(backend_dir, "enhanced_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Provider weights for reconciliation
        self.provider_weights = {
            'nse_official': 0.4,
            'yahoo_finance': 0.35,
            'alternative': 0.25
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'max_missing_pct': 0.02,  # Max 2% missing bars
            'min_data_points': 250,    # Minimum data points for 2 years
            'price_deviation_pct': 0.05,  # Max 5% deviation between providers
            'volume_deviation_pct': 0.20   # Max 20% volume deviation
        }
    
    def fetch_from_multiple_providers(self, symbol: str, period: str = '2y') -> Dict[str, pd.DataFrame]:
        """
        Fetch data from multiple providers in parallel
        
        Args:
            symbol: Stock symbol
            period: Time period for historical data
            
        Returns:
            Dictionary mapping provider names to DataFrames
        """
        provider_data = {}
        
        # Define provider fetch functions
        providers = {
            'nse_official': lambda: self._fetch_nse_data(symbol, period),
            'yahoo_finance': lambda: self._fetch_yahoo_data(symbol, period),
            'alternative': lambda: self._fetch_alternative_data(symbol, period)
        }
        
        # Fetch from all providers in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_provider = {
                executor.submit(fetch_func): provider_name
                for provider_name, fetch_func in providers.items()
            }
            
            for future in as_completed(future_to_provider):
                provider_name = future_to_provider[future]
                try:
                    data = future.result(timeout=TIMEOUT_SECONDS)
                    if not data.empty:
                        provider_data[provider_name] = data
                        logger.info(f"Fetched {len(data)} data points from {provider_name} for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to fetch from {provider_name} for {symbol}: {e}")
        
        return provider_data
    
    def _fetch_nse_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from NSE official API"""
        try:
            period_days = self._period_to_days(period)
            return self.alt_fetcher.get_nse_stock_data(symbol, period_days)
        except Exception as e:
            logger.error(f"NSE fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_yahoo_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            return get_historical_data_with_retry(symbol, period=period)
        except Exception as e:
            logger.error(f"Yahoo fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_alternative_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Fetch data from alternative sources"""
        try:
            return self.alt_fetcher.get_historical_data(symbol, period=period)
        except Exception as e:
            logger.error(f"Alternative fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def reconcile_data(self, provider_data: Dict[str, pd.DataFrame], symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Reconcile data from multiple providers
        
        Args:
            provider_data: Dictionary of provider DataFrames
            symbol: Stock symbol for logging
            
        Returns:
            Tuple of (reconciled DataFrame, reconciliation metrics)
        """
        if not provider_data:
            logger.warning(f"No data from any provider for {symbol}")
            return pd.DataFrame(), {'status': 'no_data'}
        
        # If only one provider, use it directly but log warning
        if len(provider_data) == 1:
            provider_name = list(provider_data.keys())[0]
            logger.warning(f"Only {provider_name} provided data for {symbol}")
            return provider_data[provider_name], {'status': 'single_provider', 'provider': provider_name}
        
        # Find common date range
        all_dates = []
        for df in provider_data.values():
            all_dates.extend(df.index.tolist())
        
        common_dates = pd.DatetimeIndex(sorted(set(all_dates)))
        
        # Create reconciled DataFrame
        reconciled_data = []
        reconciliation_metrics = {
            'status': 'reconciled',
            'providers': list(provider_data.keys()),
            'disagreements': 0,
            'total_days': len(common_dates),
            'confidence_scores': []
        }
        
        for date in common_dates:
            daily_data = {}
            provider_values = {}
            
            # Collect values from each provider
            for provider_name, df in provider_data.items():
                if date in df.index:
                    provider_values[provider_name] = {
                        'Open': df.loc[date, 'Open'] if 'Open' in df.columns else None,
                        'High': df.loc[date, 'High'] if 'High' in df.columns else None,
                        'Low': df.loc[date, 'Low'] if 'Low' in df.columns else None,
                        'Close': df.loc[date, 'Close'] if 'Close' in df.columns else None,
                        'Volume': df.loc[date, 'Volume'] if 'Volume' in df.columns else None
                    }
            
            if not provider_values:
                continue
            
            # Reconcile each field
            reconciled_row = {'Date': date}
            confidence_score = 1.0
            
            for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
                field_values = []
                field_weights = []
                
                for provider_name, values in provider_values.items():
                    if values.get(field) is not None:
                        field_values.append(values[field])
                        field_weights.append(self.provider_weights.get(provider_name, 0.2))
                
                if field_values:
                    # Check for disagreements
                    if field != 'Volume':
                        # For price fields, check deviation
                        mean_val = np.mean(field_values)
                        max_deviation = max(abs(v - mean_val) / mean_val for v in field_values if mean_val != 0)
                        
                        if max_deviation > self.quality_thresholds['price_deviation_pct']:
                            reconciliation_metrics['disagreements'] += 1
                            confidence_score *= 0.9
                            logger.debug(f"Price disagreement for {symbol} on {date}: {field} deviation {max_deviation:.2%}")
                    
                    # Use weighted average for reconciliation
                    if field_weights:
                        weighted_sum = sum(v * w for v, w in zip(field_values, field_weights))
                        weight_sum = sum(field_weights)
                        reconciled_row[field] = weighted_sum / weight_sum if weight_sum > 0 else np.mean(field_values)
                    else:
                        reconciled_row[field] = np.mean(field_values)
            
            reconciled_data.append(reconciled_row)
            reconciliation_metrics['confidence_scores'].append(confidence_score)
        
        if not reconciled_data:
            return pd.DataFrame(), {'status': 'reconciliation_failed'}
        
        # Create final DataFrame
        df = pd.DataFrame(reconciled_data)
        df.set_index('Date', inplace=True)
        df = df.sort_index()
        
        # Calculate average confidence
        reconciliation_metrics['avg_confidence'] = np.mean(reconciliation_metrics['confidence_scores'])
        reconciliation_metrics['disagreement_rate'] = reconciliation_metrics['disagreements'] / (len(reconciled_data) * 5)  # 5 fields per day
        
        return df, reconciliation_metrics
    
    def check_data_continuity(self, df: pd.DataFrame, symbol: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for data continuity and missing bars
        
        Args:
            df: DataFrame to check
            symbol: Stock symbol for logging
            
        Returns:
            Tuple of (is_continuous, continuity_metrics)
        """
        if df.empty:
            return False, {'status': 'empty_data'}
        
        # Check for missing trading days
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')  # Business days
        missing_days = date_range.difference(df.index)
        
        # Filter out known holidays (simplified - you'd want a proper holiday calendar)
        # For now, just check the percentage of missing days
        total_trading_days = len(date_range)
        missing_count = len(missing_days)
        missing_pct = missing_count / total_trading_days if total_trading_days > 0 else 1.0
        
        continuity_metrics = {
            'total_days': len(df),
            'expected_days': total_trading_days,
            'missing_days': missing_count,
            'missing_pct': missing_pct,
            'date_gaps': []
        }
        
        # Find large gaps
        if len(df) > 1:
            date_diffs = df.index[1:] - df.index[:-1]
            large_gaps = [(df.index[i], df.index[i+1], date_diffs[i].days) 
                         for i in range(len(date_diffs)) 
                         if date_diffs[i].days > 3]  # Gaps larger than 3 days
            continuity_metrics['date_gaps'] = large_gaps
        
        # Check if continuity meets threshold
        is_continuous = missing_pct <= self.quality_thresholds['max_missing_pct']
        
        if not is_continuous:
            logger.warning(f"{symbol} has {missing_pct:.1%} missing data (threshold: {self.quality_thresholds['max_missing_pct']:.1%})")
        
        return is_continuous, continuity_metrics
    
    def normalize_corporate_actions(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize for splits, dividends, and corporate actions
        
        Args:
            df: DataFrame to normalize
            symbol: Stock symbol
            
        Returns:
            Normalized DataFrame
        """
        try:
            # Get split and dividend data from yfinance
            ticker = yf.Ticker(f"{symbol}.NS")
            
            # Get splits
            splits = ticker.splits
            if not splits.empty:
                logger.info(f"Adjusting {symbol} for {len(splits)} stock splits")
                
                # Apply split adjustments
                for split_date, split_ratio in splits.items():
                    if split_date in df.index:
                        # Adjust prices before split date
                        pre_split = df.index < split_date
                        df.loc[pre_split, ['Open', 'High', 'Low', 'Close']] /= split_ratio
                        df.loc[pre_split, 'Volume'] *= split_ratio
            
            # Get dividends
            dividends = ticker.dividends
            if not dividends.empty:
                logger.info(f"Found {len(dividends)} dividends for {symbol}")
                # You might want to adjust for dividends depending on your strategy
                # For now, we'll just log them
            
            # Ensure data types are correct
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Volume' in df.columns:
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
            
        except Exception as e:
            logger.warning(f"Could not normalize corporate actions for {symbol}: {e}")
        
        return df
    
    def calculate_checksum(self, df: pd.DataFrame) -> str:
        """Calculate checksum for DataFrame"""
        if df.empty:
            return ""
        
        # Create a string representation of the data
        data_str = df.to_json(orient='split', date_format='iso')
        
        # Calculate SHA256 hash
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def save_with_checksum(self, df: pd.DataFrame, filepath: str, metadata: Dict[str, Any] = None):
        """Save DataFrame with checksum and metadata"""
        try:
            # Calculate checksum
            checksum = self.calculate_checksum(df)
            
            # Save data
            df.to_csv(filepath)
            
            # Save metadata
            meta_filepath = filepath.replace('.csv', '_meta.json')
            metadata = metadata or {}
            metadata.update({
                'checksum': checksum,
                'saved_at': datetime.now().isoformat(),
                'rows': len(df),
                'columns': list(df.columns)
            })
            
            with open(meta_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved data with checksum to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving data with checksum: {e}")
    
    def load_with_checksum_verification(self, filepath: str) -> Tuple[Optional[pd.DataFrame], bool]:
        """Load DataFrame and verify checksum"""
        try:
            if not os.path.exists(filepath):
                return None, False
            
            # Load data
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Load metadata
            meta_filepath = filepath.replace('.csv', '_meta.json')
            if os.path.exists(meta_filepath):
                with open(meta_filepath, 'r') as f:
                    metadata = json.load(f)
                
                # Verify checksum
                current_checksum = self.calculate_checksum(df)
                stored_checksum = metadata.get('checksum', '')
                
                if current_checksum == stored_checksum:
                    return df, True
                else:
                    logger.warning(f"Checksum mismatch for {filepath}")
                    return df, False
            
            return df, False
            
        except Exception as e:
            logger.error(f"Error loading data with checksum: {e}")
            return None, False
    
    def detect_data_drift(self, current_df: pd.DataFrame, historical_df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Detect if there's significant drift in the data
        
        Args:
            current_df: Current data
            historical_df: Historical cached data
            symbol: Stock symbol
            
        Returns:
            Drift metrics
        """
        drift_metrics = {
            'has_drift': False,
            'price_drift': 0,
            'volume_drift': 0,
            'volatility_drift': 0
        }
        
        try:
            # Find overlapping dates
            common_dates = current_df.index.intersection(historical_df.index)
            
            if len(common_dates) < 10:
                return drift_metrics
            
            # Compare closing prices
            current_prices = current_df.loc[common_dates, 'Close']
            historical_prices = historical_df.loc[common_dates, 'Close']
            
            price_diff = abs(current_prices - historical_prices) / historical_prices
            drift_metrics['price_drift'] = price_diff.mean()
            
            # Compare volumes
            if 'Volume' in current_df.columns and 'Volume' in historical_df.columns:
                current_volume = current_df.loc[common_dates, 'Volume']
                historical_volume = historical_df.loc[common_dates, 'Volume']
                
                volume_diff = abs(current_volume - historical_volume) / (historical_volume + 1)
                drift_metrics['volume_drift'] = volume_diff.mean()
            
            # Compare volatility
            current_volatility = current_prices.pct_change().std()
            historical_volatility = historical_prices.pct_change().std()
            
            if historical_volatility > 0:
                drift_metrics['volatility_drift'] = abs(current_volatility - historical_volatility) / historical_volatility
            
            # Check if drift exceeds thresholds
            if (drift_metrics['price_drift'] > 0.05 or 
                drift_metrics['volume_drift'] > 0.20 or 
                drift_metrics['volatility_drift'] > 0.30):
                drift_metrics['has_drift'] = True
                logger.warning(f"Data drift detected for {symbol}: price={drift_metrics['price_drift']:.2%}, "
                             f"volume={drift_metrics['volume_drift']:.2%}, "
                             f"volatility={drift_metrics['volatility_drift']:.2%}")
            
        except Exception as e:
            logger.error(f"Error detecting drift for {symbol}: {e}")
        
        return drift_metrics
    
    def get_enhanced_historical_data(self, symbol: str, period: str = '2y', 
                                    force_refresh: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get enhanced historical data with multi-provider reconciliation
        
        Args:
            symbol: Stock symbol
            period: Time period
            force_refresh: Force refresh from providers
            
        Returns:
            Tuple of (DataFrame, quality_metrics)
        """
        logger.info(f"Getting enhanced historical data for {symbol}")
        
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{period}_enhanced.csv")
        
        if not force_refresh and os.path.exists(cache_file):
            df, checksum_valid = self.load_with_checksum_verification(cache_file)
            
            if df is not None and checksum_valid:
                # Check if cache is fresh (less than 1 day old)
                file_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))).days
                
                if file_age < 1:
                    logger.info(f"Using cached enhanced data for {symbol}")
                    return df, {'source': 'cache', 'checksum_valid': True}
        
        # Fetch from multiple providers
        provider_data = self.fetch_from_multiple_providers(symbol, period)
        
        if not provider_data:
            logger.error(f"No data available from any provider for {symbol}")
            return pd.DataFrame(), {'status': 'no_data_available'}
        
        # Reconcile data
        reconciled_df, reconciliation_metrics = self.reconcile_data(provider_data, symbol)
        
        if reconciled_df.empty:
            return reconciled_df, reconciliation_metrics
        
        # Check data continuity
        is_continuous, continuity_metrics = self.check_data_continuity(reconciled_df, symbol)
        
        if not is_continuous and continuity_metrics['missing_pct'] > self.quality_thresholds['max_missing_pct']:
            logger.warning(f"Skipping {symbol} due to poor data continuity")
            return pd.DataFrame(), {'status': 'poor_continuity', 'metrics': continuity_metrics}
        
        # Normalize for corporate actions
        reconciled_df = self.normalize_corporate_actions(reconciled_df, symbol)
        
        # Check for data drift if we have historical cache
        drift_metrics = {}
        if os.path.exists(cache_file):
            old_df, _ = self.load_with_checksum_verification(cache_file)
            if old_df is not None:
                drift_metrics = self.detect_data_drift(reconciled_df, old_df, symbol)
        
        # Optimize memory
        reconciled_df = optimize_dataframe_memory(reconciled_df)
        
        # Save with checksum
        quality_metrics = {
            'reconciliation': reconciliation_metrics,
            'continuity': continuity_metrics,
            'drift': drift_metrics,
            'providers_used': list(provider_data.keys()),
            'final_rows': len(reconciled_df)
        }
        
        self.save_with_checksum(reconciled_df, cache_file, quality_metrics)
        
        return reconciled_df, quality_metrics
    
    def _period_to_days(self, period: str) -> int:
        """Convert period string to days"""
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90,
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        return period_map.get(period, 365)


# Singleton instance
_enhanced_fetcher = None

def get_enhanced_fetcher() -> EnhancedDataFetcher:
    """Get singleton instance of enhanced fetcher"""
    global _enhanced_fetcher
    if _enhanced_fetcher is None:
        _enhanced_fetcher = EnhancedDataFetcher()
    return _enhanced_fetcher
