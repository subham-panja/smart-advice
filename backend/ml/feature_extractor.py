#!/usr/bin/env python3
"""
ML Feature Extractor
===================

Extracts comprehensive feature set for ML-based swing trading prediction:
- Returns (1/5/10 day)
- RSI regimes
- ADX trends
- ATR bands
- Volume Z-scores
- Pattern flags
"""

import os
import sys
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logging

logger = setup_logging()

class MLFeatureExtractor:
    """
    Extracts ML features for swing trading prediction
    """
    
    def __init__(self):
        """Initialize the feature extractor"""
        self.feature_config = {
            # Return horizons
            'return_periods': [1, 5, 10],
            
            # Technical indicators
            'rsi_period': 14,
            'adx_period': 14,
            'atr_period': 14,
            'bb_period': 20,
            'bb_std': 2,
            'volume_ma_period': 20,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            
            # Pattern detection windows
            'pattern_lookback': 20,
            'support_resistance_window': 50,
            
            # Regime detection
            'regime_window': 252,  # 1 year
            'volatility_window': 20
        }
        
    def extract_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Extract comprehensive feature set from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol (optional)
            
        Returns:
            DataFrame with extracted features
        """
        try:
            logger.info(f"Extracting ML features for {symbol or 'Unknown'}")
            
            if len(df) < 100:
                logger.warning(f"Insufficient data for feature extraction: {len(df)} rows")
                return pd.DataFrame()
            
            features_df = pd.DataFrame(index=df.index)
            
            # 1. Price and Return Features
            features_df = self._add_price_return_features(features_df, df)
            
            # 2. Technical Indicator Features
            features_df = self._add_technical_indicators(features_df, df)
            
            # 3. RSI Regime Features
            features_df = self._add_rsi_regime_features(features_df, df)
            
            # 4. ADX Trend Features
            features_df = self._add_adx_features(features_df, df)
            
            # 5. ATR Volatility Band Features
            features_df = self._add_atr_band_features(features_df, df)
            
            # 6. Volume Features
            features_df = self._add_volume_features(features_df, df)
            
            # 7. Pattern Recognition Features
            features_df = self._add_pattern_features(features_df, df)
            
            # 8. Market Regime Features
            features_df = self._add_market_regime_features(features_df, df)
            
            # 9. Cross-Sectional Features (if symbol provided)
            if symbol:
                features_df = self._add_cross_sectional_features(features_df, df, symbol)
            
            # 10. Target Variable (Future Returns)
            features_df = self._add_target_variables(features_df, df)
            
            # Clean up features
            features_df = self._clean_features(features_df)
            
            logger.info(f"Extracted {len(features_df.columns)} features for {len(features_df)} observations")
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def _add_price_return_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add price and return-based features"""
        try:
            # Basic price features
            features_df['price'] = df['Close']
            features_df['price_log'] = np.log(df['Close'])
            
            # Returns at different horizons
            for period in self.feature_config['return_periods']:
                features_df[f'return_{period}d'] = df['Close'].pct_change(period)
                features_df[f'return_{period}d_log'] = np.log(df['Close'] / df['Close'].shift(period))
            
            # Rolling statistics
            features_df['return_1d_ma_5'] = features_df['return_1d'].rolling(5).mean()
            features_df['return_1d_ma_20'] = features_df['return_1d'].rolling(20).mean()
            features_df['return_1d_std_20'] = features_df['return_1d'].rolling(20).std()
            
            # Price momentum
            features_df['price_momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
            features_df['price_momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
            features_df['price_momentum_60'] = df['Close'] / df['Close'].shift(60) - 1
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding price/return features: {e}")
            return features_df
    
    def _add_technical_indicators(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features"""
        try:
            # Moving averages
            sma_20 = talib.SMA(df['Close'], timeperiod=20)
            sma_50 = talib.SMA(df['Close'], timeperiod=50)
            sma_200 = talib.SMA(df['Close'], timeperiod=200)
            ema_12 = talib.EMA(df['Close'], timeperiod=12)
            ema_26 = talib.EMA(df['Close'], timeperiod=26)
            
            features_df['sma_20'] = sma_20
            features_df['sma_50'] = sma_50
            features_df['sma_200'] = sma_200
            features_df['price_vs_sma_20'] = (df['Close'] / sma_20) - 1
            features_df['price_vs_sma_50'] = (df['Close'] / sma_50) - 1
            features_df['price_vs_sma_200'] = (df['Close'] / sma_200) - 1
            features_df['sma_20_vs_50'] = (sma_20 / sma_50) - 1
            features_df['sma_50_vs_200'] = (sma_50 / sma_200) - 1
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(df['Close'], 
                                                     fastperiod=self.feature_config['macd_fast'],
                                                     slowperiod=self.feature_config['macd_slow'], 
                                                     signalperiod=self.feature_config['macd_signal'])
            features_df['macd'] = macd
            features_df['macd_signal'] = macd_signal
            features_df['macd_histogram'] = macd_hist
            features_df['macd_above_signal'] = (macd > macd_signal).astype(int)
            features_df['macd_above_zero'] = (macd > 0).astype(int)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], 
                                                        timeperiod=self.feature_config['bb_period'],
                                                        nbdevup=self.feature_config['bb_std'], 
                                                        nbdevdn=self.feature_config['bb_std'])
            features_df['bb_upper'] = bb_upper
            features_df['bb_lower'] = bb_lower
            features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features_df['bb_position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
            features_df['price_vs_bb_upper'] = (df['Close'] / bb_upper) - 1
            features_df['price_vs_bb_lower'] = (df['Close'] / bb_lower) - 1
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return features_df
    
    def _add_rsi_regime_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI regime-based features"""
        try:
            # RSI calculation
            rsi = talib.RSI(df['Close'], timeperiod=self.feature_config['rsi_period'])
            features_df['rsi'] = rsi
            
            # RSI regimes
            features_df['rsi_oversold'] = (rsi < 30).astype(int)
            features_df['rsi_overbought'] = (rsi > 70).astype(int)
            features_df['rsi_neutral'] = ((rsi >= 30) & (rsi <= 70)).astype(int)
            
            # RSI momentum and trends
            features_df['rsi_change'] = rsi.diff()
            features_df['rsi_slope_5'] = (rsi - rsi.shift(5)) / 5
            features_df['rsi_slope_10'] = (rsi - rsi.shift(10)) / 10
            
            # RSI divergence (simplified)
            price_change_5 = df['Close'].pct_change(5)
            rsi_change_5 = rsi.pct_change(5)
            features_df['rsi_price_divergence_5'] = np.sign(price_change_5) != np.sign(rsi_change_5)
            
            # RSI regime persistence
            features_df['rsi_oversold_days'] = self._calculate_regime_persistence(rsi < 30)
            features_df['rsi_overbought_days'] = self._calculate_regime_persistence(rsi > 70)
            
            # Multi-timeframe RSI (using different periods)
            rsi_7 = talib.RSI(df['Close'], timeperiod=7)
            rsi_21 = talib.RSI(df['Close'], timeperiod=21)
            features_df['rsi_7'] = rsi_7
            features_df['rsi_21'] = rsi_21
            features_df['rsi_alignment'] = ((rsi_7 > 50) & (rsi > 50) & (rsi_21 > 50)).astype(int)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding RSI features: {e}")
            return features_df
    
    def _add_adx_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX trend strength features"""
        try:
            # ADX calculation
            adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=self.feature_config['adx_period'])
            di_plus = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=self.feature_config['adx_period'])
            di_minus = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=self.feature_config['adx_period'])
            
            features_df['adx'] = adx
            features_df['di_plus'] = di_plus
            features_df['di_minus'] = di_minus
            features_df['di_spread'] = di_plus - di_minus
            
            # ADX regimes
            features_df['adx_trending'] = (adx > 25).astype(int)
            features_df['adx_strong_trend'] = (adx > 40).astype(int)
            features_df['adx_sideways'] = (adx < 20).astype(int)
            
            # Trend direction
            features_df['trend_bullish'] = (di_plus > di_minus).astype(int)
            features_df['trend_bearish'] = (di_minus > di_plus).astype(int)
            
            # ADX momentum
            features_df['adx_change'] = adx.diff()
            features_df['adx_increasing'] = (adx.diff() > 0).astype(int)
            features_df['adx_slope_5'] = (adx - adx.shift(5)) / 5
            
            # Trend persistence
            features_df['bullish_trend_days'] = self._calculate_regime_persistence(di_plus > di_minus)
            features_df['bearish_trend_days'] = self._calculate_regime_persistence(di_minus > di_plus)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding ADX features: {e}")
            return features_df
    
    def _add_atr_band_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add ATR volatility band features"""
        try:
            # ATR calculation
            atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=self.feature_config['atr_period'])
            features_df['atr'] = atr
            features_df['atr_pct'] = (atr / df['Close']) * 100
            
            # ATR bands
            sma_20 = talib.SMA(df['Close'], timeperiod=20)
            atr_upper_1 = sma_20 + atr
            atr_lower_1 = sma_20 - atr
            atr_upper_2 = sma_20 + (atr * 2)
            atr_lower_2 = sma_20 - (atr * 2)
            
            features_df['atr_upper_1'] = atr_upper_1
            features_df['atr_lower_1'] = atr_lower_1
            features_df['atr_upper_2'] = atr_upper_2
            features_df['atr_lower_2'] = atr_lower_2
            
            # Position relative to ATR bands
            features_df['price_vs_atr_upper_1'] = (df['Close'] / atr_upper_1) - 1
            features_df['price_vs_atr_lower_1'] = (df['Close'] / atr_lower_1) - 1
            features_df['atr_band_position'] = (df['Close'] - atr_lower_1) / (atr_upper_1 - atr_lower_1)
            
            # ATR regime classification
            atr_percentile = atr.rolling(100).rank(pct=True)
            features_df['atr_percentile'] = atr_percentile
            features_df['atr_low_vol'] = (atr_percentile < 0.2).astype(int)
            features_df['atr_high_vol'] = (atr_percentile > 0.8).astype(int)
            features_df['atr_normal_vol'] = ((atr_percentile >= 0.2) & (atr_percentile <= 0.8)).astype(int)
            
            # Volatility momentum
            features_df['atr_change'] = atr.pct_change()
            features_df['atr_expanding'] = (atr > atr.shift(5)).astype(int)
            features_df['atr_contracting'] = (atr < atr.shift(5)).astype(int)
            
            # Volatility clustering
            features_df['high_vol_cluster'] = self._calculate_regime_persistence(atr_percentile > 0.8)
            features_df['low_vol_cluster'] = self._calculate_regime_persistence(atr_percentile < 0.2)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding ATR band features: {e}")
            return features_df
    
    def _add_volume_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            # Volume moving averages
            volume_ma_20 = df['Volume'].rolling(self.feature_config['volume_ma_period']).mean()
            volume_ma_50 = df['Volume'].rolling(50).mean()
            
            features_df['volume'] = df['Volume']
            features_df['volume_ma_20'] = volume_ma_20
            features_df['volume_vs_ma_20'] = (df['Volume'] / volume_ma_20) - 1
            features_df['volume_vs_ma_50'] = (df['Volume'] / df['Volume'].rolling(50).mean()) - 1
            
            # Volume Z-score
            volume_std_20 = df['Volume'].rolling(20).std()
            features_df['volume_zscore'] = (df['Volume'] - volume_ma_20) / volume_std_20
            features_df['volume_spike'] = (features_df['volume_zscore'] > 2).astype(int)
            features_df['volume_dry_up'] = (features_df['volume_zscore'] < -1).astype(int)
            
            # On-Balance Volume
            obv = talib.OBV(df['Close'], df['Volume'])
            features_df['obv'] = obv
            features_df['obv_slope_20'] = (obv - obv.shift(20)) / 20
            features_df['obv_trending_up'] = (features_df['obv_slope_20'] > 0).astype(int)
            
            # Volume-Price Trend
            vpt = talib.VPT(df['Close'], df['Volume'])
            features_df['vpt'] = vpt
            features_df['vpt_change'] = vpt.pct_change()
            
            # Price-Volume patterns
            price_change = df['Close'].pct_change()
            volume_change = df['Volume'].pct_change()
            features_df['price_volume_correlation_10'] = price_change.rolling(10).corr(volume_change)
            
            # Volume breakout patterns
            features_df['volume_breakout'] = (
                (df['Volume'] > volume_ma_20 * 1.5) & 
                (price_change > 0.02)
            ).astype(int)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return features_df
    
    def _add_pattern_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features"""
        try:
            # Support and Resistance levels
            lookback = self.feature_config['support_resistance_window']
            
            # Rolling highs and lows
            rolling_high = df['High'].rolling(lookback).max()
            rolling_low = df['Low'].rolling(lookback).min()
            
            features_df['near_resistance'] = (df['Close'] / rolling_high > 0.98).astype(int)
            features_df['near_support'] = (df['Close'] / rolling_low < 1.02).astype(int)
            features_df['resistance_distance'] = (rolling_high / df['Close']) - 1
            features_df['support_distance'] = (df['Close'] / rolling_low) - 1
            
            # Candlestick patterns (using talib)
            features_df['hammer'] = talib.CDLHAMMER(df['Open'], df['High'], df['Low'], df['Close'])
            features_df['doji'] = talib.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
            features_df['engulfing'] = talib.CDLENGULFING(df['Open'], df['High'], df['Low'], df['Close'])
            features_df['morning_star'] = talib.CDLMORNINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
            features_df['evening_star'] = talib.CDLEVENINGSTAR(df['Open'], df['High'], df['Low'], df['Close'])
            
            # Gap patterns
            gap_up = df['Open'] > df['Close'].shift(1) * 1.02
            gap_down = df['Open'] < df['Close'].shift(1) * 0.98
            features_df['gap_up'] = gap_up.astype(int)
            features_df['gap_down'] = gap_down.astype(int)
            
            # Higher highs, higher lows pattern
            features_df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
            features_df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
            features_df['lower_high'] = (df['High'] < df['High'].shift(1)).astype(int)
            features_df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
            
            # Swing structure
            features_df['uptrend_structure'] = (
                features_df['higher_high'] & features_df['higher_low']
            ).astype(int)
            features_df['downtrend_structure'] = (
                features_df['lower_high'] & features_df['lower_low']
            ).astype(int)
            
            # Breakout patterns
            breakout_high = df['Close'] > rolling_high.shift(1)
            breakout_low = df['Close'] < rolling_low.shift(1)
            features_df['breakout_high'] = breakout_high.astype(int)
            features_df['breakout_low'] = breakout_low.astype(int)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding pattern features: {e}")
            return features_df
    
    def _add_market_regime_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features"""
        try:
            # Rolling volatility regimes
            returns = df['Close'].pct_change()
            vol_window = self.feature_config['volatility_window']
            
            rolling_vol = returns.rolling(vol_window).std() * np.sqrt(252)
            vol_percentile = rolling_vol.rolling(self.feature_config['regime_window']).rank(pct=True)
            
            features_df['volatility_regime'] = pd.cut(vol_percentile, bins=3, labels=[0, 1, 2])
            features_df['low_vol_regime'] = (vol_percentile < 0.33).astype(int)
            features_df['normal_vol_regime'] = ((vol_percentile >= 0.33) & (vol_percentile <= 0.67)).astype(int)
            features_df['high_vol_regime'] = (vol_percentile > 0.67).astype(int)
            
            # Trend regime
            sma_50 = talib.SMA(df['Close'], timeperiod=50)
            sma_200 = talib.SMA(df['Close'], timeperiod=200)
            
            features_df['bull_market'] = (sma_50 > sma_200).astype(int)
            features_df['bear_market'] = (sma_50 < sma_200).astype(int)
            
            # Market momentum
            momentum_20 = (df['Close'] / df['Close'].shift(20)) - 1
            momentum_60 = (df['Close'] / df['Close'].shift(60)) - 1
            
            features_df['momentum_regime_20'] = pd.cut(
                momentum_20.rolling(252).rank(pct=True), 
                bins=3, labels=[0, 1, 2]
            )
            features_df['momentum_regime_60'] = pd.cut(
                momentum_60.rolling(252).rank(pct=True), 
                bins=3, labels=[0, 1, 2]
            )
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding market regime features: {e}")
            return features_df
    
    def _add_cross_sectional_features(self, features_df: pd.DataFrame, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add cross-sectional features (relative to market/sector)"""
        try:
            # Placeholder for cross-sectional features
            # In a full implementation, this would compare against market indices and sector peers
            
            # Market cap tier (would need external data)
            features_df['large_cap'] = 1  # Placeholder
            features_df['mid_cap'] = 0
            features_df['small_cap'] = 0
            
            # Sector momentum (placeholder)
            features_df['sector_momentum'] = 0  # Would calculate relative to sector index
            features_df['vs_market_beta'] = 1.0  # Would calculate rolling beta vs market
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding cross-sectional features: {e}")
            return features_df
    
    def _add_target_variables(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variables for ML training"""
        try:
            # Forward returns (targets for prediction)
            for horizon in [5, 10, 15, 20]:
                forward_return = (df['Close'].shift(-horizon) / df['Close']) - 1
                features_df[f'target_return_{horizon}d'] = forward_return
                
                # Binary classification targets
                features_df[f'target_positive_{horizon}d'] = (forward_return > 0).astype(int)
                features_df[f'target_strong_positive_{horizon}d'] = (forward_return > 0.05).astype(int)
                
                # R-multiple targets (assuming 2% stop-loss)
                stop_loss_pct = 0.02
                r_multiple = forward_return / stop_loss_pct
                features_df[f'target_r_multiple_{horizon}d'] = r_multiple
                features_df[f'target_r_positive_{horizon}d'] = (r_multiple > 1.0).astype(int)
            
            # Maximum favorable excursion (MFE) and Maximum adverse excursion (MAE)
            for horizon in [10, 20]:
                if len(df) >= horizon:
                    future_highs = df['High'].rolling(horizon).max().shift(-horizon)
                    future_lows = df['Low'].rolling(horizon).min().shift(-horizon)
                    
                    mfe = (future_highs / df['Close']) - 1
                    mae = 1 - (future_lows / df['Close'])
                    
                    features_df[f'target_mfe_{horizon}d'] = mfe
                    features_df[f'target_mae_{horizon}d'] = mae
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error adding target variables: {e}")
            return features_df
    
    def _calculate_regime_persistence(self, condition: pd.Series) -> pd.Series:
        """Calculate how many consecutive days a condition has been true"""
        try:
            groups = (condition != condition.shift()).cumsum()
            return condition.groupby(groups).cumsum()
            
        except Exception as e:
            logger.error(f"Error calculating regime persistence: {e}")
            return pd.Series(0, index=condition.index)
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML"""
        try:
            # Handle infinite values
            features_df = features_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate methods
            numeric_columns = features_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col.startswith('target_'):
                    # Don't forward fill targets
                    continue
                elif 'regime' in col or 'pattern' in col:
                    # Forward fill regime and pattern flags
                    features_df[col] = features_df[col].fillna(method='ffill')
                else:
                    # Forward fill then backward fill for other features
                    features_df[col] = features_df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Drop rows where targets are NaN (can't train on these)
            target_cols = [col for col in features_df.columns if col.startswith('target_')]
            if target_cols:
                features_df = features_df.dropna(subset=target_cols, how='all')
            
            return features_df
            
        except Exception as e:
            logger.error(f"Error cleaning features: {e}")
            return features_df
    
    def get_feature_names(self, include_targets: bool = False) -> List[str]:
        """Get list of feature names"""
        feature_categories = [
            # Price and returns
            ['price', 'price_log', 'return_1d', 'return_5d', 'return_10d', 'return_1d_ma_5', 'return_1d_ma_20', 
             'return_1d_std_20', 'price_momentum_5', 'price_momentum_20', 'price_momentum_60'],
            
            # Technical indicators
            ['sma_20', 'sma_50', 'sma_200', 'price_vs_sma_20', 'price_vs_sma_50', 'price_vs_sma_200',
             'sma_20_vs_50', 'sma_50_vs_200', 'macd', 'macd_signal', 'macd_histogram', 'macd_above_signal',
             'macd_above_zero', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position'],
            
            # RSI features
            ['rsi', 'rsi_oversold', 'rsi_overbought', 'rsi_neutral', 'rsi_change', 'rsi_slope_5', 'rsi_slope_10',
             'rsi_price_divergence_5', 'rsi_oversold_days', 'rsi_overbought_days', 'rsi_7', 'rsi_21', 'rsi_alignment'],
            
            # ADX features
            ['adx', 'di_plus', 'di_minus', 'di_spread', 'adx_trending', 'adx_strong_trend', 'adx_sideways',
             'trend_bullish', 'trend_bearish', 'adx_change', 'adx_increasing', 'adx_slope_5', 'bullish_trend_days',
             'bearish_trend_days'],
            
            # ATR features
            ['atr', 'atr_pct', 'atr_upper_1', 'atr_lower_1', 'atr_upper_2', 'atr_lower_2', 'price_vs_atr_upper_1',
             'price_vs_atr_lower_1', 'atr_band_position', 'atr_percentile', 'atr_low_vol', 'atr_high_vol',
             'atr_normal_vol', 'atr_change', 'atr_expanding', 'atr_contracting', 'high_vol_cluster', 'low_vol_cluster'],
            
            # Volume features
            ['volume', 'volume_ma_20', 'volume_vs_ma_20', 'volume_vs_ma_50', 'volume_zscore', 'volume_spike',
             'volume_dry_up', 'obv', 'obv_slope_20', 'obv_trending_up', 'vpt', 'vpt_change', 'price_volume_correlation_10',
             'volume_breakout'],
            
            # Pattern features
            ['near_resistance', 'near_support', 'resistance_distance', 'support_distance', 'hammer', 'doji',
             'engulfing', 'morning_star', 'evening_star', 'gap_up', 'gap_down', 'higher_high', 'higher_low',
             'lower_high', 'lower_low', 'uptrend_structure', 'downtrend_structure', 'breakout_high', 'breakout_low'],
            
            # Market regime features
            ['volatility_regime', 'low_vol_regime', 'normal_vol_regime', 'high_vol_regime', 'bull_market',
             'bear_market', 'momentum_regime_20', 'momentum_regime_60'],
            
            # Cross-sectional features
            ['large_cap', 'mid_cap', 'small_cap', 'sector_momentum', 'vs_market_beta']
        ]
        
        all_features = [feature for category in feature_categories for feature in category]
        
        if include_targets:
            target_features = [
                'target_return_5d', 'target_return_10d', 'target_return_15d', 'target_return_20d',
                'target_positive_5d', 'target_positive_10d', 'target_positive_15d', 'target_positive_20d',
                'target_strong_positive_5d', 'target_strong_positive_10d', 'target_strong_positive_15d', 'target_strong_positive_20d',
                'target_r_multiple_5d', 'target_r_multiple_10d', 'target_r_multiple_15d', 'target_r_multiple_20d',
                'target_r_positive_5d', 'target_r_positive_10d', 'target_r_positive_15d', 'target_r_positive_20d',
                'target_mfe_10d', 'target_mfe_20d', 'target_mae_10d', 'target_mae_20d'
            ]
            all_features.extend(target_features)
        
        return all_features


def main():
    """Main function for testing feature extraction"""
    # Create sample data for testing
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = [base_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices[:-1]],
        'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices[:-1]],
        'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices[:-1]],
        'Close': prices[:-1],
        'Volume': np.random.randint(10000, 1000000, len(dates))
    })
    df.set_index('Date', inplace=True)
    
    # Test feature extraction
    extractor = MLFeatureExtractor()
    features = extractor.extract_features(df, 'TEST_STOCK')
    
    print(f"Generated {len(features.columns)} features for {len(features)} observations")
    print("\nFeature categories:")
    feature_names = extractor.get_feature_names(include_targets=True)
    print(f"Total available features: {len(feature_names)}")
    
    # Show sample of features
    print("\nFirst 10 features:")
    print(features.iloc[-10:, :15])  # Last 10 rows, first 15 columns


if __name__ == "__main__":
    main()
