#!/usr/bin/env python3
"""
Offline Stock Analysis Script
Analyzes stocks using only cached data without making any API calls.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import glob

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging
from config import NSE_CACHE_FILE

logger = setup_logging(verbose=True)

class OfflineStockAnalyzer:
    """Analyzer that works only with cached/offline data."""
    
    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        self.results = []
        
    def get_cached_symbols(self, max_stocks: Optional[int] = None) -> List[str]:
        """Get list of symbols that have cached data."""
        symbols = set()  # Use set to avoid duplicates
        
        # Look for cached CSV files with different periods
        patterns = [
            '*_1y_1d.csv',
            '*_2y_1d.csv', 
            '*_3mo_1d.csv',
            '*_6mo_1d.csv'
        ]
        
        for pattern in patterns:
            csv_pattern = os.path.join(self.cache_dir, pattern)
            csv_files = glob.glob(csv_pattern)
            
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                symbol = filename.split('_')[0]
                if symbol and len(symbol) <= 20:  # Valid symbol length
                    symbols.add(symbol)
        
        symbols = sorted(list(symbols))  # Convert to sorted list
        logger.info(f"Found {len(symbols)} symbols with cached data")
        
        if max_stocks and len(symbols) > max_stocks:
            symbols = symbols[:max_stocks]
            logger.info(f"Limited to {max_stocks} symbols")
            
        return symbols
    
    def load_cached_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> Optional[pd.DataFrame]:
        """Load cached data for a symbol."""
        # Try different periods in order of preference
        periods_to_try = ['2y', '1y', '6mo', '3mo']
        
        for period_try in periods_to_try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_{period_try}_{interval}.csv")
            if os.path.exists(cache_file):
                period = period_try
                break
        else:
            # No cache file found for any period
            return None
            
        try:
            # Try different index column names
            for index_col in ['Date', 'Datetime', 0]:
                try:
                    data = pd.read_csv(cache_file, index_col=index_col, parse_dates=True)
                    if not data.empty:
                        logger.debug(f"Loaded {len(data)} rows for {symbol}")
                        return data
                except (KeyError, ValueError):
                    continue
                    
            # If all attempts fail, read without index
            data = pd.read_csv(cache_file, parse_dates=True)
            if not data.empty and len(data.columns) > 0:
                # Set first column as index if it looks like a date
                first_col = data.columns[0]
                if 'date' in first_col.lower():
                    data.set_index(first_col, inplace=True)
                return data
                
        except Exception as e:
            logger.error(f"Error loading cached data for {symbol}: {e}")
            
        return None
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic technical indicators."""
        indicators = {}
        
        try:
            if 'Close' not in data.columns:
                return indicators
                
            close_prices = data['Close']
            
            # Simple Moving Averages
            indicators['sma_20'] = close_prices.rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = close_prices.rolling(window=50).mean().iloc[-1]
            indicators['sma_200'] = close_prices.rolling(window=200).mean().iloc[-1] if len(data) >= 200 else None
            
            # Current price
            indicators['current_price'] = close_prices.iloc[-1]
            
            # Price change
            indicators['price_change_1d'] = ((close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2] * 100) if len(data) > 1 else 0
            indicators['price_change_5d'] = ((close_prices.iloc[-1] - close_prices.iloc[-5]) / close_prices.iloc[-5] * 100) if len(data) > 5 else 0
            indicators['price_change_30d'] = ((close_prices.iloc[-1] - close_prices.iloc[-30]) / close_prices.iloc[-30] * 100) if len(data) > 30 else 0
            
            # RSI
            if len(data) > 14:
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Volume
            if 'Volume' in data.columns:
                indicators['avg_volume'] = data['Volume'].mean()
                indicators['volume_ratio'] = data['Volume'].iloc[-1] / indicators['avg_volume'] if indicators['avg_volume'] > 0 else 1
                
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            
        return indicators
    
    def generate_recommendation(self, symbol: str, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a simple recommendation based on indicators."""
        recommendation = {
            'symbol': symbol,
            'current_price': indicators.get('current_price', 0),
            'technical_score': 0,
            'signals': []
        }
        
        score = 50  # Neutral base score
        
        # Moving average signals
        current_price = indicators.get('current_price', 0)
        if current_price and indicators.get('sma_20'):
            if current_price > indicators['sma_20']:
                score += 10
                recommendation['signals'].append('Price above SMA20')
        
        if current_price and indicators.get('sma_50'):
            if current_price > indicators['sma_50']:
                score += 10
                recommendation['signals'].append('Price above SMA50')
                
        # RSI signals
        rsi = indicators.get('rsi')
        if rsi:
            if rsi < 30:
                score += 20
                recommendation['signals'].append(f'Oversold (RSI={rsi:.1f})')
            elif rsi > 70:
                score -= 20
                recommendation['signals'].append(f'Overbought (RSI={rsi:.1f})')
            else:
                recommendation['signals'].append(f'RSI neutral ({rsi:.1f})')
        
        # Volume signals
        volume_ratio = indicators.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 5
            recommendation['signals'].append(f'High volume ({volume_ratio:.1f}x avg)')
        
        # Price momentum
        price_change_5d = indicators.get('price_change_5d', 0)
        if price_change_5d > 5:
            score += 10
            recommendation['signals'].append(f'Strong 5d momentum ({price_change_5d:.1f}%)')
        elif price_change_5d < -5:
            score -= 10
            recommendation['signals'].append(f'Weak 5d momentum ({price_change_5d:.1f}%)')
            
        recommendation['technical_score'] = min(100, max(0, score))
        
        # Determine recommendation strength
        if score >= 70:
            recommendation['recommendation'] = 'BUY'
            recommendation['strength'] = 'Strong' if score >= 80 else 'Moderate'
        elif score <= 30:
            recommendation['recommendation'] = 'SELL'
            recommendation['strength'] = 'Strong' if score <= 20 else 'Moderate'
        else:
            recommendation['recommendation'] = 'HOLD'
            recommendation['strength'] = 'Neutral'
            
        return recommendation
    
    def analyze_stock(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single stock using cached data."""
        logger.info(f"Analyzing {symbol} from cache...")
        
        # Load cached data
        data = self.load_cached_data(symbol)
        if data is None or data.empty:
            logger.warning(f"No cached data available for {symbol}")
            return None
            
        # Calculate indicators
        indicators = self.calculate_technical_indicators(data)
        
        # Generate recommendation
        recommendation = self.generate_recommendation(symbol, indicators)
        
        # Add metadata
        recommendation['data_points'] = len(data)
        recommendation['last_update'] = str(data.index[-1]) if hasattr(data, 'index') else 'Unknown'
        recommendation['indicators'] = indicators
        
        return recommendation
    
    def run_analysis(self, max_stocks: Optional[int] = None):
        """Run offline analysis on all cached stocks."""
        logger.info("Starting offline stock analysis...")
        
        # Get symbols with cached data
        symbols = self.get_cached_symbols(max_stocks)
        
        if not symbols:
            logger.error("No cached data found. Please run online analysis first to build cache.")
            return
            
        logger.info(f"Analyzing {len(symbols)} stocks with cached data...")
        
        buy_recommendations = []
        hold_recommendations = []
        sell_recommendations = []
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"[{i}/{len(symbols)}] Analyzing {symbol}...")
            
            result = self.analyze_stock(symbol)
            
            if result:
                self.results.append(result)
                
                if result['recommendation'] == 'BUY':
                    buy_recommendations.append(result)
                elif result['recommendation'] == 'HOLD':
                    hold_recommendations.append(result)
                else:
                    sell_recommendations.append(result)
                    
                # Log the result
                logger.info(f"  {symbol}: {result['recommendation']} ({result['strength']}) - "
                          f"Score: {result['technical_score']}, Price: {result['current_price']:.2f}")
                
                if result['signals']:
                    for signal in result['signals'][:3]:  # Show top 3 signals
                        logger.info(f"    - {signal}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Total stocks analyzed: {len(self.results)}")
        logger.info(f"BUY recommendations: {len(buy_recommendations)}")
        logger.info(f"HOLD recommendations: {len(hold_recommendations)}")
        logger.info(f"SELL recommendations: {len(sell_recommendations)}")
        
        if buy_recommendations:
            logger.info("\nTop BUY Recommendations:")
            # Sort by technical score
            buy_recommendations.sort(key=lambda x: x['technical_score'], reverse=True)
            for rec in buy_recommendations[:10]:  # Top 10
                logger.info(f"  {rec['symbol']}: Score={rec['technical_score']}, "
                          f"Price={rec['current_price']:.2f}, Strength={rec['strength']}")
        
        # Save results to JSON
        self.save_results()
        
    def save_results(self):
        """Save analysis results to a JSON file."""
        if not self.results:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"offline_analysis_results_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'total_analyzed': len(self.results),
                    'results': self.results
                }, f, indent=2, default=str)
                
            logger.info(f"\nResults saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Offline Stock Analysis using cached data')
    parser.add_argument('--max-stocks', type=int, default=None,
                       help='Maximum number of stocks to analyze')
    
    args = parser.parse_args()
    
    analyzer = OfflineStockAnalyzer()
    analyzer.run_analysis(max_stocks=args.max_stocks)

if __name__ == '__main__':
    main()
