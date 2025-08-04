"""
Sector Analysis Module
File: scripts/sector_analysis.py

This module implements sector momentum and rotation analysis for better
stock selection and market timing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from scripts.data_fetcher import get_historical_data
from utils.logger import setup_logging

logger = setup_logging()

class SectorAnalyzer:
    """
    Analyze sector momentum and rotation patterns.
    """
    
    def __init__(self):
        """Initialize the sector analyzer."""
        # NSE sector mapping (simplified)
        self.sector_mapping = {
            'Technology': ['TCS', 'INFY', 'WIPRO', 'HCLTECH', 'TECHM', 'MINDTREE'],
            'Banking': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'SBIN', 'AXISBANK', 'INDUSINDBK'],
            'Energy': ['RELIANCE', 'ONGC', 'GAIL', 'NTPC', 'POWERGRID', 'COALINDIA'],
            'Consumer': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR', 'MARICO'],
            'Pharmaceuticals': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'AUROPHARMA', 'DIVISLAB'],
            'Automotive': ['MARUTI', 'TATAMOTORS', 'BAJAJ-AUTO', 'MAHINDRA', 'EICHERMOT', 'HEROMOTOCO'],
            'Metals': ['TATASTEEL', 'HINDALCO', 'VEDL', 'JSWSTEEL', 'SAIL', 'NMDC'],
            'Telecom': ['BHARTIARTL', 'IDEA', 'RCOM'],
            'Cement': ['ULTRATECH', 'SHREECEM', 'ACC', 'AMBUJA', 'JKCEMENT'],
            'Real Estate': ['DLF', 'GODREJPROP', 'SOBHA', 'PRESTIGE', 'BRIGADE']
        }
        
    def analyze_sector_momentum(self, period: str = '3mo') -> Dict[str, Any]:
        """
        Analyze momentum across different sectors.
        
        Args:
            period: Time period for analysis
            
        Returns:
            Dictionary with sector momentum analysis
        """
        try:
            sector_performance = {}
            
            for sector, symbols in self.sector_mapping.items():
                logger.info(f"Analyzing sector momentum for {sector}")
                
                sector_returns = []
                valid_symbols = []
                
                for symbol in symbols:
                    try:
                        data = get_historical_data(symbol, period)
                        if not data.empty and len(data) > 1:
                            # Calculate return
                            start_price = data['Close'].iloc[0]
                            end_price = data['Close'].iloc[-1]
                            return_pct = ((end_price - start_price) / start_price) * 100
                            
                            sector_returns.append(return_pct)
                            valid_symbols.append(symbol)
                            
                    except Exception as e:
                        logger.warning(f"Error getting data for {symbol}: {e}")
                        continue
                
                if sector_returns:
                    # Calculate sector metrics
                    avg_return = np.mean(sector_returns)
                    median_return = np.median(sector_returns)
                    volatility = np.std(sector_returns)
                    
                    # Count positive performers
                    positive_count = sum(1 for r in sector_returns if r > 0)
                    total_count = len(sector_returns)
                    
                    sector_performance[sector] = {
                        'average_return': avg_return,
                        'median_return': median_return,
                        'volatility': volatility,
                        'positive_ratio': positive_count / total_count,
                        'total_stocks': total_count,
                        'valid_symbols': valid_symbols,
                        'momentum_score': self._calculate_momentum_score(avg_return, positive_count / total_count, volatility)
                    }
            
            # Rank sectors by momentum
            ranked_sectors = sorted(sector_performance.items(), 
                                  key=lambda x: x[1]['momentum_score'], 
                                  reverse=True)
            
            return {
                'sector_performance': sector_performance,
                'ranked_sectors': ranked_sectors,
                'period': period,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sector momentum analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_momentum_score(self, avg_return: float, positive_ratio: float, volatility: float) -> float:
        """
        Calculate a momentum score for a sector.
        
        Args:
            avg_return: Average return of sector
            positive_ratio: Ratio of positive performing stocks
            volatility: Volatility of sector returns
            
        Returns:
            Momentum score
        """
        # Normalize components with improved scaling
        return_score = max(0, min(1, (avg_return + 30) / 60))  # Wider range for better differentiation
        consistency_score = positive_ratio  # Already 0-1
        volatility_penalty = max(0, 1 - (volatility / 40))  # More sensitive to volatility
        
        # Enhanced weighted combination with momentum bias
        momentum_score = (return_score * 0.6 + consistency_score * 0.25 + volatility_penalty * 0.15)
        
        return momentum_score
    
    def calculate_sector_relative_strength(self, sector_returns: Dict[str, float], 
                                         benchmark_return: float = None) -> Dict[str, float]:
        """
        Calculate relative strength of sectors against a benchmark.
        
        Args:
            sector_returns: Dictionary of sector returns
            benchmark_return: Benchmark return (market average if None)
            
        Returns:
            Dictionary of relative strength scores
        """
        if benchmark_return is None:
            # Use average of all sectors as benchmark
            benchmark_return = np.mean(list(sector_returns.values()))
        
        relative_strength = {}
        for sector, return_pct in sector_returns.items():
            # Calculate relative strength ratio
            if benchmark_return != 0:
                rs_ratio = return_pct / benchmark_return if benchmark_return > 0 else (return_pct - benchmark_return)
            else:
                rs_ratio = 1.0 if return_pct >= 0 else -1.0
            
            # Normalize to 0-1 scale with 0.5 as neutral
            if rs_ratio >= 1:
                relative_strength[sector] = 0.5 + min(0.5, (rs_ratio - 1) * 0.5)
            else:
                relative_strength[sector] = 0.5 * rs_ratio
            
        return relative_strength
    
    def detect_sector_rotation_signals(self, rotation_analysis: Dict[str, Any]) -> Dict[str, str]:
        """
        Detect sector rotation signals based on momentum changes.
        
        Args:
            rotation_analysis: Historical rotation analysis data
            
        Returns:
            Dictionary of sector rotation signals
        """
        signals = {}
        
        if len(rotation_analysis) < 2:
            return signals
        
        # Get the two most recent periods for comparison
        periods = sorted(rotation_analysis.keys())
        if len(periods) >= 2:
            recent_period = periods[-1]
            previous_period = periods[-2]
            
            recent_tops = [sector for sector, score in rotation_analysis[recent_period]['top_sectors']]
            previous_tops = [sector for sector, score in rotation_analysis[previous_period]['top_sectors']]
            
            recent_bottoms = [sector for sector, score in rotation_analysis[recent_period]['bottom_sectors']]
            previous_bottoms = [sector for sector, score in rotation_analysis[previous_period]['bottom_sectors']]
            
            # Detect emerging leaders
            emerging_leaders = [sector for sector in recent_tops if sector not in previous_tops]
            
            # Detect declining sectors
            declining_sectors = [sector for sector in recent_bottoms if sector not in previous_bottoms]
            
            # Detect sector rotation (from bottom to top)
            rotating_up = [sector for sector in recent_tops if sector in previous_bottoms]
            
            # Assign signals
            for sector in emerging_leaders:
                signals[sector] = 'EMERGING_LEADER'
            
            for sector in declining_sectors:
                signals[sector] = 'DECLINING'
            
            for sector in rotating_up:
                signals[sector] = 'ROTATING_UP'
        
        return signals
    
    def get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """
        Get the sector for a given symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Sector name or None if not found
        """
        for sector, symbols in self.sector_mapping.items():
            if symbol in symbols:
                return sector
        return None
    
    def analyze_sector_rotation(self, periods: List[str] = ['1mo', '3mo', '6mo']) -> Dict[str, Any]:
        """
        Analyze sector rotation patterns across multiple time periods.
        
        Args:
            periods: List of time periods to analyze
            
        Returns:
            Dictionary with sector rotation analysis
        """
        try:
            rotation_analysis = {}
            
            for period in periods:
                momentum_data = self.analyze_sector_momentum(period)
                
                if 'ranked_sectors' in momentum_data:
                    # Extract top and bottom sectors
                    top_3_sectors = momentum_data['ranked_sectors'][:3]
                    bottom_3_sectors = momentum_data['ranked_sectors'][-3:]
                    
                    rotation_analysis[period] = {
                        'top_sectors': [(sector, data['momentum_score']) for sector, data in top_3_sectors],
                        'bottom_sectors': [(sector, data['momentum_score']) for sector, data in bottom_3_sectors],
                        'sector_count': len(momentum_data['ranked_sectors'])
                    }
            
            # Identify consistent performers
            consistent_performers = self._identify_consistent_performers(rotation_analysis)
            
            return {
                'rotation_analysis': rotation_analysis,
                'consistent_performers': consistent_performers,
                'analysis_timestamp': pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in sector rotation analysis: {e}")
            return {'error': str(e)}
    
    def _identify_consistent_performers(self, rotation_analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Identify sectors that consistently perform well or poorly.
        
        Args:
            rotation_analysis: Rotation analysis data
            
        Returns:
            Dictionary with consistent performers
        """
        sector_appearances = {}
        
        # Count appearances in top/bottom sectors
        for period, data in rotation_analysis.items():
            for sector, score in data['top_sectors']:
                if sector not in sector_appearances:
                    sector_appearances[sector] = {'top': 0, 'bottom': 0}
                sector_appearances[sector]['top'] += 1
            
            for sector, score in data['bottom_sectors']:
                if sector not in sector_appearances:
                    sector_appearances[sector] = {'top': 0, 'bottom': 0}
                sector_appearances[sector]['bottom'] += 1
        
        # Identify consistent performers
        consistent_top = [sector for sector, counts in sector_appearances.items() 
                         if counts['top'] >= 2 and counts['bottom'] == 0]
        consistent_bottom = [sector for sector, counts in sector_appearances.items() 
                           if counts['bottom'] >= 2 and counts['top'] == 0]
        
        return {
            'consistent_top_performers': consistent_top,
            'consistent_bottom_performers': consistent_bottom,
            'sector_appearances': sector_appearances
        }
    
    def get_sector_recommendation(self, symbol: str) -> Dict[str, Any]:
        """
        Get sector-based recommendation for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with sector recommendation
        """
        try:
            sector = self.get_sector_for_symbol(symbol)
            
            if not sector:
                return {
                    'sector': 'Unknown',
                    'sector_momentum': 'Unknown',
                    'recommendation': 'Neutral'
                }
            
            # Get current sector momentum
            momentum_data = self.analyze_sector_momentum()
            
            if 'sector_performance' in momentum_data and sector in momentum_data['sector_performance']:
                sector_data = momentum_data['sector_performance'][sector]
                momentum_score = sector_data['momentum_score']
                
                # Determine recommendation based on momentum
                if momentum_score > 0.7:
                    recommendation = 'Strong Sector Momentum - Favorable'
                elif momentum_score > 0.5:
                    recommendation = 'Moderate Sector Momentum - Neutral'
                else:
                    recommendation = 'Weak Sector Momentum - Caution'
                
                return {
                    'sector': sector,
                    'momentum_score': momentum_score,
                    'average_return': sector_data['average_return'],
                    'positive_ratio': sector_data['positive_ratio'],
                    'recommendation': recommendation,
                    'sector_rank': self._get_sector_rank(sector, momentum_data['ranked_sectors'])
                }
            
            return {
                'sector': sector,
                'sector_momentum': 'Data unavailable',
                'recommendation': 'Neutral'
            }
            
        except Exception as e:
            logger.error(f"Error getting sector recommendation for {symbol}: {e}")
            return {
                'sector': 'Unknown',
                'error': str(e),
                'recommendation': 'Neutral'
            }
    
    def _get_sector_rank(self, sector: str, ranked_sectors: List[tuple]) -> int:
        """Get the rank of a sector in the momentum ranking."""
        for i, (sector_name, data) in enumerate(ranked_sectors, 1):
            if sector_name == sector:
                return i
        return len(ranked_sectors)  # Return last rank if not found
    
    def get_comprehensive_sector_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive sector analysis for a symbol, including momentum,
        relative strength, and rotation signals.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with comprehensive sector analysis
        """
        try:
            sector = self.get_sector_for_symbol(symbol)

            if not sector:
                return {
                    'sector': 'Unknown',
                    'sector_score': 0.0,
                    'recommendation': 'Neutral - Sector not identified',
                    'error': 'Sector not found for symbol'
                }

            # 1. Momentum Analysis
            momentum_data = self.analyze_sector_momentum()
            if 'error' in momentum_data:
                raise Exception(f"Momentum analysis failed: {momentum_data['error']}")
            
            sector_performance = momentum_data.get('sector_performance', {}).get(sector, {})
            momentum_score = sector_performance.get('momentum_score', 0.5)

            # 2. Relative Strength Analysis
            all_sector_returns = {s: p.get('average_return', 0) for s, p in momentum_data.get('sector_performance', {}).items()}
            relative_strength_scores = self.calculate_sector_relative_strength(all_sector_returns)
            relative_strength = relative_strength_scores.get(sector, 0.5)

            # 3. Rotation Signal Analysis
            rotation_analysis = self.analyze_sector_rotation()
            rotation_signals = self.detect_sector_rotation_signals(rotation_analysis.get('rotation_analysis', {}))
            rotation_signal = rotation_signals.get(sector, 'NEUTRAL')

            # 4. Combine into a final sector score (-1 to 1)
            # Higher weight for momentum, then relative strength, then rotation
            sector_score = (momentum_score * 0.6) + (relative_strength * 0.3) + \
                           {'EMERGING_LEADER': 0.2, 'ROTATING_UP': 0.1, 'DECLINING': -0.2, 'NEUTRAL': 0.0}.get(rotation_signal, 0.0)
            
            # Normalize to -1 to 1
            sector_score = (sector_score * 2) - 1

            # 5. Generate a descriptive recommendation
            if sector_score > 0.4:
                recommendation = f"Very Strong Sector ({sector}): Favorable momentum, relative strength, and rotation."
            elif sector_score > 0.1:
                recommendation = f"Strong Sector ({sector}): Positive momentum and relative strength."
            elif sector_score < -0.4:
                recommendation = f"Weak Sector ({sector}): Significant underperformance and negative momentum."
            elif sector_score < -0.1:
                recommendation = f"Weakening Sector ({sector}): Caution advised due to lagging performance."
            else:
                recommendation = f"Neutral Sector ({sector}): No strong tailwinds or headwinds."


            return {
                'sector': sector,
                'sector_score': round(sector_score, 3),
                'recommendation': recommendation,
                'momentum_analysis': {
                    'momentum_score': round(momentum_score, 3),
                    'rank': self._get_sector_rank(sector, momentum_data.get('ranked_sectors', [])),
                    'total_sectors': len(momentum_data.get('ranked_sectors', [])),
                },
                'relative_strength': round(relative_strength, 3),
                'rotation_signal': rotation_signal
            }

        except Exception as e:
            logger.error(f"Error in comprehensive sector analysis for {symbol}: {e}")
            return {
                'sector': 'Unknown',
                'sector_score': 0.0,
                'recommendation': 'Neutral - Analysis error',
                'error': str(e)
            }
    
    def get_sector_summary(self) -> Dict[str, Any]:
        """
        Get a summary of sector analysis capabilities.
        
        Returns:
            Dictionary with sector analysis summary
        """
        return {
            'total_sectors': len(self.sector_mapping),
            'sectors': list(self.sector_mapping.keys()),
            'total_stocks_covered': sum(len(symbols) for symbols in self.sector_mapping.values()),
            'analysis_features': [
                'Sector Momentum Analysis',
                'Sector Rotation Patterns',
                'Consistent Performer Identification',
                'Sector-based Recommendations'
            ]
        }
