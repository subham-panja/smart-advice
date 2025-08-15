"""
Fundamental Analysis Module
===========================

Handles fundamental analysis of stocks including financial metrics.
Extracted from analyzer.py for better organization.

SWING TRADING ENHANCED: Fundamentals used as quality filter and tie-breaker.
Focuses on EPS growth, debt/equity ratio, and stable margins.
"""

from typing import Dict, Any, Optional
import yfinance as yf
from utils.logger import setup_logging

logger = setup_logging()

class FundamentalAnalyzer:
    """
    Performs fundamental analysis using financial metrics.
    Enhanced for swing trading with quality filters.
    """

    def __init__(self):
        """Initialize the fundamental analyzer."""
        # Quality thresholds for swing trading
        self.quality_thresholds = {
            'min_eps_growth': 0.05,      # 5% EPS growth YoY
            'max_debt_to_equity': 1.5,   # Max 1.5x debt/equity
            'min_profit_margin': 0.05,   # Min 5% profit margin
            'min_roe': 0.10,             # Min 10% ROE
            'max_pe_ratio': 40,          # Max P/E of 40
            'min_current_ratio': 1.0     # Min current ratio of 1.0
        }

    def perform_fundamental_analysis(self, symbol: str) -> Dict[str, Any]:
        """
        Perform fundamental analysis for the given stock symbol.
        Enhanced with quality filters for swing trading.

        Args:
            symbol: Stock symbol to analyze
        
        Returns:
            Dictionary containing score and detailed metrics
        """
        try:
            logger.info(f"Performing fundamental analysis for {symbol}")
            
            # Get financial metrics
            metrics = self.get_financial_metrics(symbol)
            
            if not metrics:
                logger.warning(f"No fundamental data available for {symbol}")
                return {'score': 0.1, 'details': {}}  # Default neutral score
            
            # Calculate quality score based on multiple factors
            score_components = []
            
            # EPS Growth (weight: 0.3)
            if metrics.get('eps_growth') is not None:
                eps_score = self._score_eps_growth(metrics['eps_growth'])
                score_components.append(('eps_growth', eps_score, 0.3))
                logger.debug(f"{symbol}: EPS growth score = {eps_score:.2f}")
            
            # Debt to Equity (weight: 0.2)
            if metrics.get('debt_to_equity') is not None:
                debt_score = self._score_debt_to_equity(metrics['debt_to_equity'])
                score_components.append(('debt_to_equity', debt_score, 0.2))
                logger.debug(f"{symbol}: Debt/Equity score = {debt_score:.2f}")
            
            # Profit Margin (weight: 0.2)
            if metrics.get('profit_margin') is not None:
                margin_score = self._score_profit_margin(metrics['profit_margin'])
                score_components.append(('profit_margin', margin_score, 0.2))
                logger.debug(f"{symbol}: Profit margin score = {margin_score:.2f}")
            
            # ROE (weight: 0.15)
            if metrics.get('roe') is not None:
                roe_score = self._score_roe(metrics['roe'])
                score_components.append(('roe', roe_score, 0.15))
                logger.debug(f"{symbol}: ROE score = {roe_score:.2f}")
            
            # P/E Ratio (weight: 0.15)
            if metrics.get('pe_ratio') is not None:
                pe_score = self._score_pe_ratio(metrics['pe_ratio'])
                score_components.append(('pe_ratio', pe_score, 0.15))
                logger.debug(f"{symbol}: P/E score = {pe_score:.2f}")
            
            # Calculate weighted average score
            if score_components:
                total_weight = sum(weight for _, _, weight in score_components)
                weighted_sum = sum(score * weight for _, score, weight in score_components)
                fundamental_score = weighted_sum / total_weight if total_weight > 0 else 0.1
            else:
                fundamental_score = 0.1  # Default neutral score
            
            # Log the final score with reasoning
            logger.info(f"{symbol}: Fundamental score = {fundamental_score:.3f} "
                       f"(components: {len(score_components)})")
            
            # Return both score and details for tie-breaker decisions
            return {
                'score': max(0.0, min(1.0, fundamental_score)),  # Clamp to [0, 1]
                'details': {
                    'eps_growth': metrics.get('eps_growth'),
                    'de_ratio': metrics.get('debt_to_equity'),
                    'profit_margins': metrics.get('profit_margin'),
                    'roe': metrics.get('roe'),
                    'pe_ratio': metrics.get('pe_ratio'),
                    'pb_ratio': metrics.get('pb_ratio'),
                    'revenue_growth': metrics.get('revenue_growth'),
                    'dividend_yield': metrics.get('dividend_yield'),
                    'current_ratio': metrics.get('current_ratio'),
                    'market_cap': metrics.get('market_cap'),
                    'beta': metrics.get('beta')
                }
            }
            
        except Exception as e:
            logger.error(f"Error during fundamental analysis for {symbol}: {e}")
            return {'score': 0.1, 'details': {}}  # Default neutral positive score
    
    def _score_eps_growth(self, eps_growth: float) -> float:
        """Score EPS growth rate."""
        if eps_growth >= 0.20:  # 20%+ growth
            return 1.0
        elif eps_growth >= 0.10:  # 10-20% growth
            return 0.8
        elif eps_growth >= self.quality_thresholds['min_eps_growth']:
            return 0.6
        elif eps_growth >= 0:
            return 0.4
        else:
            return 0.2  # Negative growth
    
    def _score_debt_to_equity(self, debt_to_equity: float) -> float:
        """Score debt to equity ratio (lower is better)."""
        if debt_to_equity < 0.3:
            return 1.0
        elif debt_to_equity < 0.7:
            return 0.8
        elif debt_to_equity < 1.0:
            return 0.6
        elif debt_to_equity < self.quality_thresholds['max_debt_to_equity']:
            return 0.4
        else:
            return 0.2  # High debt
    
    def _score_profit_margin(self, profit_margin: float) -> float:
        """Score profit margin."""
        if profit_margin >= 0.20:  # 20%+ margin
            return 1.0
        elif profit_margin >= 0.15:
            return 0.8
        elif profit_margin >= 0.10:
            return 0.6
        elif profit_margin >= self.quality_thresholds['min_profit_margin']:
            return 0.4
        else:
            return 0.2  # Low margin
    
    def _score_roe(self, roe: float) -> float:
        """Score return on equity."""
        if roe >= 0.25:  # 25%+ ROE
            return 1.0
        elif roe >= 0.20:
            return 0.8
        elif roe >= 0.15:
            return 0.6
        elif roe >= self.quality_thresholds['min_roe']:
            return 0.4
        else:
            return 0.2  # Low ROE
    
    def _score_pe_ratio(self, pe_ratio: float) -> float:
        """Score P/E ratio (moderate is better)."""
        if pe_ratio < 0:  # Negative earnings
            return 0.1
        elif 10 <= pe_ratio <= 20:
            return 1.0  # Ideal range
        elif 5 <= pe_ratio < 10 or 20 < pe_ratio <= 30:
            return 0.7
        elif pe_ratio < 5:
            return 0.5  # Too low might indicate problems
        elif pe_ratio <= self.quality_thresholds['max_pe_ratio']:
            return 0.4
        else:
            return 0.2  # Overvalued
    
    def get_financial_metrics(self, symbol: str) -> Dict[str, Any]:
        """
        Get financial metrics for the stock using yfinance.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing financial metrics
        """
        try:
            # Add .NS suffix for NSE stocks if not present
            if not symbol.endswith('.NS'):
                symbol_yf = f"{symbol}.NS"
            else:
                symbol_yf = symbol
            
            # Get stock info from yfinance
            stock = yf.Ticker(symbol_yf)
            info = stock.info
            
            # Extract key financial metrics
            metrics = {
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'pb_ratio': info.get('priceToBook'),
                'debt_to_equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else None,
                'eps_growth': self._calculate_eps_growth(stock),
                'revenue_growth': info.get('revenueGrowth'),
                'dividend_yield': info.get('dividendYield'),
                'profit_margin': info.get('profitMargins'),
                'roe': info.get('returnOnEquity'),
                'current_ratio': info.get('currentRatio'),
                'market_cap': info.get('marketCap'),
                'beta': info.get('beta')
            }
            
            # Log retrieved metrics
            non_null_metrics = {k: v for k, v in metrics.items() if v is not None}
            logger.debug(f"{symbol}: Retrieved {len(non_null_metrics)} fundamental metrics")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error retrieving financial metrics for {symbol}: {e}")
            return {}
    
    def _calculate_eps_growth(self, stock: yf.Ticker) -> Optional[float]:
        """
        Calculate EPS growth rate from quarterly earnings.
        
        Args:
            stock: yfinance Ticker object
            
        Returns:
            EPS growth rate or None if not available
        """
        try:
            # Get quarterly earnings
            earnings = stock.quarterly_earnings
            
            if earnings is None or earnings.empty or len(earnings) < 5:
                return None
            
            # Calculate YoY growth from most recent quarters
            current_eps = earnings.iloc[0]['Earnings']
            year_ago_eps = earnings.iloc[4]['Earnings'] if len(earnings) > 4 else earnings.iloc[-1]['Earnings']
            
            if year_ago_eps and year_ago_eps != 0:
                growth = (current_eps - year_ago_eps) / abs(year_ago_eps)
                return growth
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not calculate EPS growth: {e}")
            return None
    
    def is_quality_stock(self, symbol: str) -> bool:
        """
        Check if stock meets quality criteria for swing trading.
        Used as a tie-breaker between similar technical signals.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if stock meets quality criteria
        """
        try:
            metrics = self.get_financial_metrics(symbol)
            
            # Check each quality criterion
            quality_checks = []
            
            # EPS Growth check
            if metrics.get('eps_growth') is not None:
                passes_eps = metrics['eps_growth'] >= self.quality_thresholds['min_eps_growth']
                quality_checks.append(('eps_growth', passes_eps))
            
            # Debt/Equity check
            if metrics.get('debt_to_equity') is not None:
                passes_debt = metrics['debt_to_equity'] <= self.quality_thresholds['max_debt_to_equity']
                quality_checks.append(('debt_to_equity', passes_debt))
            
            # Profit Margin check
            if metrics.get('profit_margin') is not None:
                passes_margin = metrics['profit_margin'] >= self.quality_thresholds['min_profit_margin']
                quality_checks.append(('profit_margin', passes_margin))
            
            # ROE check
            if metrics.get('roe') is not None:
                passes_roe = metrics['roe'] >= self.quality_thresholds['min_roe']
                quality_checks.append(('roe', passes_roe))
            
            # Require at least 3 checks to pass, and majority must be positive
            if len(quality_checks) >= 3:
                passed = sum(1 for _, result in quality_checks if result)
                is_quality = passed >= len(quality_checks) * 0.6  # 60% must pass
                
                logger.info(f"{symbol}: Quality check {'PASSED' if is_quality else 'FAILED'} "
                           f"({passed}/{len(quality_checks)} criteria met)")
                return is_quality
            
            logger.warning(f"{symbol}: Insufficient data for quality check")
            return False
            
        except Exception as e:
            logger.error(f"Error in quality check for {symbol}: {e}")
            return False
