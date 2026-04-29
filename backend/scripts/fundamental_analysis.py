import yfinance as yf
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FundamentalAnalysis:
    """Performs financial metric evaluation using yfinance data."""
    
    @staticmethod
    def get_data(symbol: str) -> Optional[Dict[str, Any]]:
        try:
            ticker = yf.Ticker(f"{symbol}.NS" if ".NS" not in symbol else symbol)
            info = ticker.info
            if not info or 'regularMarketPrice' not in info: return None
            return {
                'pe': info.get('forwardPE') or info.get('trailingPE'),
                'pb': info.get('priceToBook'),
                'de': info.get('debtToEquity'),
                'eps_g': info.get('earningsGrowth'),
                'rev_g': info.get('revenueGrowth'),
                'roe': info.get('returnOnEquity')
            }
        except Exception as e:
            logger.error(f"Fundamental fetch error {symbol}: {e}")
            return None

    @staticmethod
    def calculate_score(data: Dict[str, Any]) -> float:
        score, weight = 0.0, 0.0
        
        # P/E Ratio (Lower is better)
        if data.get('pe'):
            pe = data['pe']
            score += (1.0 if pe < 15 else (0.5 if pe < 25 else -0.5)) * 0.3
            weight += 0.3
            
        # Debt to Equity (Lower is better)
        if data.get('de') is not None:
            de = data['de']
            score += (1.0 if de < 0.5 else (0.0 if de < 1.0 else -1.0)) * 0.4
            weight += 0.4
            
        # EPS Growth (Higher is better)
        if data.get('eps_g') is not None:
            g = data['eps_g']
            score += (1.0 if g > 0.15 else (0.5 if g > 0 else -0.5)) * 0.3
            weight += 0.3
            
        return score / weight if weight > 0 else 0.1

    @staticmethod
    def perform_fundamental_analysis(symbol: str) -> float:
        data = FundamentalAnalysis.get_data(symbol)
        if not data: return 0.0
        score = FundamentalAnalysis.calculate_score(data)
        logger.info(f"Fundamental {symbol}: {score:.2f}")
        return score
