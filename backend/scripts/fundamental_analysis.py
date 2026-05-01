import logging
from typing import Any, Dict

import yfinance as yf

logger = logging.getLogger(__name__)


class FundamentalAnalysis:
    """Performs financial metric evaluation using yfinance data strictly."""

    @staticmethod
    def get_data(symbol: str) -> Dict[str, Any]:
        """Fetches fundamental data or raises error."""
        ticker = yf.Ticker(f"{symbol}.NS" if ".NS" not in symbol else symbol)
        info = ticker.info
        if not info or "regularMarketPrice" not in info and "previousClose" not in info:
            raise ValueError(f"No fundamental info found for {symbol}")

        return {
            "pe": info.get("forwardPE") or info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "de": info.get("debtToEquity"),
            "eps_g": info.get("earningsGrowth"),
            "rev_g": info.get("revenueGrowth"),
            "roe": info.get("returnOnEquity"),
        }

    @staticmethod
    def calculate_score(data: Dict[str, Any], config: Dict[str, Any] = None) -> float:
        """Calculates score strictly. If all metrics are None, it raises ValueError.
        Uses strategy config thresholds when provided."""
        cfg = config or {}
        score, weight = 0.0, 0.0

        # P/E Ratio
        pe = data["pe"]
        if pe is not None:
            score += (1.0 if pe < 15 else (0.5 if pe < 25 else -0.5)) * 0.3
            weight += 0.3

        # Debt to Equity
        de = data["de"]
        max_de = cfg.get("max_debt_to_equity", 1.0)
        if de is not None:
            score += (1.0 if de < max_de * 0.5 else (0.0 if de < max_de else -1.0)) * 0.4
            weight += 0.4

        # EPS Growth
        g = data["eps_g"]
        min_eps_g = cfg.get("min_quarterly_eps_growth", 0.15)
        if g is not None:
            score += (1.0 if g > min_eps_g else (0.5 if g > 0 else -0.5)) * 0.3
            weight += 0.3

        # ROCE (if available)
        roce = data.get("roce")
        min_roce = cfg.get("min_roce")
        if roce is not None and min_roce is not None:
            score += (1.0 if roce > min_roce else -0.5) * 0.2
            weight += 0.2

        if weight == 0:
            raise ValueError("No valid fundamental metrics found to calculate score.")

        return score / weight

    @staticmethod
    def perform_fundamental_analysis(symbol: str, config: Dict[str, Any] = None) -> float:
        """Entry point for fundamental analysis. Fails gracefully."""
        try:
            data = FundamentalAnalysis.get_data(symbol)
            score = FundamentalAnalysis.calculate_score(data, config=config)
            logger.info(f"Fundamental {symbol}: {score:.2f}")
            return score
        except Exception as e:
            logger.warning(f"Fundamental analysis failed for {symbol}: {e}. Defaulting to 0.0")
            return 0.0
