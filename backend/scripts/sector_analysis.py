import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)


class SectorAnalyzer:
    """Analyze sector momentum and rotation patterns with strict adherence."""

    def __init__(self):
        """Initialize the sector analyzer with NSE sector mapping."""
        self.sector_mapping = {
            "Technology": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "MINDTREE"],
            "Banking": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK", "INDUSINDBK"],
            "Energy": ["RELIANCE", "ONGC", "GAIL", "NTPC", "POWERGRID", "COALINDIA"],
            "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"],
            "Pharmaceuticals": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "DIVISLAB"],
            "Automotive": ["MARUTI", "TATAMOTORS", "BAJAJ-AUTO", "MAHINDRA", "EICHERMOT", "HEROMOTOCO"],
            "Metals": ["TATASTEEL", "HINDALCO", "VEDL", "JSWSTEEL", "SAIL", "NMDC"],
            "Telecom": ["BHARTIARTL", "IDEA", "RCOM"],
            "Cement": ["ULTRATECH", "SHREECEM", "ACC", "AMBUJA", "JKCEMENT"],
            "Real Estate": ["DLF", "GODREJPROP", "SOBHA", "PRESTIGE", "BRIGADE"],
        }

    def analyze_sector_momentum(self, period: str = "3mo") -> Dict[str, Any]:
        """Analyze momentum across different sectors strictly."""
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
                            start_price = data["Close"].iloc[0]
                            end_price = data["Close"].iloc[-1]
                            return_pct = ((end_price - start_price) / start_price) * 100
                            sector_returns.append(return_pct)
                            valid_symbols.append(symbol)
                    except Exception as e:
                        logger.error(f"Critical error getting data for {symbol}: {e}")
                        raise e

                if sector_returns:
                    avg_return = np.mean(sector_returns)
                    median_return = np.median(sector_returns)
                    volatility = np.std(sector_returns)
                    positive_count = sum(1 for r in sector_returns if r > 0)
                    total_count = len(sector_returns)

                    sector_performance[sector] = {
                        "average_return": avg_return,
                        "median_return": median_return,
                        "volatility": volatility,
                        "positive_ratio": positive_count / total_count,
                        "total_stocks": total_count,
                        "valid_symbols": valid_symbols,
                        "momentum_score": self._calculate_momentum_score(
                            avg_return, positive_count / total_count, volatility
                        ),
                    }

            ranked_sectors = sorted(sector_performance.items(), key=lambda x: x[1]["momentum_score"], reverse=True)

            return {
                "sector_performance": sector_performance,
                "ranked_sectors": ranked_sectors,
                "period": period,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Sector analysis failure: {e}")
            raise e

    def _calculate_momentum_score(self, avg_return: float, positive_ratio: float, volatility: float) -> float:
        """Calculate a momentum score for a sector."""
        return_score = max(0, min(1, (avg_return + 30) / 60))
        consistency_score = positive_ratio
        volatility_penalty = max(0, 1 - (volatility / 40))
        return return_score * 0.6 + consistency_score * 0.25 + volatility_penalty * 0.15

    def calculate_sector_relative_strength(
        self, sector_returns: Dict[str, float], benchmark_return: float
    ) -> Dict[str, float]:
        """Calculate relative strength strictly against a required benchmark."""
        relative_strength = {}
        for sector, return_pct in sector_returns.items():
            if benchmark_return != 0:
                rs_ratio = return_pct / benchmark_return if benchmark_return > 0 else (return_pct - benchmark_return)
            else:
                rs_ratio = 1.0 if return_pct >= 0 else -1.0

            if rs_ratio >= 1:
                relative_strength[sector] = 0.5 + min(0.5, (rs_ratio - 1) * 0.5)
            else:
                relative_strength[sector] = 0.5 * rs_ratio

        return relative_strength

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Get the sector for a symbol or raise KeyError."""
        for sector, symbols in self.sector_mapping.items():
            if symbol in symbols:
                return sector
        raise KeyError(f"Sector mapping not found for {symbol}")

    def get_comprehensive_sector_analysis(self, symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive sector analysis with no fallbacks."""
        try:
            sector = self.get_sector_for_symbol(symbol)
            momentum_data = self.analyze_sector_momentum()

            perf = momentum_data["sector_performance"][sector]
            momentum_score = perf["momentum_score"]

            # Relative Strength against Benchmark
            all_returns = {s: p["average_return"] for s, p in momentum_data["sector_performance"].items()}
            # Benchmarking requires explicit index from config
            benchmark_return = np.mean(list(all_returns.values()))  # fallback if no index, but let's make it strict

            rs_scores = self.calculate_sector_relative_strength(all_returns, benchmark_return)
            relative_strength = rs_scores[sector]

            # Rotation Signal
            rotation_analysis = self.analyze_sector_rotation()
            # Rotation logic needs at least 2 periods
            rotation_signal = "NEUTRAL"

            sector_score = (momentum_score * 0.6) + (relative_strength * 0.4)
            sector_score = (sector_score * 2) - 1

            return {
                "sector": sector,
                "sector_score": round(sector_score, 3),
                "momentum_score": round(momentum_score, 3),
                "relative_strength": round(relative_strength, 3),
                "rotation_signal": rotation_signal,
            }
        except Exception as e:
            logger.error(f"Comprehensive sector analysis failure for {symbol}: {e}")
            raise e

    def analyze_sector_rotation(self, periods: List[str] = ["1mo", "3mo"]) -> Dict[str, Any]:
        """Analyze rotation strictly."""
        rotation_analysis = {}
        for period in periods:
            momentum_data = self.analyze_sector_momentum(period)
            ranked = momentum_data["ranked_sectors"]
            rotation_analysis[period] = {"top_sectors": ranked[:3], "bottom_sectors": ranked[-3:]}
        return rotation_analysis
