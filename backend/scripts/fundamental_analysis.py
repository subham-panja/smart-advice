import json
import logging
import os
from typing import Any, Dict

import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import config
from utils.trading_clock import is_replay

logger = logging.getLogger(__name__)

FUNDAMENTAL_CACHE_FILE = os.path.join(config.BACKEND_DIR, "data", "fundamental_cache.json")
_fundamental_cache = {}
_cache_loaded = False


def _load_fundamental_cache():
    global _fundamental_cache, _cache_loaded
    if not _cache_loaded:
        if os.path.exists(FUNDAMENTAL_CACHE_FILE):
            try:
                with open(FUNDAMENTAL_CACHE_FILE, "r") as f:
                    _fundamental_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load fundamental cache: {e}")
                _fundamental_cache = {}
        _cache_loaded = True


def _save_fundamental_cache():
    try:
        os.makedirs(os.path.dirname(FUNDAMENTAL_CACHE_FILE), exist_ok=True)
        with open(FUNDAMENTAL_CACHE_FILE, "w") as f:
            json.dump(_fundamental_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save fundamental cache: {e}")


_session = None


def _get_fundamental_session():
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=50)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


class FundamentalAnalysis:
    """Performs financial metric evaluation using cached data strictly."""

    @staticmethod
    def get_data(symbol: str) -> Dict[str, Any]:
        """Fetches fundamental data or uses persistent disk cache."""
        _load_fundamental_cache()

        if symbol in _fundamental_cache:
            cached = _fundamental_cache[symbol]
            if cached:
                return cached
            elif is_replay():
                return {
                    "pe": 20.0,
                    "pb": 2.5,
                    "de": 0.5,
                    "eps_g": 0.15,
                    "rev_g": 0.15,
                    "roe": 0.15,
                }

        # In replay mode, do not make un-cached live HTTP calls to Yahoo Finance
        if is_replay():
            fallback = {
                "pe": 20.0,
                "pb": 2.5,
                "de": 0.5,
                "eps_g": 0.15,
                "rev_g": 0.15,
                "roe": 0.15,
            }
            _fundamental_cache[symbol] = fallback
            return fallback

        session = _get_fundamental_session()
        ticker_symbol = f"{symbol}.NS" if ".NS" not in symbol else symbol
        try:
            ticker = yf.Ticker(ticker_symbol, session=session)
            info = ticker.info
        except Exception:
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
            except Exception:
                info = None

        if not info or ("regularMarketPrice" not in info and "previousClose" not in info):
            _fundamental_cache[symbol] = {}
            _save_fundamental_cache()
            raise ValueError(f"No fundamental info found for {symbol}")

        de_raw = info.get("debtToEquity")
        if de_raw is not None and de_raw > 5.0:
            de_raw = de_raw / 100.0

        res = {
            "pe": info.get("forwardPE") or info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "de": de_raw,
            "eps_g": info.get("earningsGrowth"),
            "rev_g": info.get("revenueGrowth"),
            "roe": info.get("returnOnEquity"),
        }
        _fundamental_cache[symbol] = res
        _save_fundamental_cache()
        return res

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

        # Revenue Growth (3-year and 5-year profit growth thresholds)
        rev_g = data.get("rev_g")
        min_profit_growth_3y = cfg.get("min_profit_growth_3y", None)
        min_profit_growth_5y = cfg.get("min_profit_growth_5y", None)

        if rev_g is not None:
            # Score based on 3-year growth threshold if configured
            if min_profit_growth_3y is not None:
                growth_3y_ok = rev_g > (min_profit_growth_3y / 100.0)  # Convert from percentage
                score += (0.5 if growth_3y_ok else 0.0) * 0.2
                weight += 0.2

            # Score based on 5-year growth threshold if configured
            if min_profit_growth_5y is not None:
                growth_5y_ok = rev_g > (min_profit_growth_5y / 100.0)  # Convert from percentage
                score += (0.5 if growth_5y_ok else 0.0) * 0.15
                weight += 0.15

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
