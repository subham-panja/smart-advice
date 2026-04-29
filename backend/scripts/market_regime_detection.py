import logging
from typing import Any, Dict

from config import MARKET_REGIME_CONFIG
from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)


class MarketRegimeDetection:
    """Detects market conditions (Bull/Bear) using SMA 200 on Nifty."""

    def get_simple_regime_check(self) -> Dict[str, Any]:
        try:
            idx = MARKET_REGIME_CONFIG.get("index", "^NSEI")
            df = get_historical_data(idx, period="1y")
            if df.empty:
                return {"passed": True, "reason": "No index data"}

            rule = MARKET_REGIME_CONFIG.get("bull_market_rule", "sma(200)")
            import re

            period_match = re.search(r"sma\((\d+)\)", rule)
            period = int(period_match.group(1)) if period_match else 200

            sma = df["Close"].rolling(period).mean().iloc[-1]
            cp = df["Close"].iloc[-1]
            bull = cp > sma
            return {
                "passed": bull,
                "status": "BULL" if bull else "BEAR",
                "reason": f"{idx} {'above' if bull else 'below'} {period} SMA",
            }
        except Exception as e:
            logger.error(f"Regime error: {e}")
            return {"passed": True}
