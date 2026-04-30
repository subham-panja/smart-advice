import logging
from typing import Any, Dict

from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)


class MarketRegimeDetection:
    """Detects market conditions (Bull/Bear) using SMA 200 on Nifty."""

    def get_simple_regime_check(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            idx = strategy_config["index"]
            df = get_historical_data(idx, period="1y")
            if df.empty:
                raise ValueError(f"No historical data found for index: {idx}")

            rule = strategy_config["bull_market_rule"]
            import re

            period_match = re.search(r"sma\((\d+)\)", rule)
            if not period_match:
                raise ValueError(f"Invalid bull_market_rule format: {rule}. Expected 'sma(N)'")

            period = int(period_match.group(1))

            sma = df["Close"].rolling(period).mean().iloc[-1]
            cp = df["Close"].iloc[-1]
            bull = cp > sma
            return {
                "passed": bull,
                "status": "BULL" if bull else "BEAR",
                "reason": f"{idx} {'above' if bull else 'below'} {period} SMA",
            }
        except Exception as e:
            logger.error(f"Market regime detection failure: {e}")
            raise e
