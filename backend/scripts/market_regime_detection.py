import logging
from typing import Any, Dict

from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)


class MarketRegimeDetection:
    """Detects market conditions (Bull/Bear) using SMA 200 on Nifty."""

    def get_simple_regime_check(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        try:
            from utils.trading_clock import is_replay, trading_now

            idx = strategy_config["index"]
            fetch_period = "10y" if is_replay() else "1y"
            df = get_historical_data(idx, period=fetch_period)
            if df.empty:
                raise ValueError(f"No historical data found for index: {idx}")

            if is_replay():
                sim_dt = trading_now().replace(tzinfo=None)
                sim_date = sim_dt.date() if hasattr(sim_dt, "date") else sim_dt
                df = df[df.index.date <= sim_date]
                if df.empty:
                    raise ValueError(f"No index data found for {idx} on/before simulated date {sim_date}")

            rule = strategy_config["bull_market_rule"]
            import re

            period_match = re.search(r"sma\((\d+)\)", rule)
            if not period_match:
                raise ValueError(f"Invalid bull_market_rule format: {rule}. Expected 'sma(N)'")

            period = int(period_match.group(1))

            effective_period = min(period, len(df))
            sma = df["Close"].rolling(effective_period, min_periods=min(30, effective_period)).mean().iloc[-1]
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
