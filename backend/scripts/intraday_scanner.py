"""
Intraday Confluence Scanner
===========================
Scans liquid NSE equities for ORB, VWAP, and CPR confluence setups.
"""

import logging
from typing import List, Optional

import pandas as pd

from scripts.intraday_engine import load_intraday_config
from scripts.intraday_indicators import evaluate_intraday_confluence

logger = logging.getLogger(__name__)


class IntradayScanner:
    """Scans liquid stocks for active intraday trading opportunities."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_intraday_config()
        self.time_cfg = self.config.get("timeframe_config", {})

    def scan_universe(
        self,
        symbols: Optional[List[str]] = None,
        symbols_5m_data: Optional[dict] = None,
        current_time_str: str = "10:15",
    ) -> List[dict]:
        """
        Scans provided symbols (or defaults) against intraday confluence rules.
        """
        if not symbols_5m_data:
            return []

        candidates = []
        for symbol, data in symbols_5m_data.items():
            df_5m = data.get("df_5m", pd.DataFrame())
            prev_day = data.get("prev_day", {})

            if df_5m.empty or not prev_day:
                continue

            result = evaluate_intraday_confluence(
                symbol=symbol,
                df_5m=df_5m,
                prev_day_ohlc=prev_day,
                current_time_str=current_time_str,
                config=self.config,
            )

            # Assign confluence score
            gates = result.get("gates", {})
            passed_count = sum(1 for v in gates.values() if v)
            total_gates = max(len(gates), 1)
            score = round((passed_count / total_gates), 2)

            result["confluence_score"] = score
            candidates.append(result)

        # Sort candidates: BUY signals first, then by confluence score and volume ratio
        candidates.sort(
            key=lambda x: (
                1 if x.get("signal") == "BUY" else 0,
                x.get("confluence_score", 0),
                x.get("volume_ratio", 0),
            ),
            reverse=True,
        )

        return candidates
