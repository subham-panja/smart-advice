import logging
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Analyzes NSE Option Chain with no fallbacks."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/option-chain"}
        self.session = requests.Session()
        # Initialize session cookies
        self.session.get("https://www.nseindia.com", headers=self.headers, timeout=10)

    def get_oi_analysis(self, symbol: str) -> Dict[str, Any]:
        """Strictly analyzes OI data."""
        if not self.cfg["enabled"]:
            return {"passed": False}

        is_idx = symbol in ["NIFTY", "BANKNIFTY"]
        url = f"https://www.nseindia.com/api/option-chain-{'indices' if is_idx else 'equities'}?symbol={symbol}"
        res = self.session.get(url, headers=self.headers, timeout=10)

        if res.status_code != 200:
            raise RuntimeError(f"NSE Option Chain API failed for {symbol} (HTTP {res.status_code})")

        data = res.json()
        if "filtered" not in data:
            return {"passed": False, "reason": "Not in F&O"}

        ce_oi = data["filtered"]["CE"]["totOI"]
        pe_oi = data["filtered"]["PE"]["totOI"]
        if ce_oi == 0:
            raise ValueError(f"Zero CE OI for {symbol}; PCR undefined.")

        pcr = pe_oi / ce_oi

        underlying = data["records"]["underlyingValue"]
        f_data = data["filtered"]["data"]
        if not f_data or underlying == 0:
            raise ValueError(f"Incomplete option chain records for {symbol}")

        atm_idx = min(range(len(f_data)), key=lambda i: abs(f_data[i]["strikePrice"] - underlying))
        unwinding = any(
            f_data[i]["CE"]["changeinOpenInterest"] < 0 for i in range(atm_idx, min(atm_idx + 3, len(f_data)))
        )

        bull_t = self.cfg["pcr_bullish_threshold"]
        return {
            "passed": True,
            "pcr": round(pcr, 2),
            "unwinding": unwinding,
            "sentiment": "Bullish" if pcr < bull_t or unwinding else "Neutral",
        }


def analyze_oi(symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
    return OptionsAnalyzer(config=config).get_oi_analysis(symbol)
