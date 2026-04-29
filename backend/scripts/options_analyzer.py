import logging
from typing import Any, Dict

import requests

from config import OPTIONS_OI_CONFIG

logger = logging.getLogger(__name__)


class OptionsAnalyzer:
    """Analyzes NSE Option Chain for institutional support and short squeeze signals."""

    def __init__(self):
        self.cfg = OPTIONS_OI_CONFIG
        self.headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/option-chain"}
        self.session = requests.Session()
        try:
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=5)
        except Exception:
            pass

    def get_oi_analysis(self, symbol: str) -> Dict[str, Any]:
        try:
            if not self.cfg.get("enabled", True):
                return {"passed": False}
            is_idx = symbol in ["NIFTY", "BANKNIFTY"]
            url = f"https://www.nseindia.com/api/option-chain-{'indices' if is_idx else 'equities'}?symbol={symbol}"
            res = self.session.get(url, headers=self.headers, timeout=10)
            if res.status_code != 200:
                return {"passed": False}

            data = res.json()
            if "filtered" not in data:
                return {"passed": False, "reason": "Not in F&O"}

            ce_oi = data["filtered"].get("CE", {}).get("totOI", 0)
            pe_oi = data["filtered"].get("PE", {}).get("totOI", 0)
            pcr = pe_oi / ce_oi if ce_oi > 0 else 0

            underlying = data.get("records", {}).get("underlyingValue", 0)
            f_data = data["filtered"].get("data", [])
            if not f_data or underlying == 0:
                return {"passed": True, "pcr": round(pcr, 2), "sentiment": "Neutral"}

            atm_idx = min(range(len(f_data)), key=lambda i: abs(f_data[i]["strikePrice"] - underlying))
            unwinding = any(
                f_data[i].get("CE", {}).get("changeinOpenInterest", 0) < 0
                for i in range(atm_idx, min(atm_idx + 3, len(f_data)))
            )

            bull_t = self.cfg.get("pcr_bullish_threshold", 1.2)
            return {
                "passed": True,
                "pcr": round(pcr, 2),
                "unwinding": unwinding,
                "sentiment": "Bullish" if pcr > bull_t or unwinding else "Neutral",
            }
        except Exception:
            return {"passed": False}


def analyze_oi(symbol: str) -> Dict[str, Any]:
    return OptionsAnalyzer().get_oi_analysis(symbol)
