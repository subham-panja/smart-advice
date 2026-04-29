import logging
from typing import Dict

import requests

from config import CHARTINK_CONFIG

logger = logging.getLogger(__name__)


class ChartinkFilter:
    """Connects to Chartink API for technical stock screening."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})

    def get_filtered_symbols(self, scan_clause: str = None, max_stocks: int = None) -> Dict[str, str]:
        try:
            self.session.get("https://chartink.com/screener", timeout=10)
            token = requests.utils.unquote(self.session.cookies.get("XSRF-TOKEN", ""))

            resp = self.session.post(
                "https://chartink.com/screener/process",
                data={"scan_clause": scan_clause or CHARTINK_CONFIG.get("scan_clause")},
                headers={"x-xsrf-token": token},
                timeout=30,
            )
            data = resp.json().get("data", [])
            res = {s["nsecode"].strip().upper(): s["name"].strip() for s in data if s.get("nsecode")}
            return dict(list(res.items())[:max_stocks]) if max_stocks else res
        except Exception as e:
            logger.error(f"Chartink filter error: {e}")
            return {}
