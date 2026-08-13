import logging
from typing import Dict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class ScreenerFilter:
    """Connects to Chartink API for technical stock screening."""

    def __init__(self):
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=20)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update({"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"})

    def get_filtered_symbols(self, scan_clause: str = None, max_stocks: int = None) -> Dict[str, str]:
        if not scan_clause:
            logger.error("No scan_clause provided to ScreenerFilter.")
            return {}

        try:
            self.session.get("https://chartink.com/screener", timeout=10)
            token = requests.utils.unquote(self.session.cookies.get("XSRF-TOKEN", ""))

            resp = self.session.post(
                "https://chartink.com/screener/process",
                data={"scan_clause": scan_clause},
                headers={"x-xsrf-token": token},
                timeout=30,
            )
            data = resp.json().get("data", [])
            res = {s["nsecode"].strip().upper(): s["name"].strip() for s in data if s.get("nsecode")}
            return dict(list(res.items())[:max_stocks]) if max_stocks else res
        except Exception as e:
            logger.error(f"Screener filter error: {e}")
            return {}
