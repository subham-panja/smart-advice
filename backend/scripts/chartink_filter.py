import requests
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

DEFAULT_SCAN = (
    "( {cash} ( latest close > latest sma( close,50 ) and latest close > latest sma( close,200 ) and "
    "latest volume > latest sma( volume,20 ) * 2 and latest close > 1 day ago max( 20, high ) and "
    "latest rsi( 14 ) > 60 and latest close > latest open and latest close >= latest high * 0.98 ) )"
)

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
                data={"scan_clause": scan_clause or DEFAULT_SCAN},
                headers={"x-xsrf-token": token},
                timeout=30
            )
            data = resp.json().get("data", [])
            res = {s["nsecode"].strip().upper(): s["name"].strip() for s in data if s.get("nsecode")}
            return dict(list(res.items())[:max_stocks]) if max_stocks else res
        except Exception as e:
            logger.error(f"Chartink filter error: {e}")
            return {}
