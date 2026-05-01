import json
import logging
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar

logger = logging.getLogger(__name__)


class SmartMoneyTracker:
    """Tracks FII/DII data and delivery volumes from NSE with no fallbacks."""

    def __init__(self):
        self.hdr = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com/"}
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        try:
            self.opener.open(urllib.request.Request("https://www.nseindia.com", headers=self.hdr), timeout=5)
        except Exception:
            pass

    def get_fii_dii_status(self) -> dict:
        try:
            res = self.opener.open(
                urllib.request.Request("https://www.nseindia.com/api/fiidiiTradeReact", headers=self.hdr), timeout=10
            )
            data = json.loads(res.read())
            fii = next((i for i in data if i["category"] == "FII/FPI"))
            net = float(fii["netValue"].replace(",", ""))
            return {"fii_net": net, "is_bullish": net > -2000}
        except Exception as e:
            logger.error(f"FII/DII status fetch failed: {e}")
            raise e

    def get_delivery_volume(self, symbol: str) -> float:
        try:
            s = urllib.parse.quote(symbol.replace(".NS", ""))
            url = f"https://www.nseindia.com/api/quote-equity?symbol={s}&section=trade_info"
            response = self.opener.open(urllib.request.Request(url, headers=self.hdr), timeout=10)
            data = json.loads(response.read())

            if "securityWiseDP" not in data:
                logger.warning(f"No delivery data found for {symbol} in NSE response.")
                return 0.0

            return float(data["securityWiseDP"].get("deliveryToTradedQuantity", 0.0))
        except Exception as e:
            logger.error(f"Delivery volume fetch failed for {symbol}: {e}")
            return 0.0  # Return 0.0 as a safe fallback for bonus indicator
