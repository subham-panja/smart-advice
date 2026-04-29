import urllib.request
import urllib.parse
import json
import logging
from http.cookiejar import CookieJar

logger = logging.getLogger(__name__)

class SmartMoneyTracker:
    """Tracks FII/DII data and delivery volumes from NSE."""
    
    def __init__(self):
        self.hdr = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/'}
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
        try: self.opener.open(urllib.request.Request('https://www.nseindia.com', headers=self.hdr), timeout=5)
        except: pass

    def get_fii_dii_status(self) -> dict:
        try:
            res = self.opener.open(urllib.request.Request('https://www.nseindia.com/api/fiidiiTradeReact', headers=self.hdr), timeout=10)
            data = json.loads(res.read())
            fii = next((i for i in data if i['category'] == 'FII/FPI'), {})
            net = float(fii.get('netValue', '0').replace(',', ''))
            return {'fii_net': net, 'is_bullish': net > -2000}
        except Exception as e:
            logger.error(f"FII Error: {e}")
            return {'fii_net': 0, 'is_bullish': True}

    def get_delivery_volume(self, symbol: str) -> float:
        try:
            s = urllib.parse.quote(symbol.replace('.NS', ''))
            url = f'https://www.nseindia.com/api/quote-equity?symbol={s}&section=trade_info'
            data = json.loads(self.opener.open(urllib.request.Request(url, headers=self.hdr), timeout=10).read())
            return float(data.get('securityWiseDP', {}).get('deliveryToTradedQuantity', 0))
        except: return 0.0
