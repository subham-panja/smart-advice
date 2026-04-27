import urllib.request
import urllib.parse
import urllib.error
import json
import logging
from http.cookiejar import CookieJar

logger = logging.getLogger(__name__)

class SmartMoneyTracker:
    """
    Advanced tracking of Institutional footprints on the NSE:
    1. FII/DII Net Buying Data
    2. Stock-specific Delivery Volume %
    """
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/'
        }
        self.cookie_jar = CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookie_jar))
        self._init_session()

    def _init_session(self):
        """Ping the main page to get cookies."""
        try:
            req = urllib.request.Request('https://www.nseindia.com', headers=self.headers)
            self.opener.open(req, timeout=10)
        except Exception as e:
            logger.warning(f"Could not initialize NSE session: {e}")

    def get_fii_dii_status(self):
        """Checks if FIIs are net buyers or sellers today."""
        try:
            req = urllib.request.Request('https://www.nseindia.com/api/fiidiiTradeReact', headers=self.headers)
            response = self.opener.open(req, timeout=10).read()
            data = json.loads(response)
            
            fii_data = next((item for item in data if item['category'] == 'FII/FPI'), None)
            dii_data = next((item for item in data if item['category'] == 'DII'), None)
            
            fii_net = float(fii_data.get('netValue', '0').replace(',', '')) if fii_data else 0.0
            dii_net = float(dii_data.get('netValue', '0').replace(',', '')) if dii_data else 0.0
            
            # Determine if the institutional sentiment is extremely bearish (e.g. FII selling > 1500Cr)
            is_bullish = fii_net > -1500
            
            return {
                'fii_net': fii_net,
                'dii_net': dii_net,
                'is_bullish': is_bullish
            }
        except Exception as e:
            logger.error(f"FII/DII Fetch Error: {e}")
            return {'fii_net': 0, 'dii_net': 0, 'is_bullish': True} # Default to bullish on error to not block pipeline

    def get_delivery_volume(self, symbol):
        """Gets the delivery percentage for a specific stock."""
        try:
            # Clean symbol (e.g., RELIANCE.NS -> RELIANCE)
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            clean_symbol = urllib.parse.quote(clean_symbol)
            
            req = urllib.request.Request(f'https://www.nseindia.com/api/quote-equity?symbol={clean_symbol}&section=trade_info', headers=self.headers)
            response = self.opener.open(req, timeout=10).read()
            data = json.loads(response)
            
            delivery_pct = data.get('securityWiseDP', {}).get('deliveryToTradedQuantity', 0)
            return float(delivery_pct) if delivery_pct else 0.0
        except Exception as e:
            logger.warning(f"Delivery Volume Error for {symbol}: {e}")
            return 0.0
