import requests
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OptionsAnalyzer:
    """Analyzes NSE Option Chain for institutional support and short squeeze signals."""
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.nseindia.com/option-chain'}
        self.session = requests.Session()
        try: self.session.get("https://www.nseindia.com", headers=self.headers, timeout=5)
        except: pass

    def get_oi_analysis(self, symbol: str) -> Dict[str, Any]:
        try:
            is_idx = symbol in ['NIFTY', 'BANKNIFTY']
            url = f"https://www.nseindia.com/api/option-chain-{'indices' if is_idx else 'equities'}?symbol={symbol}"
            res = self.session.get(url, headers=self.headers, timeout=10)
            if res.status_code != 200: return {'passed': False}

            data = res.json()
            ce_oi, pe_oi = data['filtered']['CE']['totOI'], data['filtered']['PE']['totOI']
            pcr = pe_oi / ce_oi if ce_oi > 0 else 0
            
            # Check for CE unwinding near ATM
            underlying = data['records']['underlyingValue']
            filtered = data['filtered']['data']
            atm_idx = min(range(len(filtered)), key=lambda i: abs(filtered[i]['strikePrice'] - underlying))
            unwinding = any(filtered[i].get('CE', {}).get('changeinOpenInterest', 0) < 0 for i in range(atm_idx, min(atm_idx+3, len(filtered))))

            return {
                'passed': True, 'pcr': round(pcr, 2), 'unwinding': unwinding,
                'sentiment': 'Bullish' if pcr > 1.2 or unwinding else 'Neutral'
            }
        except Exception as e:
            logger.error(f"OI Error {symbol}: {e}")
            return {'passed': False}

def analyze_oi(symbol: str) -> Dict[str, Any]: return OptionsAnalyzer().get_oi_analysis(symbol)
