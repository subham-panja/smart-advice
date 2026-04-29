"""
NSE Options OI Analyzer
File: scripts/options_analyzer.py

Fetches and analyzes NSE Option Chain data to identify institutional support/resistance
and potential short squeeze opportunities (OI Unwinding).
"""

import requests
import pandas as pd
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class OptionsAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.nseindia.com/option-chain'
        }
        self.session = requests.Session()
        self._init_session()

    def _init_session(self):
        try:
            self.session.get("https://www.nseindia.com", headers=self.headers, timeout=10)
        except Exception as e:
            logger.error(f"Failed to initialize NSE session: {e}")

    def get_oi_analysis(self, symbol: str) -> Dict[str, Any]:
        try:
            # Handle Indices vs Equities
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY']:
                url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            else:
                url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"

            response = self.session.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200:
                return {'passed': False, 'reason': f"NSE API Error: {response.status_code}"}

            data = response.json()
            filtered = data.get('filtered', {}).get('data', [])
            if not filtered:
                return {'passed': False, 'reason': "No option data found"}

            # Current market price from option chain
            underlying_price = data['records']['underlyingValue']
            
            total_call_oi = data['filtered']['CE']['totOI']
            total_put_oi = data['filtered']['PE']['totOI']
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            # Find ATM Strike
            strike_diffs = [abs(x['strikePrice'] - underlying_price) for x in filtered]
            atm_index = strike_diffs.index(min(strike_diffs))
            atm_strike = filtered[atm_index]['strikePrice']

            # Check for Unwinding (Negative Change in OI) at nearest Call strikes
            # This indicates a potential short squeeze
            unwinding_detected = False
            nearby_calls = filtered[atm_index:atm_index+3] # ATM and 2 OTM calls
            for strike_data in nearby_calls:
                ce_data = strike_data.get('CE', {})
                if ce_data.get('changeinOpenInterest', 0) < 0:
                    unwinding_detected = True
                    break

            return {
                'passed': True,
                'pcr': round(pcr, 2),
                'underlying_price': underlying_price,
                'atm_strike': atm_strike,
                'unwinding_detected': unwinding_detected,
                'total_ce_oi': total_call_oi,
                'total_pe_oi': total_put_oi,
                'sentiment': 'Bullish' if pcr > 1.2 or unwinding_detected else 'Bearish' if pcr < 0.7 else 'Neutral'
            }

        except Exception as e:
            logger.error(f"Error analyzing OI for {symbol}: {e}")
            return {'passed': False, 'reason': str(e)}

def analyze_oi(symbol: str) -> Dict[str, Any]:
    return OptionsAnalyzer().get_oi_analysis(symbol)
