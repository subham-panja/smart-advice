"""
Chartink Stock Screener Integration
=====================================

This module integrates with Chartink's screener API to filter NSE stocks
based on technical conditions. It replaces the slow per-stock yfinance
filtering with a single server-side scan that returns matching symbols
in seconds.

Flow:
  1. GET /screener page → collect XSRF-TOKEN and ci_session cookies
  2. POST scan_clause to /screener/process (form-encoded) → get matching stocks

IMPORTANT:
  - The scan_clause must be in Chartink's internal DSL format, NOT the
    human-readable query format shown in the screener UI.
  - Use lowercase keywords: latest, close, sma, volume, rsi, max, etc.
  - Wrap the entire clause in: ( {cash} ( ... ) )
"""

import requests
import logging
import time
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Chartink scan_clause
# This maps to the existing STOCK_FILTERING + SWING_TRADING_GATES
# conditions but is evaluated server-side by Chartink.
#
# NOTE: This is in Chartink's internal DSL, NOT the human-readable format.
# The human-readable version is:
#   Latest Close > Latest SMA(close, 50)
#   AND Latest Close > Latest SMA(close, 200)
#   AND Latest Volume > 2 * Latest SMA(volume, 20)
#   AND Latest Close > 1 day ago Maximum(20, High)
#   AND Latest RSI(14) > 60
#   AND Latest Close > Latest Open
#   AND Latest Close >= Latest High * 0.98
# ---------------------------------------------------------------------------
DEFAULT_SCAN_CLAUSE = (
    "( {cash} ( "
    "latest close > latest sma( close,50 ) and "
    "latest close > latest sma( close,200 ) and "
    "latest volume > latest sma( volume,20 ) * 2 and "
    "latest close > 1 day ago max( 20, high ) and "
    "latest rsi( 14 ) > 60 and "
    "latest close > latest open and "
    "latest close >= latest high * 0.98"
    " ) )"
)

CHARTINK_BASE_URL = "https://chartink.com"
CHARTINK_PROCESS_URL = f"{CHARTINK_BASE_URL}/screener/process"

# Common headers that Chartink expects
_BASE_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-GB,en;q=0.8",
    "origin": CHARTINK_BASE_URL,
    "referer": f"{CHARTINK_BASE_URL}/screener/short-term-breakouts",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36"
    ),
    "x-requested-with": "XMLHttpRequest",
}


class ChartinkFilter:
    """
    Connects to the Chartink screener to fetch pre-filtered stock lists.

    Usage:
        cf = ChartinkFilter()
        symbols = cf.get_filtered_symbols()                    # uses default scan
        symbols = cf.get_filtered_symbols(scan_clause=...)     # custom scan
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: Optional[requests.Session] = None

    # ------------------------------------------------------------------
    # Session management – we need cookies (XSRF-TOKEN, ci_session)
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        """
        Create a requests Session pre-loaded with cookies from Chartink.
        Chartink sets XSRF-TOKEN and ci_session via Set-Cookie on the
        initial page load; we need to forward them on subsequent API calls.
        """
        if self._session is not None:
            return self._session

        session = requests.Session()
        session.headers.update(_BASE_HEADERS)

        # Visit the screener page to collect cookies
        try:
            resp = session.get(
                f"{CHARTINK_BASE_URL}/screener",
                timeout=15,
            )
            resp.raise_for_status()
            logger.info(
                f"Chartink session initialized – cookies: "
                f"{list(session.cookies.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chartink session: {e}")
            raise

        self._session = session
        return session

    def _get_xsrf_token(self) -> str:
        """Extract the XSRF token value from the session cookies."""
        session = self._get_session()
        token = session.cookies.get("XSRF-TOKEN", "")
        return requests.utils.unquote(token)

    # ------------------------------------------------------------------
    # Core: POST scan_clause → get matching stocks
    # ------------------------------------------------------------------

    def _process_scan(self, scan_clause: str) -> Dict[str, Any]:
        """
        POST the scan_clause to /screener/process and return the JSON result.

        Chartink expects:
          - Content-Type: application/x-www-form-urlencoded
          - x-xsrf-token header
          - Form data: scan_clause=...
        """
        session = self._get_session()
        xsrf = self._get_xsrf_token()

        headers = {
            **_BASE_HEADERS,
            "x-xsrf-token": xsrf,
        }

        for attempt in range(self.max_retries):
            try:
                resp = session.post(
                    CHARTINK_PROCESS_URL,
                    data={"scan_clause": scan_clause},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                # Check for scan errors
                if data.get("scan_error"):
                    logger.error(f"Chartink scan error: {data['scan_error']}")
                    raise RuntimeError(f"Chartink scan error: {data['scan_error']}")

                logger.info(
                    f"Chartink scan processed successfully – "
                    f"{len(data.get('data', []))} results"
                )
                return data

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                if status == 419:
                    # CSRF token expired – reset session and retry
                    logger.warning("Chartink CSRF token expired, refreshing session")
                    self._session = None
                    session = self._get_session()
                    xsrf = self._get_xsrf_token()
                    headers["x-xsrf-token"] = xsrf
                else:
                    logger.error(
                        f"Chartink process HTTP error (attempt {attempt+1}): {e}"
                    )
            except RuntimeError:
                raise  # Don't retry scan errors (bad query)
            except Exception as e:
                logger.error(
                    f"Chartink process error (attempt {attempt+1}): {e}"
                )

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError("Failed to process Chartink scan after retries")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_filtered_symbols(
        self,
        scan_clause: str = None,
        query: str = None,
        max_stocks: int = None,
    ) -> Dict[str, str]:
        """
        Run a Chartink screener scan and return matching NSE symbols.

        Args:
            scan_clause: Chartink internal DSL scan clause.
                         Uses DEFAULT_SCAN_CLAUSE if None.
            query: Alias for scan_clause (backward compatibility).
            max_stocks: Maximum number of symbols to return.

        Returns:
            Dict mapping symbol → company name, e.g.
            {"RELIANCE": "Reliance Industries", ...}
        """
        clause = scan_clause or query or DEFAULT_SCAN_CLAUSE
        logger.info(
            f"Running Chartink screener filter "
            f"(max_stocks={max_stocks})"
        )

        try:
            # Execute the scan
            result = self._process_scan(clause)

            # Parse the response
            # Chartink returns: {"draw": N, "recordsTotal": N,
            #   "recordsFiltered": N, "data": [...], "link": "..."}
            # Each item: {"sr": 1, "nsecode": "OFSS", "name": "...",
            #   "bsecode": "532466", "close": 9360, "per_chg": 4.59,
            #   "volume": 843863}
            raw_stocks = result.get("data", [])
            logger.info(f"Chartink returned {len(raw_stocks)} matching stocks")

            symbols: Dict[str, str] = {}
            for stock in raw_stocks:
                symbol = stock.get("nsecode", "").strip().upper()
                name = stock.get("name", symbol).strip()
                if symbol:
                    symbols[symbol] = name

                if max_stocks and len(symbols) >= max_stocks:
                    break

            logger.info(
                f"Chartink filter returned {len(symbols)} symbols"
                + (f" (capped at {max_stocks})" if max_stocks else "")
            )
            return symbols

        except Exception as e:
            logger.error(f"Chartink filtering failed: {e}")
            raise

    def get_scan_details(
        self, scan_clause: str = None
    ) -> Dict[str, Any]:
        """
        Return the full scan result including per-stock metrics
        (close, percent change, volume).

        Useful for diagnostics and comparing with the legacy pipeline.
        """
        clause = scan_clause or DEFAULT_SCAN_CLAUSE
        return self._process_scan(clause)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def get_chartink_filtered_symbols(
    scan_clause: str = None,
    max_stocks: int = None,
) -> Dict[str, str]:
    """
    Convenience wrapper – instantiates ChartinkFilter and runs a scan.

    Args:
        scan_clause: Chartink scan clause string (optional).
        max_stocks: Maximum number of symbols to return.

    Returns:
        Dict mapping symbol → company name.
    """
    cf = ChartinkFilter()
    return cf.get_filtered_symbols(scan_clause=scan_clause, max_stocks=max_stocks)
