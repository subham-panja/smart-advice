"""
Screener.in Stock Filter Integration
======================================

Screener.in does NOT provide a public API. This module uses web scraping
to execute screener queries and extract matching stock symbols.

IMPORTANT:
  - Requires a valid Screener.in account for full access.
  - Scraping may break if Screener.in changes its HTML structure.
  - Respect Screener.in's Terms of Service and rate limits.

Flow:
  1. Obtain a session by visiting the login page and extracting CSRF token.
  2. (Optional) Authenticate with username/password for premium screens.
  3. POST the screen query to /screen/raw/ → parse HTML table for results.
"""

import os
import re
import requests
import logging
import time
from typing import Dict, List, Any, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Screener.in query
# Screener uses its own query language for fundamental + price data.
# This is a swing-trading oriented default that maps to our STOCK_FILTERING.
# ---------------------------------------------------------------------------
DEFAULT_SCREENER_QUERY = (
    "Current price > 20 AND "
    "Current price < 50000 AND "
    "Market Capitalization > 5000 AND "
    "Volume > 100000 AND "
    "Current price > DMA 50 AND "
    "Current price > DMA 200"
)

SCREENER_BASE_URL = "https://www.screener.in"
SCREENER_SCREEN_URL = f"{SCREENER_BASE_URL}/screen/raw/"
SCREENER_LOGIN_URL = f"{SCREENER_BASE_URL}/login/"

_BASE_HEADERS = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "accept-language": "en-GB,en;q=0.8",
    "user-agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36"
    ),
}


class ScreenerFilter:
    """
    Connects to Screener.in to fetch pre-filtered stock lists.

    Usage:
        sf = ScreenerFilter()
        symbols = sf.get_filtered_symbols()  # uses default query
        symbols = sf.get_filtered_symbols(query=...)  # custom query

    For premium screens, provide credentials:
        sf = ScreenerFilter(username="...", password="...")
    """

    def __init__(
        self,
        username: str = None,
        password: str = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.username = username or os.getenv("SCREENER_USERNAME", "")
        self.password = password or os.getenv("SCREENER_PASSWORD", "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._session: Optional[requests.Session] = None
        self._authenticated = False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        """Create a requests Session with cookies from Screener.in."""
        if self._session is not None:
            return self._session

        session = requests.Session()
        session.headers.update(_BASE_HEADERS)

        # Visit the login page to get the CSRF token
        try:
            resp = session.get(SCREENER_LOGIN_URL, timeout=15)
            resp.raise_for_status()
            logger.info(
                f"Screener.in session initialized – cookies: "
                f"{list(session.cookies.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Screener.in session: {e}")
            raise

        self._session = session

        # Authenticate if credentials are provided
        if self.username and self.password and not self._authenticated:
            self._authenticate()

        return session

    def _get_csrf_token(self) -> str:
        """Extract Django CSRF token from the session cookies."""
        session = self._get_session()
        return session.cookies.get("csrftoken", "")

    def _authenticate(self):
        """
        Log in to Screener.in using username and password.
        Required for accessing premium screens.
        """
        session = self._session
        if session is None:
            return

        csrf = self._get_csrf_token()
        login_payload = {
            "username": self.username,
            "password": self.password,
            "csrfmiddlewaretoken": csrf,
        }

        try:
            resp = session.post(
                SCREENER_LOGIN_URL,
                data=login_payload,
                headers={
                    **_BASE_HEADERS,
                    "referer": SCREENER_LOGIN_URL,
                    "content-type": "application/x-www-form-urlencoded",
                },
                timeout=15,
                allow_redirects=True,
            )
            resp.raise_for_status()

            # Check if login was successful (redirected to home page)
            if "/login/" not in resp.url:
                self._authenticated = True
                logger.info("Screener.in authentication successful")
            else:
                logger.warning(
                    "Screener.in authentication may have failed – "
                    "still on login page"
                )
        except Exception as e:
            logger.error(f"Screener.in authentication failed: {e}")

    # ------------------------------------------------------------------
    # Screen execution
    # ------------------------------------------------------------------

    def _run_screen(self, query: str, page: int = 1) -> Dict[str, Any]:
        """
        POST a screener query to /screen/raw/ and return parsed results.

        Screener.in returns an HTML page with a table of results.
        We parse the table to extract stock names and symbols.
        """
        session = self._get_session()
        csrf = self._get_csrf_token()

        payload = {
            "query": query,
            "page": page,
            "csrfmiddlewaretoken": csrf,
        }

        for attempt in range(self.max_retries):
            try:
                resp = session.post(
                    SCREENER_SCREEN_URL,
                    data=payload,
                    headers={
                        **_BASE_HEADERS,
                        "referer": f"{SCREENER_BASE_URL}/screen/raw/",
                        "content-type": "application/x-www-form-urlencoded",
                        "x-requested-with": "XMLHttpRequest",
                    },
                    timeout=30,
                )
                resp.raise_for_status()

                # The response could be HTML or JSON depending on endpoint
                content_type = resp.headers.get("content-type", "")

                if "application/json" in content_type:
                    data = resp.json()
                    return data
                else:
                    # Parse HTML response
                    return self._parse_html_results(resp.text)

            except requests.exceptions.HTTPError as e:
                status = getattr(e.response, "status_code", None)
                logger.error(
                    f"Screener.in screen HTTP error {status} "
                    f"(attempt {attempt+1}): {e}"
                )
            except Exception as e:
                logger.error(
                    f"Screener.in screen error (attempt {attempt+1}): {e}"
                )

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError("Failed to run Screener.in screen after retries")

    def _parse_html_results(self, html: str) -> Dict[str, Any]:
        """
        Parse the HTML table returned by Screener.in's screen endpoint.

        Returns:
            Dict with 'stocks' list and 'total_count'.
        """
        soup = BeautifulSoup(html, "html.parser")
        stocks = []

        # Find the results table
        table = soup.find("table", class_="data-table")
        if not table:
            # Try alternative selectors
            table = soup.find("table")

        if not table:
            logger.warning("No results table found in Screener.in response")
            # Check for error messages
            error_div = soup.find("div", class_="alert")
            if error_div:
                logger.error(f"Screener.in error: {error_div.get_text(strip=True)}")
            return {"stocks": [], "total_count": 0}

        # Parse table rows
        rows = table.find_all("tr")
        for row in rows[1:]:  # Skip header row
            cols = row.find_all("td")
            if len(cols) >= 2:
                # First column typically has the company link
                link = cols[0].find("a")
                if link:
                    href = link.get("href", "")
                    name = link.get_text(strip=True)

                    # Extract symbol from URL (e.g., /company/RELIANCE/consolidated/)
                    symbol_match = re.search(r"/company/([^/]+)/", href)
                    symbol = symbol_match.group(1) if symbol_match else name

                    stock_data = {
                        "symbol": symbol.upper(),
                        "name": name,
                    }

                    # Extract additional columns if available
                    for i, col in enumerate(cols[1:], 1):
                        stock_data[f"col_{i}"] = col.get_text(strip=True)

                    stocks.append(stock_data)

        # Check for pagination info
        total_count = len(stocks)
        count_text = soup.find("span", class_="count")
        if count_text:
            count_match = re.search(r"(\d+)", count_text.get_text())
            if count_match:
                total_count = int(count_match.group(1))

        return {"stocks": stocks, "total_count": total_count}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_filtered_symbols(
        self,
        query: str = None,
        max_stocks: int = None,
        fetch_all_pages: bool = False,
    ) -> Dict[str, str]:
        """
        Run a Screener.in query and return matching NSE symbols.

        Args:
            query: Screener.in query string. Uses DEFAULT if None.
            max_stocks: Maximum number of symbols to return.
            fetch_all_pages: If True, paginate through all results.

        Returns:
            Dict mapping symbol → company name.
        """
        query = query or DEFAULT_SCREENER_QUERY
        logger.info(
            f"Running Screener.in filter "
            f"(max_stocks={max_stocks})"
        )

        try:
            symbols: Dict[str, str] = {}
            page = 1

            while True:
                result = self._run_screen(query, page=page)
                stocks = result.get("stocks", [])

                if not stocks:
                    break

                for stock in stocks:
                    symbol = stock.get("symbol", "").strip().upper()
                    name = stock.get("name", symbol).strip()
                    if symbol:
                        symbols[symbol] = name

                    if max_stocks and len(symbols) >= max_stocks:
                        break

                if max_stocks and len(symbols) >= max_stocks:
                    break

                # Check if we should paginate
                total = result.get("total_count", 0)
                if not fetch_all_pages or len(symbols) >= total:
                    break

                page += 1
                time.sleep(1)  # Rate limit between pages

            logger.info(
                f"Screener.in filter returned {len(symbols)} symbols"
                + (f" (capped at {max_stocks})" if max_stocks else "")
            )
            return symbols

        except Exception as e:
            logger.error(f"Screener.in filtering failed: {e}")
            raise

    def get_company_details(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch detailed company page data from Screener.in.

        This can be used for fundamental analysis on specific stocks
        identified by the filter.
        """
        session = self._get_session()
        url = f"{SCREENER_BASE_URL}/company/{symbol}/consolidated/"

        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            details = {"symbol": symbol}

            # Extract key ratios
            ratios_section = soup.find("section", id="ratios")
            if ratios_section:
                ratio_items = ratios_section.find_all("li")
                for item in ratio_items:
                    name_el = item.find("span", class_="name")
                    value_el = item.find("span", class_="number")
                    if name_el and value_el:
                        key = name_el.get_text(strip=True)
                        val = value_el.get_text(strip=True)
                        details[key] = val

            return details

        except Exception as e:
            logger.error(f"Failed to fetch Screener.in details for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def get_screener_filtered_symbols(
    query: str = None,
    max_stocks: int = None,
) -> Dict[str, str]:
    """
    Convenience wrapper – instantiates ScreenerFilter and runs a screen.

    Args:
        query: Screener.in query string (optional).
        max_stocks: Maximum number of symbols to return.

    Returns:
        Dict mapping symbol → company name.
    """
    sf = ScreenerFilter()
    return sf.get_filtered_symbols(query=query, max_stocks=max_stocks)
