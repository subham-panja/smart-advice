import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChartinkDebug")


def test_specific_query():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://chartink.com",
            "Referer": "https://chartink.com/screener",
        }
    )

    logger.info("Fetching tokens...")
    session.get("https://chartink.com/screener")
    token = requests.utils.unquote(session.cookies.get("XSRF-TOKEN", ""))

    # THE EXACT QUERY THAT RETURNED ZERO
    scan_clause = "( {cash} ( latest close > 20.0 and latest close < 6000.0 and latest volume > 100000 and market cap > 5000.0 and latest close > latest sma( close,50 ) and latest close > latest sma( close,200 ) and latest rsi( 14 ) > 60.0 and ( [0] 1 day ago volume > [0] 1 day ago sma(volume,50) * 1.5 or [0] 2 day ago volume > [0] 2 day ago sma(volume,50) * 1.5 or [0] 3 day ago volume > [0] 3 day ago sma(volume,50) * 1.5 or [0] 4 day ago volume > [0] 4 day ago sma(volume,50) * 1.5 or [0] 5 day ago volume > [0] 5 day ago sma(volume,50) * 1.5 ) ) )"

    logger.info("Testing exact query...")

    try:
        resp = session.post(
            "https://chartink.com/screener/process",
            data={"scan_clause": scan_clause},
            headers={"x-xsrf-token": token},
            timeout=30,
        )

        if resp.status_code == 200:
            data = resp.json().get("data", [])
            logger.info(f"Result: Found {len(data)} stocks.")
        else:
            logger.error(f"Status {resp.status_code}: {resp.text}")

    except Exception as e:
        logger.error(f"Request failed: {e}")


if __name__ == "__main__":
    test_specific_query()
