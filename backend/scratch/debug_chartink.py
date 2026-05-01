import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChartinkDebug")


def test_chartink():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://chartink.com",
            "Referer": "https://chartink.com/screener",
        }
    )

    # 1. Get initial cookies and token
    logger.info("Fetching initial tokens...")
    session.get("https://chartink.com/screener")
    token = requests.utils.unquote(session.cookies.get("XSRF-TOKEN", ""))

    if not token:
        logger.error("Failed to get XSRF-TOKEN")
        return

    # 2. Test a very simple filter that SHOULD return many stocks
    # Basic Cash segment, Price > 20
    scan_clause = "( {cash} ( latest close > 20 ) )"

    logger.info(f"Testing scan: {scan_clause}")

    try:
        resp = session.post(
            "https://chartink.com/screener/process",
            data={"scan_clause": scan_clause},
            headers={"x-xsrf-token": token},
            timeout=30,
        )

        logger.info(f"Response Status: {resp.status_code}")

        if resp.status_code == 200:
            data = resp.json().get("data", [])
            logger.info(f"Success! Found {len(data)} stocks.")
            if data:
                logger.info(f"Sample stocks: {[s['nsecode'] for s in data[:5]]}")
        else:
            logger.error(f"Error Response: {resp.text}")

    except Exception as e:
        logger.error(f"Request failed: {e}")


if __name__ == "__main__":
    test_chartink()
