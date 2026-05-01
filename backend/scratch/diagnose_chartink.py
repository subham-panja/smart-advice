import logging

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChartinkFix")


def diagnose():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://chartink.com",
            "Referer": "https://chartink.com/screener",
        }
    )
    session.get("https://chartink.com/screener")
    token = requests.utils.unquote(session.cookies.get("XSRF-TOKEN", ""))

    queries = {
        "Base (Price/Volume/RSI)": "( {cash} ( latest close > 20 and latest volume > 100000 and latest rsi(14) > 55 ) )",
        "With Market Cap": "( {cash} ( latest close > 20 and market cap > 5000 ) )",
        "With SMAs": "( {cash} ( latest close > latest sma(close,50) and latest close > latest sma(close,200) ) )",
        "Full Practical": "( {cash} ( latest close > 20.0 and latest close < 6000.0 and latest volume > 100000 and market cap > 5000.0 and latest close > latest sma( close,50 ) and latest close > latest sma( close,200 ) and latest rsi( 14 ) > 55.0 ) )",
    }

    for name, q in queries.items():
        try:
            resp = session.post(
                "https://chartink.com/screener/process", data={"scan_clause": q}, headers={"x-xsrf-token": token}
            )
            data = resp.json().get("data", [])
            logger.info(f"{name}: Found {len(data)} stocks.")
        except Exception as e:
            logger.error(f"{name} Error: {e}")


if __name__ == "__main__":
    diagnose()
