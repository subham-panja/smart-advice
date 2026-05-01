import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()


def get_backtest_audit():
    host = os.getenv("MONGODB_HOST", "127.0.0.1")
    port = int(os.getenv("MONGODB_PORT", "27017"))
    db_name = os.getenv("MONGODB_DATABASE", "super_advice")

    client = MongoClient(host, port)
    db = client[db_name]

    symbols = ["MMTC", "SUZLON", "RENUKA"]

    print(f"{'Symbol':<12} | {'Trades':<8} | {'Win%':<8} | {'CAGR%':<8} | {'Profit Factor':<15}")
    print("-" * 65)

    for symbol in symbols:
        res = db.backtest_results.find_one({"symbol": symbol}, sort=[("_id", -1)])
        if res:
            m = res.get("combined_metrics", {})
            print(
                f"{symbol:<12} | {m.get('total_trades', 0):<8} | {m.get('win_rate', 0):<8.2f} | {m.get('avg_cagr', 0):<8.2f} | {m.get('profit_factor', 0):<15.2f}"
            )
        else:
            print(f"{symbol:<12} | {'NOT FOUND':<8}")


if __name__ == "__main__":
    get_backtest_audit()
