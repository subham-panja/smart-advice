import config
from scripts.analyzer import StockAnalyzer
from scripts.data_fetcher import get_historical_data

symbols = [
    "KSHITIJPOL",
    "SILVERTUC",
    "WELENT",
    "STLTECH",
    "POCL",
    "SKYGOLD",
    "INDIAGLYCO",
    "CENTURYPLY",
    "CHENNPETRO",
    "RRKABEL",
    "COCHINSHIP",
    "BSE",
    "CDSL",
    "MCX",
    "ZOMATO",
]

print(f"{'Symbol':<15} | {'Recommended':<12} | {'Reason':<15} | {'Trend':<6} | {'Vol':<6} | {'Vlty':<6}")
print("-" * 80)

for s in symbols:
    try:
        hist = get_historical_data(s, "1y")
        res = StockAnalyzer().analyze_stock_with_data(s, "", hist, config.ANALYSIS_CONFIG)
        gates = res.get("swing_analysis", {}).get("gates", {})
        print(
            f"{s:<15} | {str(res.get('is_recommended')):<12} | {res.get('reason', ''):<15} | {str(gates.get('trend')):<6} | {str(gates.get('volume')):<6} | {str(gates.get('volatality', gates.get('volatility'))):<6}"
        )
    except Exception as e:
        print(f"{s:<15} | ERROR: {e}")
