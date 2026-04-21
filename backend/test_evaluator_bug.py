import sys
sys.path.append('.')
from scripts.data_fetcher import get_historical_data
from scripts.strategy_evaluator import StrategyEvaluator
import pandas as pd

def test():
    # Fetch real data for test
    df = get_historical_data("TCS", "1y", fresh=False)
    if df.empty:
        print("Empty DF")
        return
    
    # Serialize and deserialize like worker
    idx_str = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    data_dict = df.to_dict()
    
    recon_df = pd.DataFrame.from_dict(data_dict)
    recon_df.index = pd.to_datetime(idx_str)
    recon_df.index.name = 'Date'
    
    print("Datatypes after reconstruction:")
    print(recon_df.dtypes)
    
    # Run evaluator
    evaluator = StrategyEvaluator()
    res = evaluator.evaluate_strategies("TCS", recon_df)
    print("\nResult:")
    print("Positive:", res['positive_signals'])
    print("Total:", res['total_strategies'])
    print("Technical Score:", res['technical_score'])
    
    for k, v in res['strategy_results'].items():
        if v.get('signal') == -1 and v.get('signal_type') == 'ERROR':
            print(f"ERROR in {k}: {v.get('error')}")

if __name__ == "__main__":
    test()
