import sys
sys.path.append('.')
from scripts.data_fetcher import get_historical_data
from scripts.strategy_evaluator import StrategyEvaluator
from scripts.strategies.ma_crossover_50_200 import MA_Crossover_50_200
from scripts.strategies.rsi_overbought_oversold import RSI_Overbought_Oversold
import pandas as pd

def test():
    # Fetch real data for test
    df = get_historical_data("TCS", "1y", fresh=False)
    
    # Let's test standard execution
    print("Normal df tail:")
    print(df.tail(2))
    
    ma = MA_Crossover_50_200()
    rsi = RSI_Overbought_Oversold()
    
    print("\n--- NORMAL DF ---")
    print("MA:", ma._execute_strategy_logic(df))
    print("RSI:", rsi._execute_strategy_logic(df))
    
    print("\n--- RECONSTRUCTED DF ---")
    idx_str = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
    data_dict = df.to_dict()
    recon_df = pd.DataFrame.from_dict(data_dict)
    recon_df.index = pd.to_datetime(idx_str)
    recon_df.index.name = 'Date'
    
    print("MA:", ma._execute_strategy_logic(recon_df))
    print("RSI:", rsi._execute_strategy_logic(recon_df))

if __name__ == "__main__":
    test()
