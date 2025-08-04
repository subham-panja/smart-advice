import pandas as pd
from typing import List
from models.stock import StockData

def scale_score(score, min_val, max_val, target_min=-1, target_max=1):
    """Scale a score from one range to another."""
    if max_val == min_val:
        return target_min if score <= min_val else target_max
    
    scaled_score = ((score - min_val) / (max_val - min_val)) * (target_max - target_min) + target_min
    return max(target_min, min(target_max, scaled_score))  # Clamp between target min/max

def convert_df_to_stockdata_list(df: pd.DataFrame) -> List[StockData]:
    """Convert pandas DataFrame to list of StockData objects."""
    stock_data_list = []
    for index, row in df.iterrows():
        stock_data = StockData(
            open=row['Open'],
            high=row['High'],
            low=row['Low'],
            close=row['Close'],
            volume=row['Volume'],
            date=index.date() if hasattr(index, 'date') else index
        )
        stock_data_list.append(stock_data)
    return stock_data_list

def ensure_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure OHLCV columns are numeric."""
    numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
