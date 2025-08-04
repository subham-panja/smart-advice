"""
Memory Optimization Utilities
File: utils/memory_utils.py

Utilities for optimizing memory usage, particularly for pandas DataFrames.
"""

import pandas as pd
from utils.logger import setup_logging

logger = setup_logging()

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    if df.empty:
        return df
    
    try:
        original_memory = df.memory_usage(deep=True).sum()
        
        # Downcast integer columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # Downcast float columns
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Log memory savings
        final_memory = df.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - final_memory) / original_memory * 100
        
        if memory_reduction > 5:  # Only log if significant reduction
            logger.debug(f"Memory optimization: {memory_reduction:.1f}% reduction "
                        f"({original_memory/1024:.1f}KB -> {final_memory/1024:.1f}KB)")
        
        return df
    except Exception as e:
        logger.debug(f"Failed to optimize DataFrame memory: {e}")
        return df
