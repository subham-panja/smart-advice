import logging
from typing import Dict, List, Any
from scripts.data_fetcher import get_all_nse_symbols, get_filtered_nse_symbols

logger = logging.getLogger(__name__)

class StockScanner:
    """Handles fetching and filtering of stock symbols for analysis."""
    
    @staticmethod
    def get_symbols(max_stocks: int = None, use_all_symbols: bool = False) -> Dict[str, Any]:
        """Fetch stock symbols based on criteria."""
        if use_all_symbols:
            logger.info(f"Fetching all NSE symbols (max={max_stocks})...")
            all_symbols = get_all_nse_symbols()
            if not all_symbols:
                return {}
            
            # Convert list to dict format if needed
            if isinstance(all_symbols, list):
                symbols = {s: {'company_name': s} for s in all_symbols}
            else:
                symbols = all_symbols
        else:
            logger.info(f"Fetching filtered NSE symbols (max={max_stocks})...")
            symbols = get_filtered_nse_symbols(max_stocks)
            
        # Apply limit if specified
        if max_stocks and len(symbols) > max_stocks:
            keys = list(symbols.keys())[:max_stocks]
            symbols = {k: symbols[k] for k in keys}
            
        return symbols
