import os
import json
import logging
from typing import Dict, List, Any
from scripts.data_fetcher import get_all_nse_symbols, get_filtered_nse_symbols
import config

logger = logging.getLogger(__name__)

class StockScanner:
    """Handles fetching and filtering of stock symbols for analysis."""
    
    @staticmethod
    def get_symbols(max_stocks: int = None, use_all_symbols: bool = False, group_name: str = None) -> Dict[str, Any]:
        """Fetch stock symbols based on criteria."""
        
        # Branch 1: If a specific group is specified
        if group_name:
            logger.info(f"Fetching symbols for group: {group_name}...")
            groups_file = getattr(config, 'SYMBOL_GROUPS_FILE', None)
            if not groups_file or not os.path.exists(groups_file):
                logger.error(f"Symbol groups file not found at {groups_file}")
                return {}
            
            try:
                with open(groups_file, 'r') as f:
                    groups_data = json.load(f)
                
                group_symbols = groups_data.get(group_name)
                if not group_symbols:
                    logger.warning(f"Group '{group_name}' not found in {groups_file}")
                    return {}
                
                # Convert list to dict format and strip .NS suffix if present
                symbols = {}
                for s in group_symbols:
                    clean_symbol = s.replace('.NS', '').replace('.ns', '')
                    symbols[clean_symbol] = {'symbol': clean_symbol, 'company_name': clean_symbol}
                
                logger.info(f"Loaded {len(symbols)} symbols from group '{group_name}'")
                
            except Exception as e:
                logger.error(f"Error loading symbol group '{group_name}': {e}")
                return {}
                
        # Branch 2: Legacy behavior (all or filtered)
        elif use_all_symbols:
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
