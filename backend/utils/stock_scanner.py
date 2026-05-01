import json
import logging
import os
import time
from typing import Any, Dict

import config
from scripts.data_fetcher import get_all_nse_symbols
from utils.filter_translator import FilterTranslator

logger = logging.getLogger(__name__)


class StockScanner:
    """Handles stock symbol discovery via Chartink or full NSE scan."""

    @staticmethod
    def get_symbols(
        strategy_config: Dict[str, Any], max_stocks: int = None, use_all_symbols: bool = False
    ) -> Dict[str, Any]:
        """Returns symbols dictionary from Chartink (if available) or full NSE."""
        strat_name = strategy_config["name"]

        if not use_all_symbols and config.USE_CHARTINK:
            syms = StockScanner._load_cache(f"chartink_{strat_name}", strategy_config)
            if syms:
                return syms
            try:
                from scripts.chartink_filter import ChartinkFilter

                # Generate dynamic scan clause from stock_filters
                filters = strategy_config["stock_filters"]
                scan_clause = FilterTranslator.translate_to_chartink(filters)

                cf = ChartinkFilter()
                # Pass the custom scan clause if generated
                syms = cf.get_filtered_symbols(scan_clause=scan_clause, max_stocks=max_stocks)

                if syms:
                    StockScanner._save_cache(syms, f"chartink_{strat_name}")
                    return syms
            except Exception as e:
                logger.error(f"Chartink failed for {strat_name}: {e}")
                raise e

        all_syms = get_all_nse_symbols()
        res = {s: {"name": s} for s in all_syms} if isinstance(all_syms, list) else all_syms

        # Safety: If we are falling back to full NSE, cap it to 100 unless max_stocks is specified
        if not use_all_symbols and not max_stocks:
            max_stocks = 100
            logger.warning(f"Safety: No Chartink result. Capping full scan to first {max_stocks} stocks.")

        return dict(list(res.items())[:max_stocks]) if max_stocks else res

    @staticmethod
    def _save_cache(syms: dict, src: str):
        path = os.path.join("cache", f"filtered_{src}.json")
        os.makedirs("cache", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"syms": syms, "time": time.time()}, f)

    @staticmethod
    def _load_cache(src: str, strategy_config: Dict[str, Any]) -> dict:
        path = os.path.join("cache", f"filtered_{src}.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)

        ttl_mins = strategy_config["chartink_config"]["cache_ttl_minutes"]
        ttl = ttl_mins * 60
        if time.time() - data["time"] > ttl:
            return {}
        return data["syms"]
