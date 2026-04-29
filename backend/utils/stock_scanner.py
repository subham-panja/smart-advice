import json
import logging
import os
import time
from typing import Any, Dict

import config
from scripts.data_fetcher import get_all_nse_symbols

logger = logging.getLogger(__name__)


class StockScanner:
    """Handles stock symbol discovery via Chartink or full NSE scan."""

    @staticmethod
    def get_symbols(max_stocks: int = None, use_all_symbols: bool = False) -> Dict[str, Any]:
        """Returns symbols dictionary from Chartink (if available) or full NSE."""
        if not use_all_symbols and getattr(config, "USE_CHARTINK", True):
            syms = StockScanner._load_cache("chartink")
            if syms:
                return syms

            try:
                from scripts.chartink_filter import ChartinkFilter

                cf = ChartinkFilter()
                syms = cf.get_filtered_symbols(max_stocks=max_stocks)
                if syms:
                    StockScanner._save_cache(syms, "chartink")
                    return syms
            except Exception as e:
                logger.error(f"Chartink failed: {e}")

        all_syms = get_all_nse_symbols()
        res = {s: {"name": s} for s in all_syms} if isinstance(all_syms, list) else all_syms
        return dict(list(res.items())[:max_stocks]) if max_stocks else res

    @staticmethod
    def _save_cache(syms: dict, src: str):
        path = os.path.join("cache", f"filtered_{src}.json")
        os.makedirs("cache", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"syms": syms, "time": time.time()}, f)

    @staticmethod
    def _load_cache(src: str) -> dict:
        path = os.path.join("cache", f"filtered_{src}.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            data = json.load(f)
        ttl = config.CHARTINK_CONFIG.get("cache_ttl_minutes", 30) * 60
        if time.time() - data["time"] > ttl:
            return {}
        return data["syms"]
