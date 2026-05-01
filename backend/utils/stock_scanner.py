import logging
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

        # Always try Chartink first if enabled
        if not use_all_symbols and config.USE_CHARTINK:
            try:
                from scripts.chartink_filter import ChartinkFilter

                # Generate dynamic scan clause from stock_filters
                filters = strategy_config["stock_filters"]
                scan_clause = FilterTranslator.translate_to_chartink(filters)
                logger.info(f"🚀 GENERATED CHARTINK QUERY: {scan_clause}")
                print(f"🚀 GENERATED CHARTINK QUERY: {scan_clause}")

                cf = ChartinkFilter()
                # Pass the custom scan clause if generated
                syms = cf.get_filtered_symbols(scan_clause=scan_clause, max_stocks=max_stocks)

                if syms:
                    logger.info(f"Chartink found {len(syms)} candidates for {strat_name}")
                    return syms
                else:
                    logger.warning(f"Chartink returned zero results for {strat_name}. Skipping scan.")
                    return {}
            except Exception as e:
                logger.error(f"Chartink failed for {strat_name}: {e}")
                # We do NOT fall back to 100 stocks anymore.
                # If Chartink fails, we return empty to avoid downloading unnecessary data.
                return {}

        if use_all_symbols:
            all_syms = get_all_nse_symbols()
            res = {s: {"name": s} for s in all_syms} if isinstance(all_syms, list) else all_syms
            return dict(list(res.items())[:max_stocks]) if max_stocks else res

        return {}
