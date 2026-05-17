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

    @staticmethod
    def get_symbols_with_fallback(
        strategy_config: Dict[str, Any],
        min_symbols: int = 50,
        max_stocks: int = 200,
        source: str = "both",
    ) -> Dict[str, Any]:
        """Chartink first, then fill from NSE universe if needed.

        Args:
            strategy_config: Strategy config with stock_filters
            min_symbols: Minimum symbol count before expanding from NSE
            max_stocks: Maximum symbols to return
            source: 'chartink' (Chartink only), 'nse_universe' (NSE only), 'both' (Chartink + NSE fill)

        Returns:
            Dict of symbols {symbol: name}
        """
        chartink_syms = {}

        # Try Chartink first
        if source in ("chartink", "both") and config.USE_CHARTINK:
            chartink_syms = StockScanner.get_symbols(
                strategy_config=strategy_config, max_stocks=max_stocks, use_all_symbols=False
            )

        # If we have enough from Chartink or Chartink-only mode, return
        if len(chartink_syms) >= min_symbols or source == "chartink":
            return dict(list(chartink_syms.items())[:max_stocks])

        # Expand from NSE universe
        from scripts.data_fetcher import get_all_nse_symbols

        nse_syms = get_all_nse_symbols()
        # nse_syms is a dict {symbol: symbol} or list

        if isinstance(nse_syms, list):
            nse_dict = {s: {"name": s} for s in nse_syms}
        else:
            nse_dict = nse_syms

        # Filter out already-found Chartink symbols, then take remaining
        remaining = {k: v for k, v in nse_dict.items() if k not in chartink_syms}

        # Combine: Chartink picks first, then fill from NSE
        combined = {**chartink_syms, **remaining}
        return dict(list(combined.items())[:max_stocks])
