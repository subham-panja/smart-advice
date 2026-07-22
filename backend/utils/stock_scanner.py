import logging
from typing import Any, Dict

import config
from scripts.data_fetcher import get_all_nse_symbols
from utils.filter_translator import FilterTranslator

logger = logging.getLogger(__name__)


class StockScanner:
    """Handles stock symbol discovery via screener API or full NSE scan."""

    @staticmethod
    def get_symbols(
        strategy_config: Dict[str, Any], max_stocks: int = None, use_all_symbols: bool = False
    ) -> Dict[str, Any]:
        """Returns symbols dictionary from screener (if available) or full NSE."""
        strat_name = strategy_config["name"]

        if not use_all_symbols and config.USE_SCREENER:
            try:
                from scripts.screener_filter import ScreenerFilter

                filters = strategy_config["stock_filters"]
                scan_clause = FilterTranslator.translate_to_scan_clause(filters)
                logger.info(f"GENERATED SCREENER QUERY: {scan_clause}")

                cf = ScreenerFilter()
                syms = cf.get_filtered_symbols(scan_clause=scan_clause, max_stocks=max_stocks)

                if syms:
                    logger.info(f"Screener found {len(syms)} candidates for {strat_name}")
                    return syms
                else:
                    logger.warning(f"Screener returned zero results for {strat_name}. Skipping scan.")
                    return {}
            except Exception as e:
                logger.error(f"Screener failed for {strat_name}: {e}")
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
        """Screener first, then fill from NSE universe if needed.

        Args:
            strategy_config: Strategy config with stock_filters
            min_symbols: Minimum symbol count before expanding from NSE
            max_stocks: Maximum symbols to return
            source: 'screener' (screener only), 'nse_universe' (NSE only), 'both' (screener + NSE fill)

        Returns:
            Dict of symbols {symbol: name}
        """
        screener_syms = {}

        if source in ("screener", "both") and config.USE_SCREENER:
            screener_syms = StockScanner.get_symbols(
                strategy_config=strategy_config, max_stocks=max_stocks, use_all_symbols=False
            )

        if len(screener_syms) >= min_symbols or source == "screener":
            return dict(list(screener_syms.items())[:max_stocks])

        from scripts.data_fetcher import get_all_nse_symbols

        nse_syms = get_all_nse_symbols()

        if isinstance(nse_syms, list):
            nse_dict = {s: {"name": s} for s in nse_syms}
        else:
            nse_dict = nse_syms

        remaining = {k: v for k, v in nse_dict.items() if k not in screener_syms}

        combined = {**screener_syms, **remaining}
        return dict(list(combined.items())[:max_stocks])
