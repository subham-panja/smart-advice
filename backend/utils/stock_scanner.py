import os
import json
import logging
import time
from typing import Dict, List, Any
from scripts.data_fetcher import get_all_nse_symbols, get_filtered_nse_symbols
import config

logger = logging.getLogger(__name__)


def _try_chartink(max_stocks: int = None) -> Dict[str, str]:
    """
    Attempt to filter stocks using Chartink screener.
    Returns empty dict on failure so caller can fallback.
    """
    try:
        from scripts.chartink_filter import ChartinkFilter

        chartink_cfg = getattr(config, 'CHARTINK_CONFIG', {})
        scan_clause = chartink_cfg.get('scan_clause', None)
        retries = chartink_cfg.get('max_retries', 3)
        delay = chartink_cfg.get('retry_delay', 2.0)

        cf = ChartinkFilter(max_retries=retries, retry_delay=delay)
        symbols = cf.get_filtered_symbols(scan_clause=scan_clause, max_stocks=max_stocks)

        if symbols:
            logger.info(f"Chartink filter returned {len(symbols)} symbols")

            # Cache the results if configured
            if chartink_cfg.get('cache_results', True):
                _cache_external_results(symbols, 'chartink', max_stocks)

            return symbols
        else:
            logger.warning("Chartink returned 0 symbols")
            return {}

    except ImportError:
        logger.error("chartink_filter module not available")
        return {}
    except Exception as e:
        logger.error(f"Chartink filtering failed: {e}")
        return {}


def _try_screener(max_stocks: int = None) -> Dict[str, str]:
    """
    Attempt to filter stocks using Screener.in.
    Returns empty dict on failure so caller can fallback.
    """
    try:
        from scripts.screener_filter import ScreenerFilter

        screener_cfg = getattr(config, 'SCREENER_CONFIG', {})
        query = screener_cfg.get('query', None)
        username = screener_cfg.get('username', '')
        password = screener_cfg.get('password', '')
        retries = screener_cfg.get('max_retries', 3)
        delay = screener_cfg.get('retry_delay', 2.0)
        fetch_all = screener_cfg.get('fetch_all_pages', False)

        sf = ScreenerFilter(
            username=username,
            password=password,
            max_retries=retries,
            retry_delay=delay,
        )
        symbols = sf.get_filtered_symbols(
            query=query,
            max_stocks=max_stocks,
            fetch_all_pages=fetch_all,
        )

        if symbols:
            logger.info(f"Screener.in filter returned {len(symbols)} symbols")
            _cache_external_results(symbols, 'screener', max_stocks)
            return symbols
        else:
            logger.warning("Screener.in returned 0 symbols")
            return {}

    except ImportError:
        logger.error("screener_filter module not available")
        return {}
    except Exception as e:
        logger.error(f"Screener.in filtering failed: {e}")
        return {}


def _cache_external_results(
    symbols: Dict[str, str], source: str, max_stocks: int = None
):
    """Cache results from external screeners for fast reload."""
    try:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(backend_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        tag = max_stocks or "all"
        cache_file = os.path.join(cache_dir, f"filtered_symbols_{source}_{tag}.json")

        with open(cache_file, 'w') as f:
            json.dump(
                {"symbols": symbols, "timestamp": time.time(), "source": source},
                f, indent=2,
            )
        logger.info(f"Cached {len(symbols)} {source} filtered symbols → {cache_file}")
    except Exception as e:
        logger.warning(f"Failed to cache {source} results: {e}")


def _load_cached_external(source: str, max_stocks: int = None) -> Dict[str, str]:
    """Load cached external screener results if still fresh."""
    try:
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(backend_dir, "cache")
        tag = max_stocks or "all"
        cache_file = os.path.join(cache_dir, f"filtered_symbols_{source}_{tag}.json")

        if not os.path.exists(cache_file):
            return {}

        # Check freshness
        if source == 'chartink':
            ttl = getattr(config, 'CHARTINK_CONFIG', {}).get('cache_ttl_minutes', 30) * 60
        else:
            ttl = 30 * 60  # 30 minutes default

        age = time.time() - os.path.getmtime(cache_file)
        if age > ttl:
            logger.info(f"Cached {source} results expired ({age/60:.0f} min old)")
            return {}

        with open(cache_file, 'r') as f:
            data = json.load(f)

        symbols = data.get("symbols", {})
        if symbols:
            logger.info(
                f"Loaded {len(symbols)} {source} filtered symbols from cache "
                f"({age/60:.0f} min old)"
            )
        return symbols

    except Exception as e:
        logger.warning(f"Failed to load cached {source} results: {e}")
        return {}


class StockScanner:
    """Handles fetching and filtering of stock symbols for analysis."""

    @staticmethod
    def get_symbols(max_stocks: int = None, use_all_symbols: bool = True, group_name: str = None) -> Dict[str, Any]:
        """Fetch stock symbols based on criteria."""

        # 1. Specific Group Scan (e.g., --group nifty50)
        if group_name:
            # ... (group logic remains the same)
            logger.info(f"Fetching symbols for group: {group_name}...")
            groups_file = getattr(config, 'SYMBOL_GROUPS_FILE', None)
            if not groups_file or not os.path.exists(groups_file):
                logger.error(f"Symbol groups file not found at {groups_file}")
                return {}
            try:
                with open(groups_file, 'r') as f:
                    groups_data = json.load(f)
                group_symbols = groups_data.get(group_name, [])
                symbols = {s.replace('.NS', '').replace('.ns', ''): {'symbol': s, 'company_name': s} for s in group_symbols}
                return symbols
            except Exception as e:
                logger.error(f"Error loading group {group_name}: {e}")
                return {}

        # 2. Main Logic: Try Chartink FIRST, then fallback to ALL NSE
        use_chartink = getattr(config, 'USE_CHARTINK', True)
        symbols = {}

        if use_chartink:
            logger.info("Chartink is enabled - checking for filtered symbols...")
            symbols = _load_cached_external('chartink', max_stocks)
            if not symbols:
                symbols = _try_chartink(max_stocks)
        
        # 3. If Chartink is OFF or returned nothing, do the FULL MARKET SCAN
        if not symbols:
            logger.info("No Chartink results - performing FULL NSE market scan...")
            all_symbols = get_all_nse_symbols()
            if isinstance(all_symbols, list):
                symbols = {s: {'company_name': s} for s in all_symbols}
            else:
                symbols = all_symbols

        # Apply limit if specified
        if max_stocks and len(symbols) > max_stocks:
            keys = list(symbols.keys())[:max_stocks]
            symbols = {k: symbols[k] for k in keys}

        return symbols
