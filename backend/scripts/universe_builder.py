"""
Point-in-Time Universe Builder
================================

Eliminates survivorship bias in backtests by returning only stocks that were
listed and tradeable in the index at the given simulation start date.

Without this, a backtest run starting in 2017 would test on stocks that are
*currently* (2026) in the index — many of which became 10-100x winners and
were only added to the index AFTER their big run. This inflates historical
results dramatically.

Usage:
    from scripts.universe_builder import get_point_in_time_symbols

    symbols = get_point_in_time_symbols(sim_start_date, strategy_config, max_stocks=50)

Constituent Data:
    Stored in data/historical_constituents.json
    Approximate Nifty 500 constituent lists per year (2016–2024).
    Source: NSE India Index Archives (https://www.niftyindices.com/reports/historical-data)
    Accuracy: ~85-90%. Eliminates the worst survivorship bias without a paid data provider.
"""

import json
import logging
import os
import random
from functools import lru_cache
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "historical_constituents.json",
)


@lru_cache(maxsize=1)
def _load_constituents() -> Dict[int, Optional[List[str]]]:
    """Load and cache the historical constituent data from JSON."""
    try:
        with open(_DATA_FILE) as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except FileNotFoundError:
        logger.warning(
            f"historical_constituents.json not found at {_DATA_FILE} — "
            "will use live scanner for all backtests (survivorship bias not corrected)"
        )
        return {}
    except Exception as e:
        logger.warning(f"Failed to load historical_constituents.json: {e}")
        return {}


def get_point_in_time_symbols(
    sim_start: pd.Timestamp,
    strategy_config: dict,
    max_stocks: int = 50,
    local_data_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Return a point-in-time stock universe for the given simulation start date.

    For backtests starting before 2025, uses the historical constituent list
    to eliminate survivorship bias. For 2025+, falls back to the current live scanner.

    Args:
        sim_start: Simulation start date
        strategy_config: Strategy configuration dict (passed to StockScanner as fallback)
        max_stocks: Maximum number of stocks to return
        local_data_dir: Override path to the data/historical/ directory (for testing)

    Returns:
        Dict of {symbol: exchange} (same format as StockScanner.get_symbols())
    """
    year = sim_start.year
    constituents = _load_constituents()
    historical_list = constituents.get(year)

    # For future years or missing data, fall back to live scanner
    if not historical_list:
        logger.info(f"No point-in-time data for {year} — using current live scanner universe")
        return _get_live_universe(strategy_config, max_stocks)

    # Filter to symbols that actually have local historical data (parquet files)
    if local_data_dir is None:
        local_data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "historical",
        )

    available = _filter_to_available(historical_list, local_data_dir)

    if len(available) < 10:
        logger.warning(
            f"Point-in-time universe for {year} has only {len(available)} symbols "
            f"with local data — falling back to live scanner"
        )
        return _get_live_universe(strategy_config, max_stocks)

    # Shuffle deterministically by year so we get a representative mix across sectors
    shuffled = list(available)
    rng = random.Random(year)
    rng.shuffle(shuffled)
    sampled = shuffled[:max_stocks]

    logger.info(
        f"📅 Point-in-time universe ({year}): {len(sampled)} stocks "
        f"(from {len(available)} with local data, {len(historical_list)} in full list)"
    )

    return {sym: "NSE" for sym in sampled}


def _filter_to_available(symbols: List[str], data_dir: str) -> List[str]:
    """Keep only symbols that have a parquet file in the local data store."""
    return [sym for sym in symbols if os.path.exists(os.path.join(data_dir, f"{sym}.parquet"))]


def _get_live_universe(strategy_config: dict, max_stocks: int) -> Dict[str, str]:
    """Fall back to the current live StockScanner universe."""
    from utils.stock_scanner import StockScanner

    symbols = StockScanner().get_symbols(strategy_config=strategy_config)
    return dict(list(symbols.items())[:max_stocks])
