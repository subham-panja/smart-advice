import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import pandas as pd

from config import TIMEOUT_SECONDS
from scripts.alternative_data_fetcher import AlternativeDataFetcher
from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)


class EnhancedDataFetcher:
    """
    Enhanced data fetcher with multi-provider reconciliation and strict quality checks.
    """

    def __init__(self, app_config: Dict):
        """Initialize with mandatory configuration."""
        self.alt_fetcher = AlternativeDataFetcher()
        self.cache_dir = app_config["enhanced_cache_dir"]
        os.makedirs(self.cache_dir, exist_ok=True)

        # Provider weights from config
        fetch_cfg = app_config["data_fetch_config"]
        self.provider_weights = fetch_cfg["provider_weights"]
        self.quality_thresholds = fetch_cfg["quality_thresholds"]

    def fetch_from_multiple_providers(self, symbol: str, period: str) -> Dict[str, pd.DataFrame]:
        """Fetch data from multiple providers strictly."""
        provider_data = {}
        providers = {
            "nse_official": lambda: self._fetch_nse_data(symbol, period),
            "yahoo_finance": lambda: self._fetch_yahoo_data(symbol, period),
            "alternative": lambda: self._fetch_alternative_data(symbol, period),
        }

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_provider = {
                executor.submit(fetch_func): provider_name for provider_name, fetch_func in providers.items()
            }
            for future in as_completed(future_to_provider):
                p_name = future_to_provider[future]
                try:
                    data = future.result(timeout=TIMEOUT_SECONDS)
                    if not data.empty:
                        provider_data[p_name] = data
                except Exception as e:
                    logger.error(f"Critical fetch failure from {p_name} for {symbol}: {e}")
                    raise e

        if not provider_data:
            raise ValueError(f"All providers failed to return data for {symbol}")

        return provider_data

    def _fetch_nse_data(self, symbol: str, period: str) -> pd.DataFrame:
        period_days = self._period_to_days(period)
        return self.alt_fetcher.get_nse_stock_data(symbol, period_days)

    def _fetch_yahoo_data(self, symbol: str, period: str) -> pd.DataFrame:
        return get_historical_data(symbol, period=period)

    def _fetch_alternative_data(self, symbol: str, period: str) -> pd.DataFrame:
        return self.alt_fetcher.get_historical_data(symbol, period=period)

    def reconcile_data(self, provider_data: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """Reconcile data strictly across multiple providers."""
        if not provider_data:
            raise ValueError(f"No provider data to reconcile for {symbol}")

        all_dates = []
        for df in provider_data.values():
            all_dates.extend(df.index.tolist())
        common_dates = pd.DatetimeIndex(sorted(set(all_dates)))

        reconciled_data = []
        for date in common_dates:
            provider_values = {}
            for p_name, df in provider_data.items():
                if date in df.index:
                    provider_values[p_name] = df.loc[date].to_dict()

            if not provider_values:
                continue

            reconciled_row = {"Date": date}
            for field in ["Open", "High", "Low", "Close", "Volume"]:
                vals, weights = [], []
                for p_name, values in provider_values.items():
                    if field in values:
                        vals.append(values[field])
                        weights.append(self.provider_weights[p_name])

                if not vals:
                    raise ValueError(f"Missing field {field} for {symbol} on {date}")

                reconciled_row[field] = sum(v * w for v, w in zip(vals, weights)) / sum(weights)

            reconciled_data.append(reconciled_row)

        df = pd.DataFrame(reconciled_data)
        df.set_index("Date", inplace=True)
        return df.sort_index()

    def _period_to_days(self, period: str) -> int:
        period_map = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
        return period_map[period]
