import logging
from typing import Any, Dict

import pandas as pd
import talib as ta

from scripts.analyzer import StockAnalyzer
from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)


class ConfluenceEngine:
    """
    Multi-Timeframe Confluence Engine with no fallbacks and strict adherence.
    """

    def __init__(self, app_config: Dict):
        """Initialize with mandatory timeframes and rules."""
        conf_cfg = app_config["confluence_config"]
        self.timeframes = conf_cfg["timeframes"]
        self.confluence_rules = conf_cfg["rules"]
        self.analyzer = StockAnalyzer(app_config)

    def analyze_multi_timeframe(self, symbol: str, period: str) -> Dict[str, Any]:
        """Analyze a stock across multiple timeframes strictly."""
        logger.info(f"Starting strict confluence analysis for {symbol}")
        try:
            results = {
                "symbol": symbol,
                "timeframe_analysis": {},
                "confluence_signals": {},
            }

            for timeframe in self.timeframes:
                data = get_historical_data(symbol, period, timeframe)
                if data.empty:
                    raise ValueError(f"No data for {symbol} on {timeframe}")

                results["timeframe_analysis"][timeframe] = self._analyze_single_timeframe(data, timeframe)

            confluence_signals = self._generate_confluence_signals(results["timeframe_analysis"])
            results["confluence_signals"] = confluence_signals

            final_rec = self._determine_final_recommendation(confluence_signals)
            results.update(final_rec)

            return results
        except Exception as e:
            logger.error(f"Confluence analysis failure for {symbol}: {e}")
            raise e

    def _analyze_single_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Analyze a single timeframe strictly."""
        close = data["Close"].values
        high = data["High"].values
        low = data["Low"].values
        volume = data["Volume"].values

        ma_50 = ta.SMA(close, timeperiod=50)
        ma_200 = ta.SMA(close, timeperiod=200)
        rsi = ta.RSI(close, timeperiod=14)
        macd_line, macd_signal, _ = ta.MACD(close)
        volume_ma = ta.SMA(volume.astype(float), timeperiod=20)

        signals = {}
        # Strict mapping - no .get()
        signals["trend_bullish"] = ma_50[-1] > ma_200[-1]
        signals["RSI_value"] = rsi[-1]
        signals["MACD_bullish"] = macd_line[-1] > macd_signal[-1]
        signals["volume_breakout"] = volume[-1] > volume_ma[-1] * 1.5

        return {"timeframe": timeframe, "signals": signals, "latest_price": close[-1]}

    def _generate_confluence_signals(self, timeframe_analysis: Dict) -> Dict[str, Any]:
        """Generate signals based on strict rules mapping."""
        confluence_signals = {}
        for rule_name, rule_def in self.confluence_rules.items():
            match_count = 0
            for tf, required_signals in rule_def["timeframe_conditions"].items():
                if tf in timeframe_analysis:
                    tf_signals = timeframe_analysis[tf]["signals"]
                    if all(tf_signals[sig] for sig in required_signals):
                        match_count += 1

            confluence_signals[rule_name] = match_count >= rule_def["min_timeframes"]
        return confluence_signals

    def _determine_final_recommendation(self, confluence_signals: Dict) -> Dict[str, Any]:
        """Determine recommendation based on rules with no fallbacks."""
        if confluence_signals["strong_multi_timeframe_buy"]:
            return {"recommendation": "STRONG_BUY", "confidence": 0.9}
        if confluence_signals["multi_timeframe_buy"]:
            return {"recommendation": "BUY", "confidence": 0.7}
        return {"recommendation": "HOLD", "confidence": 0.0}
