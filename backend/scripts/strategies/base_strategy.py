import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import backtrader as bt
import pandas as pd

from utils.volume_analysis import get_enhanced_volume_confirmation

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    def __init__(self, params: Dict[str, Any]):
        self.strat_params = params
        self.name = self.__class__.__name__

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Strictly get parameter from strat_params.
        Accepts a 'default' argument for compatibility with legacy strategies,
        but IGNORES it to enforce that all parameters must be in the JSON config.
        """
        try:
            return self.strat_params[key]
        except KeyError:
            logger.error(f"Missing mandatory parameter '{key}' in strategy configuration.")
            raise

    @abstractmethod
    def _execute_strategy_logic(self, data: pd.DataFrame, symbol: str) -> int:
        pass

    def run_strategy(self, data: pd.DataFrame, symbol: str) -> int:
        """Run strategy logic with strict error propagation."""
        try:
            raw_signal = self._execute_strategy_logic(data, symbol=symbol)
            res = self.apply_volume_filtering(raw_signal, data)
            return res["signal"]
        except Exception as e:
            logger.error(f"Critical Strategy Error {self.name} on {symbol}: {e}")
            raise e

    def validate_data(self, data: pd.DataFrame, min_periods: int) -> bool:
        if data is None or data.empty or len(data) < min_periods:
            return False
        required = ["Open", "High", "Low", "Close", "Volume"]
        return all(col in data.columns for col in required)

    def log_signal(self, signal: int, reason: str, data: pd.DataFrame, symbol: str) -> None:
        stype = "BUY" if signal == 1 else "SELL/NO_BUY"
        close = data["Close"].iloc[-1]
        logger.debug(f"[{symbol}] {self.name}: {stype} signal - {reason} (Close: {close})")

    def apply_volume_filtering(self, signal: int, data: pd.DataFrame) -> Dict[str, Any]:
        """Apply volume filters strictly using config."""
        if signal == 0:
            return {"signal": 0, "reason": "No signal"}

        from config import EPISODIC_PIVOT_MODE, STOCK_FILTERING

        if EPISODIC_PIVOT_MODE:
            return {"signal": signal, "reason": "EP Mode: Allowing dry volume entry"}

        min_v = STOCK_FILTERING["require_volume_spike"]
        stype = "bullish" if signal == 1 else "bearish"
        v_analysis = get_enhanced_volume_confirmation(data, self.strat_params, stype)

        if v_analysis["factor"] >= min_v:
            return {"signal": signal, "reason": f"Vol OK: {v_analysis['strength']}"}
        else:
            return {"signal": 0, "reason": f"Vol Filtered: {v_analysis['strength']}"}


class BacktraderStrategyMeta(type(ABC), type(bt.Strategy)):
    pass


class BacktraderStrategy(BaseStrategy, bt.Strategy, metaclass=BacktraderStrategyMeta):
    params = (("symbol", "UNKNOWN"),)

    def __init__(self, *args, **kwargs):
        BaseStrategy.__init__(self, params=kwargs)
        bt.Strategy.__init__(self, *args, **kwargs)
        self.symbol = self.params.symbol
        self.data_close = self.datas[0].close

    def next(self):
        df = pd.DataFrame(
            {
                "Open": self.datas[0].open.get(size=len(self)),
                "High": self.datas[0].high.get(size=len(self)),
                "Low": self.datas[0].low.get(size=len(self)),
                "Close": self.datas[0].close.get(size=len(self)),
                "Volume": self.datas[0].volume.get(size=len(self)),
            }
        )
        sig = self.run_strategy(df, symbol=self.symbol)
        if sig == 1 and not self.position:
            self.buy()
        elif sig == -1 and self.position:
            self.close()
