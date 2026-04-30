import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import backtrader as bt
import pandas as pd

from utils.volume_analysis import get_enhanced_volume_confirmation

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}
        self.name = self.__class__.__name__

    def get_parameter(self, key: str, default: Any) -> Any:
        return self.params.get(key, default)

    @abstractmethod
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        pass

    def run_strategy(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> int:
        try:
            self.current_symbol = symbol
            raw_signal = self._execute_strategy_logic(data)
            return self.apply_volume_filtering(raw_signal, data)["signal"]
        except Exception as e:
            logger.error(f"Strategy Error {self.name}: {e}")
            return -1

    def validate_data(self, data: pd.DataFrame, min_periods: int = 1) -> bool:
        if data is None or data.empty or len(data) < min_periods:
            return False
        required = ["Open", "High", "Low", "Close", "Volume"]
        return all(col in data.columns for col in required)

    def log_signal(self, signal: int, reason: str, data: pd.DataFrame) -> None:
        stype = "BUY" if signal == 1 else "SELL/NO_BUY"
        close = data["Close"].iloc[-1] if not data.empty else "N/A"
        sym = getattr(self, "current_symbol", "UNKNOWN")
        logger.info(f"[{sym}] {self.name}: {stype} signal - {reason} (Close: {close})")

    def apply_volume_filtering(self, signal: int, data: pd.DataFrame) -> Dict[str, Any]:
        try:
            if signal == 0:
                return {"signal": 0, "reason": "No signal"}

            from config import STOCK_FILTERING

            min_v = STOCK_FILTERING.get("require_volume_spike", 1.5)
            stype = "bullish" if signal == 1 else "bearish"
            v_analysis = get_enhanced_volume_confirmation(data, stype)

            if v_analysis["factor"] >= min_v:
                return {"signal": signal, "reason": f"Vol OK: {v_analysis['strength']}"}
            else:
                return {"signal": 0, "reason": f"Vol Filtered: {v_analysis['strength']}"}
        except Exception as e:
            return {"signal": signal, "reason": f"Vol Error: {e}"}


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
