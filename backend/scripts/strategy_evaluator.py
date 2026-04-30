import importlib
import logging
from typing import Any, Dict

import pandas as pd

from config import RS_CONFIG, STRATEGY_CONFIG

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """Dynamically loads and runs trading strategies."""

    def __init__(self, cfg: Dict[str, bool] = None):
        self.cfg = cfg or STRATEGY_CONFIG
        self.instances = {}
        self._load()

    def _load(self):
        map = {
            "MACD_Signal_Crossover": "scripts.strategies.macd_signal_crossover",
            "ADX_Trend_Strength": "scripts.strategies.adx_trend_strength",
            "On_Balance_Volume": "scripts.strategies.on_balance_volume",
            "Bollinger_Band_Squeeze": "scripts.strategies.bollinger_band_squeeze",
            "Pocket_Pivot_Entry": "scripts.strategies.pocket_pivot_entry",
            "Volume_Breakout": "scripts.strategies.volume_breakout",
            "Chart_Patterns": "scripts.strategies.chart_patterns",
            "RSI_Overbought_Oversold": "scripts.strategies.rsi_overbought_oversold",
        }
        for name, params in self.cfg.items():
            if params.get("enabled") and name in map:
                try:
                    mod = importlib.import_module(map[name])
                    cls = getattr(mod, "ChartPatterns" if name == "Chart_Patterns" else name)
                    self.instances[name] = cls(params)
                except Exception as e:
                    logger.error(f"Load error {name}: {e}")

    def evaluate_strategies(self, symbol: str, df: pd.DataFrame, index_data: pd.DataFrame = None) -> Dict[str, Any]:
        if df.empty:
            return {"symbol": symbol, "technical_score": 0.0}

        pos, total = 0, 0
        for name, inst in self.instances.items():
            try:
                sig = inst.run_strategy(df.copy(), symbol=symbol)
                if sig == 1:
                    pos += 1
                total += 1
            except Exception as e:
                logger.error(f"Strategy error {name} on {symbol}: {e}")

        if self.cfg.get("Relative_Strength_Comparison") and index_data is not None:
            try:
                combined = pd.DataFrame({"s": df["Close"], "i": index_data["Close"]}).dropna()
                if not combined.empty:
                    r = (combined["s"] / combined["i"]) * 100
                    r_sma = r.rolling(RS_CONFIG.get("period", 55)).mean()
                    rs = ((r / r_sma) - 1) * 100
                    if rs.iloc[-1] > RS_CONFIG.get("threshold", 0):
                        pos += 1
                    total += 1
            except:
                pass

        score = pos / total if total > 0 else 0.0
        return {"symbol": symbol, "technical_score": score, "pos": pos, "total": total}
