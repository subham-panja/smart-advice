import importlib
import logging
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class StrategyEvaluator:
    """Dynamically loads and runs trading strategies based on a specific strategy configuration."""

    def __init__(self, strategy_config: Dict[str, Any] = None):
        self.strategy_config = strategy_config
        self.instances = {}
        if strategy_config:
            self._load(strategy_config.get("strategy_config", {}))

    def _load(self, cfg: Dict[str, Any]):
        strategy_map = {
            "MACD_Signal_Crossover": "scripts.strategies.macd_signal_crossover",
            "ADX_Trend_Strength": "scripts.strategies.adx_trend_strength",
            "On_Balance_Volume": "scripts.strategies.on_balance_volume",
            "Bollinger_Band_Squeeze": "scripts.strategies.bollinger_band_squeeze",
            "Pocket_Pivot_Entry": "scripts.strategies.pocket_pivot_entry",
            "Volume_Breakout": "scripts.strategies.volume_breakout",
            "Chart_Patterns": "scripts.strategies.chart_patterns",
            "RSI_Overbought_Oversold": "scripts.strategies.rsi_overbought_oversold",
        }
        for name, params in cfg.items():
            if params.get("enabled") and name in strategy_map:
                try:
                    mod = importlib.import_module(strategy_map[name])
                    cls = getattr(mod, "ChartPatterns" if name == "Chart_Patterns" else name)
                    self.instances[name] = cls(params)
                except Exception as e:
                    logger.error(f"Load error {name}: {e}")

    def evaluate_strategies(
        self, symbol: str, df: pd.DataFrame, app_config: Dict[str, Any] = None, index_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        if df.empty:
            return {"symbol": symbol, "technical_score": 0.0}

        # If app_config is provided, ensure instances are up to date for this strategy
        if app_config and (not self.strategy_config or self.strategy_config.get("name") != app_config.get("name")):
            self.strategy_config = app_config
            self.instances = {}
            self._load(app_config.get("strategy_config", {}))

        pos, total = 0, 0
        for name, inst in self.instances.items():
            try:
                sig = inst.run_strategy(df.copy(), symbol=symbol)
                if sig == 1:
                    pos += 1
                total += 1
            except Exception as e:
                logger.error(f"Strategy error {name} on {symbol}: {e}")

        rs_cfg = self.strategy_config.get("rs_config", {})
        if (
            self.strategy_config.get("strategy_config", {}).get("Relative_Strength_Comparison")
            and index_data is not None
        ):
            try:
                combined = pd.DataFrame({"s": df["Close"], "i": index_data["Close"]}).dropna()
                if not combined.empty:
                    r = (combined["s"] / combined["i"]) * 100
                    r_sma = r.rolling(rs_cfg.get("period", 55)).mean()
                    rs = ((r / r_sma) - 1) * 100
                    if rs.iloc[-1] > rs_cfg.get("threshold", 0):
                        pos += 1
                    total += 1
            except:
                pass

        score = 0.0
        if total > 0:
            # Check if this strategy uses EP detection logic
            # For now, we use a simple heuristic: if strategy name contains 'EP'
            is_ep_mode = "EP" in self.strategy_config.get("name", "").upper()

            if is_ep_mode:
                # EP DETECTION BYPASS: If recent 3x spike + holding 10EMA, award high score
                try:
                    # Catalyst Check (last 5 days)
                    v_rolling = df["Volume"].rolling(window=50).mean()
                    recent_spike = any(df["Volume"].tail(6).iloc[:-1] > v_rolling.tail(6).iloc[:-1] * 3.0)

                    # Pullback Check (Today)
                    ema10 = df["Close"].ewm(span=10).mean().iloc[-1]
                    is_resting = df["Low"].iloc[-1] <= ema10 * 1.03 and df["Close"].iloc[-1] > ema10 * 0.98

                    if recent_spike and is_resting:
                        # Perfect Delayed EP Setup
                        return {
                            "symbol": symbol,
                            "technical_score": 0.50,
                            "pos": pos,
                            "total": total,
                            "ep_pattern": True,
                        }
                except:
                    pass

                # Weighted fallback logic
                weights = {
                    "Volume_Breakout": 3.0,
                    "Pocket_Pivot_Entry": 3.0,
                    "Bollinger_Band_Squeeze": 2.0,
                    "MACD_Signal_Crossover": 1.0,
                    "ADX_Trend_Strength": 1.0,
                    "On_Balance_Volume": 2.0,
                    "Relative_Strength_Comparison": 1.5,
                }
                weighted_pos, weighted_total = 0.0, 0.0
                for name, inst in self.instances.items():
                    w = weights.get(name, 1.0)
                    try:
                        sig = inst.run_strategy(df.copy(), symbol=symbol)
                        if sig == 1:
                            weighted_pos += w
                        weighted_total += w
                    except:
                        pass
                score = weighted_pos / weighted_total if weighted_total > 0 else 0.0
            else:
                score = pos / total

        return {"symbol": symbol, "technical_score": score, "pos": pos, "total": total}
