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

        from config import EPISODIC_PIVOT_MODE, VOLUME_SPIKE_THRESHOLD

        if EPISODIC_PIVOT_MODE:
            return {"signal": signal, "reason": "EP Mode: Allowing dry volume entry"}

        # strat_params from individual strategy configs don't contain volume_analysis_config
        # Only skip if volume config is explicitly present in strat_params
        if "volume_analysis_config" not in self.strat_params:
            return {"signal": signal, "reason": "Volume config not available, passing signal through"}

        min_v = VOLUME_SPIKE_THRESHOLD

        stype = "bullish" if signal == 1 else "bearish"
        v_analysis = get_enhanced_volume_confirmation(data, self.strat_params, stype)

        if v_analysis["factor"] >= min_v:
            return {"signal": signal, "reason": f"Vol OK: {v_analysis['strength']}"}
        else:
            return {"signal": 0, "reason": f"Vol Filtered: {v_analysis['strength']}"}


class BacktraderStrategyMeta(type(ABC), type(bt.Strategy)):
    pass


class BacktraderStrategy(bt.Strategy, metaclass=BacktraderStrategyMeta):
    params = (("symbol", "UNKNOWN"), ("strat_params", {}))

    def __init__(self, *args, **kwargs):
        bt.Strategy.__init__(self, *args, **kwargs)
        self.symbol = self.params.symbol
        self.strat_params = self.params.strat_params
        self.data_close = self.datas[0].close
        self.bar_executed = 0
        self.pyramid_adds_count = 0  # Track pyramid adds for current position
        self.last_add_price = 0.0  # Track last add price for pyramid triggers
        from scripts.swing_trading_signals import SwingTradingSignalAnalyzer

        self.analyzer = SwingTradingSignalAnalyzer()

    def apply_volume_filtering(self, signal, data):
        """Standardized volume filtering for backtests."""
        if signal == 0:
            return {"signal": 0, "reason": "No Signal"}

        stype = self.strat_params.get("name", "SWING")
        v_analysis = get_enhanced_volume_confirmation(data, self.strat_params, stype)
        min_v = self.strat_params.get("strategy_config", {}).get("volume_analysis", {}).get("min_volume_score", 0.1)

        if v_analysis["factor"] >= min_v:
            return {"signal": signal, "reason": f"Vol OK: {v_analysis['strength']}"}
        else:
            return {"signal": 0, "reason": f"Vol Filtered: {v_analysis['strength']}"}

    def next(self):
        # 1. Manage Active Position (Exit Logic)
        if self.position:
            # Check for Time Stop
            time_stop = self.strat_params.get("exit_rules", {}).get("time_stop_bars", 45)
            if len(self) - self.bar_executed >= time_stop:
                self.close(reason="Time Stop")

            current_price = self.datas[0].close[0]

            # 1. Manage Exits (Targets & Stops)
            exit_cfg = self.strat_params.get("exit_rules", {})
            targets = exit_cfg.get("targets", [])
            sl_mult = exit_cfg.get("atr_stop_multiplier", 2.0)

            # ATR for exit
            import talib as ta

            high = self.datas[0].high.get(size=14)
            low = self.datas[0].low.get(size=14)
            close = self.datas[0].close.get(size=14)
            atr = ta.ATR(pd.Series(high), pd.Series(low), pd.Series(close), 14).iloc[-1]

            # Initial Stop Loss logic
            if not hasattr(self, "current_stop_loss"):
                self.current_stop_loss = self.position.price - (atr * sl_mult)

            # Check Stop Loss
            if current_price < self.current_stop_loss:
                self.close(reason=f"Stop Loss Hit @ {self.current_stop_loss:.2f}")
                return

            # Check Targets
            if not hasattr(self, "targets_hit"):
                self.targets_hit = 0

            for i, target_cfg in enumerate(targets):
                if i < self.targets_hit:
                    continue

                target_price = self.position.price + (atr * target_cfg["atr_multiplier"])
                if current_price >= target_price:
                    sell_pct = target_cfg["sell_percentage"]
                    # Calculate quantity to sell
                    qty_to_sell = int(self.position.size * sell_pct)
                    if qty_to_sell > 0:
                        self.sell(size=qty_to_sell)
                        self.targets_hit += 1

                        # Breakeven logic
                        if i == 0 and exit_cfg.get("breakeven_at_target_1"):
                            self.current_stop_loss = self.position.price

                        logger.info(f"Target {i+1} Hit: {self.symbol} | Sold {qty_to_sell} units")
                        # If it was the last target, we'll be closed by the next check or final target

            # Trailing Stop Loss Update (matches portfolio_backtest_engine.py)
            if atr > 0:
                trail_mult = exit_cfg.get("trail_stop_atr", 2.5)
                new_sl = current_price - (atr * trail_mult)
                if new_sl > self.current_stop_loss:
                    logger.info(f"📉 {self.symbol}: Trailing SL updated {self.current_stop_loss:.2f} → {new_sl:.2f}")
                    self.current_stop_loss = new_sl

            # Pyramiding - Check for add triggers (matches portfolio_backtest_engine.py)
            pyramid_cfg = self.strat_params.get("pyramiding", {})
            if pyramid_cfg.get("enabled", False) and self.pyramid_adds_count < len(pyramid_cfg.get("steps", [])):
                steps = pyramid_cfg.get("steps", [])
                step = steps[self.pyramid_adds_count]
                trigger_mult = step.get("trigger_step_atr", 1.5)
                required_price = self.last_add_price + (trigger_mult * atr)

                if current_price >= required_price:
                    add_pct = step.get("add_size_pct", 0.5)
                    add_qty = max(int(self.position.size * add_pct), 1)

                    # Check if pyramid counts as new position
                    # Pyramid counting now from strategy config
                    PYRAMID_COUNTS_AS_NEW_POSITION = False  # Default

                    if PYRAMID_COUNTS_AS_NEW_POSITION:
                        # For single-stock backtest, skip this check (not applicable)
                        pass

                    # Check max position pct
                    max_position_pct = (
                        self.strat_params.get("risk_management", {}).get("max_position_pct", 10.0) / 100.0
                    )
                    portfolio_value = self.broker.get_value()
                    new_position_value = self.position.size * current_price + add_qty * current_price
                    if (new_position_value / portfolio_value) <= max_position_pct:
                        self.buy(size=add_qty)
                        self.pyramid_adds_count += 1
                        self.last_add_price = current_price
                        logger.info(
                            f"🔼 PYRAMID ADD {self.symbol} @ ₹{current_price:.2f} | Qty: {add_qty} | Step: {self.pyramid_adds_count}"
                        )

            return

        # 2. Look for New Entry
        df = pd.DataFrame(
            {
                "Open": self.datas[0].open.get(size=len(self)),
                "High": self.datas[0].high.get(size=len(self)),
                "Low": self.datas[0].low.get(size=len(self)),
                "Close": self.datas[0].close.get(size=len(self)),
                "Volume": self.datas[0].volume.get(size=len(self)),
            }
        )

        sig_res = self.analyzer.analyze_swing_opportunity(self.symbol, df, strategy_config=self.strat_params)
        sig = 1 if sig_res.get("recommendation") == "BUY" else 0

        if sig == 1:
            # Set stop loss for the sizer to calculate quantity
            trade_plan = sig_res.get("trade_plan", {})
            stop_loss = trade_plan.get("stop_loss")
            entry_price = self.datas[0].close[0]

            if stop_loss and stop_loss < entry_price:
                # Calculate Risk-Based Size
                risk_pct = self.strat_params.get("risk_management", {}).get("risk_per_trade_pct", 1.0)
                total_value = self.broker.get_value()
                total_risk_allowed = total_value * (risk_pct / 100.0)
                risk_per_share = entry_price - stop_loss
                size = int(total_risk_allowed / risk_per_share)
                size = max(size, 1)

                self.current_stop_loss = stop_loss
                self.pyramid_adds_count = 0  # Reset pyramid tracking for new position
                self.last_add_price = entry_price  # Set base price for pyramid triggers
                self.buy(size=size)
                logger.info(
                    f"BUY Signal: {self.symbol} | Price: {entry_price:.2f} | SL: {stop_loss:.2f} | Qty: {size} (Risk: {risk_pct}%)"
                )
            else:
                # Fallback
                self.pyramid_adds_count = 0
                self.last_add_price = self.datas[0].close[0]
                self.buy()

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.bar_executed = len(self)
            # Store the last executed size for the analyzer to pick up
            self.last_executed_size = abs(order.executed.size)
