import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import backtrader as bt
import pandas as pd
import talib as ta

from scripts.market_regime_detection import MarketRegimeDetection
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

        # Regime tracking (for regime-adaptive exits, risk, pyramiding)
        self._regime_status = "BULL"  # Default, updated via _check_regime()
        self._regime_detector = None
        self._regime_config = self.strat_params.get("market_regime_config", {})
        self._regime_enabled = self.strat_params.get("analysis_config", {}).get("market_regime_detection", False)

        # Volatility scaling cache (for regime-adaptive risk sizing)
        self._atr_percentile = None

        # Leader exception tracking
        self._is_leader = False
        self._leader_peak_gain = 0.0

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

    def _check_regime(self):
        """Detect market regime (BULL/BEAR) using NSE index vs SMA.

        Updates self._regime_status. Falls back to BULL if regime detection is disabled or fails.
        """
        if not self._regime_enabled or not self._regime_config:
            self._regime_status = "BULL"
            return

        try:
            if self._regime_detector is None:
                self._regime_detector = MarketRegimeDetection()

            result = self._regime_detector.get_simple_regime_check(self._regime_config)
            self._regime_status = result.get("status", "BULL")
        except Exception as e:
            logger.warning(f"[{self.symbol}] Regime detection failed: {e}. Defaulting to BULL.")
            self._regime_status = "BULL"

    def _calculate_volatility_scaling(self) -> float:
        """Calculate volatility-scaled risk multiplier (0.5x to 1.0x) based on ATR percentile.

        Returns risk multiplier. Low vol (<=25th %ile) = 1.0x, high vol (>=75th %ile) = 0.5x.
        """
        vol_cfg = self.strat_params.get("risk_management", {}).get("volatility_scaling", {})
        if not vol_cfg.get("enabled", False):
            return 1.0

        lookback = vol_cfg.get("atr_lookback_days", 100)
        low_pct = vol_cfg.get("low_volatility_percentile", 25)
        high_pct = vol_cfg.get("high_volatility_percentile", 75)
        min_mult = vol_cfg.get("min_risk_multiplier", 0.5)

        try:
            high = self.datas[0].high.get(size=lookback)
            low = self.datas[0].low.get(size=lookback)
            close = self.datas[0].close.get(size=lookback)

            if len(high) < 30:
                return 1.0

            atr_series = ta.ATR(pd.Series(high), pd.Series(low), pd.Series(close), 14)
            atr_values = atr_series.dropna().values

            if len(atr_values) < 20:
                return 1.0

            current_atr = atr_values[-1]
            historical_percentile = (atr_values < current_atr).sum() / len(atr_values) * 100

            if historical_percentile <= low_pct:
                return 1.0
            elif historical_percentile >= high_pct:
                return min_mult
            else:
                ratio = (historical_percentile - low_pct) / (high_pct - low_pct)
                return 1.0 - ratio * (1.0 - min_mult)
        except Exception:
            return 1.0

    def _get_regime_params(self) -> Dict[str, Any]:
        """Get regime-adaptive exit parameters for current regime."""
        exit_cfg = self.strat_params.get("exit_rules", {})
        regime_adaptive = exit_cfg.get("regime_adaptive_exits", {})
        regime_key = self._regime_status.lower()
        return regime_adaptive.get(regime_key, {})

    def _get_risk_per_trade(self) -> float:
        """Get regime-adaptive risk per trade percentage.

        Regime-adaptive risk overwrites volatility scaling when present.
        """
        risk_cfg = self.strat_params.get("risk_management", {})

        # 1. Check regime-adaptive risk first (overwrites volatility scaling)
        regime_risk_cfg = risk_cfg.get("regime_adaptive_risk", {})
        regime_key = self._regime_status.lower()
        if regime_key in regime_risk_cfg:
            return (
                regime_risk_cfg[regime_key].get("risk_per_trade_pct", risk_cfg.get("risk_per_trade_pct", 2.0)) / 100.0
            )

        # 2. Apply volatility scaling if no regime config
        vol_mult = self._calculate_volatility_scaling()
        base_risk = risk_cfg.get("risk_per_trade_pct", 2.0) / 100.0
        return base_risk * vol_mult

    def _get_entry_price(self) -> float:
        """Get the effective entry price for the current position."""
        return self.last_add_price if self.last_add_price > 0 else self.position.price

    def next(self):
        # Update regime check on every bar
        self._check_regime()

        # 1. Manage Active Position (Exit Logic)
        if self.position:
            current_price = self.datas[0].close[0]

            # ATR for exit calculations
            high = self.datas[0].high.get(size=14)
            low = self.datas[0].low.get(size=14)
            close = self.datas[0].close.get(size=14)
            atr = ta.ATR(pd.Series(high), pd.Series(low), pd.Series(close), 14).iloc[-1]

            # Initialize state on first bar of position
            if not hasattr(self, "current_stop_loss"):
                exit_cfg = self.strat_params.get("exit_rules", {})
                sl_mult = exit_cfg.get("atr_stop_multiplier", 1.5)
                self.current_stop_loss = self.position.price - (atr * sl_mult)

            if not hasattr(self, "targets_hit"):
                self.targets_hit = 0

            exit_cfg = self.strat_params.get("exit_rules", {})
            regime_params = self._get_regime_params()
            targets = exit_cfg.get("targets", [])
            leader_cfg = exit_cfg.get("leader_exception", {})

            # Calculate days/weeks held
            days_held = len(self) - self.bar_executed
            weeks_held = days_held / 5.0  # ~5 trading days per week

            gain_pct = (current_price - self.position.price) / self.position.price * 100

            # Track leader status
            if leader_cfg.get("enabled", False) and gain_pct >= leader_cfg.get("min_gain_pct", 20.0):
                if weeks_held <= leader_cfg.get("max_weeks", 8):
                    if not self._is_leader:
                        self._is_leader = True
                        logger.info(
                            f"🏆 LEADER: {self.symbol} | Gain: {gain_pct:.1f}% in {weeks_held:.1f}w | Holding & trailing"
                        )
                else:
                    self._is_leader = False  # No longer qualifies as leader

            # 0. O'Neil Fixed Stop Loss (regime-adaptive: 8% bull / 5% bear)
            stop_loss_pct = regime_params.get("stop_loss_pct", None)
            if stop_loss_pct:
                oneil_stop = self.position.price * (1 - stop_loss_pct / 100.0)
                if current_price <= oneil_stop:
                    self.close(reason=f"ONEIL_STOP_{stop_loss_pct}%")
                    return

            # 1. O'Neil Profit Target + Leader Exception
            oneil_target_pct = regime_params.get("oneil_target_pct", 25.0)
            if gain_pct >= oneil_target_pct and not self._is_leader:
                self.close(reason=f"ONEIL_TARGET_{oneil_target_pct:.0f}%")
                return

            # Leader: skip normal targets and time stop, only trail
            if self._is_leader and leader_cfg.get("action") == "hold_and_trail":
                pass  # Skip to trailing stop only
            else:
                # 2. Time Stop (regime-adaptive: 20 bars bull / 8 bars bear)
                time_stop = regime_params.get("time_stop_bars", exit_cfg.get("time_stop_bars", 12))
                if days_held >= time_stop:
                    self.close(reason="TIME_STOP")
                    return

                # 3. ATR Stop Loss
                if current_price < self.current_stop_loss:
                    self.close(reason=f"Stop Loss Hit @ {self.current_stop_loss:.2f}")
                    return

                # 4. ATR Targets
                if self.targets_hit < len(targets):
                    target_cfg = targets[self.targets_hit].copy()
                    # Regime-adaptive T1 sell percentage
                    if self.targets_hit == 0:
                        t1_pct = regime_params.get("t1_sell_percentage", target_cfg["sell_percentage"])
                        target_cfg["sell_percentage"] = t1_pct

                    target_price = self.position.price + (target_cfg["atr_multiplier"] * atr)

                    if current_price >= target_price:
                        sell_pct = target_cfg["sell_percentage"]
                        qty_to_sell = int(self.position.size * sell_pct)

                        if qty_to_sell > 0 and sell_pct < 1.0:
                            # Partial sell
                            self.sell(size=qty_to_sell)
                            self.targets_hit += 1

                            # Breakeven at T1
                            if self.targets_hit == 1 and exit_cfg.get("breakeven_at_target_1"):
                                self.current_stop_loss = self.position.price
                                logger.info(f"🛡️ {self.symbol}: SL moved to breakeven ₹{self.position.price:.2f}")

                            logger.info(f"Target {self.targets_hit} Hit: {self.symbol} | Sold {qty_to_sell} units")
                        elif sell_pct >= 1.0:
                            # Full exit
                            self.close(reason=f"FINAL_{target_cfg['name']}")
                        return

            # 5. Trailing Stop (regime-adaptive: ATR or MA-based)
            trail_type = regime_params.get("trail_stop_type", "atr")
            if trail_type == "ma" and len(close) >= regime_params.get("trail_stop_ma_period", 20):
                ma_period = regime_params.get("trail_stop_ma_period", 20)
                ma_value = pd.Series(close).rolling(ma_period).mean().iloc[-1]
                if ma_value > self.current_stop_loss:
                    logger.info(f"📉 {self.symbol}: MA{ma_period} trail SL updated to ₹{ma_value:.2f}")
                    self.current_stop_loss = ma_value
            elif atr > 0:
                trail_mult = regime_params.get("trail_stop_atr_multiplier", exit_cfg.get("trail_stop_atr", 2.0))
                new_sl = current_price - (atr * trail_mult)
                if new_sl > self.current_stop_loss:
                    logger.info(f"📉 {self.symbol}: Trailing SL updated {self.current_stop_loss:.2f} → {new_sl:.2f}")
                    self.current_stop_loss = new_sl

            # 6. Pyramiding (regime-controlled)
            self._check_pyramid(current_price, atr)

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
            trade_plan = sig_res.get("trade_plan", {})
            stop_loss = trade_plan.get("stop_loss")
            entry_price = self.datas[0].close[0]

            if stop_loss and stop_loss < entry_price:
                # Regime-adaptive + volatility-scaled risk sizing
                risk_pct = self._get_risk_per_trade() * 100.0  # Convert back to percentage for logging
                total_value = self.broker.get_value()
                total_risk_allowed = total_value * (risk_pct / 100.0)
                risk_per_share = entry_price - stop_loss
                size = int(total_risk_allowed / risk_per_share)
                size = max(size, 1)

                # Apply max_position_pct cap
                max_position_pct = self.strat_params.get("risk_management", {}).get("max_position_pct", 10.0) / 100.0
                max_size_by_pct = int((total_value * max_position_pct) / entry_price)
                size = min(size, max_size_by_pct)

                self.current_stop_loss = stop_loss
                self.targets_hit = 0
                self.pyramid_adds_count = 0
                self.last_add_price = entry_price
                self._is_leader = False
                self._leader_peak_gain = 0.0
                self.buy(size=size)
                logger.info(
                    f"BUY Signal: {self.symbol} | Price: ₹{entry_price:.2f} | SL: ₹{stop_loss:.2f} | "
                    f"Qty: {size} | Risk: {risk_pct:.1f}% | Regime: {self._regime_status}"
                )
            else:
                # Fallback
                self.pyramid_adds_count = 0
                self.last_add_price = self.datas[0].close[0]
                self.targets_hit = 0
                self._is_leader = False
                self.buy()

    def _check_pyramid(self, current_price: float, atr: float):
        """Check for pyramid add triggers with regime control.

        Blocked in bear markets when regime_controlled=true.
        """
        pyramid_cfg = self.strat_params.get("pyramiding", {})
        if not pyramid_cfg.get("enabled", False):
            return

        # Regime control: skip pyramiding in bear markets
        if pyramid_cfg.get("regime_controlled", False):
            regime_params = self._get_regime_params()
            if not regime_params.get("pyramiding_allowed", True):
                logger.debug(f"🚫 {self.symbol}: Pyramiding blocked in {self._regime_status} regime")
                return

        steps = pyramid_cfg.get("steps", [])
        if self.pyramid_adds_count >= len(steps):
            return

        step = steps[self.pyramid_adds_count]
        trigger_mult = step.get("trigger_step_atr", 1.5)
        required_price = self.last_add_price + (trigger_mult * atr)

        if current_price < required_price:
            return

        add_pct = step.get("add_size_pct", 0.5)
        add_qty = max(int(self.position.size * add_pct), 1)

        # Check max position pct
        max_position_pct = self.strat_params.get("risk_management", {}).get("max_position_pct", 10.0) / 100.0
        portfolio_value = self.broker.get_value()
        new_position_value = self.position.size * current_price + add_qty * current_price
        if (new_position_value / portfolio_value) > max_position_pct:
            return

        self.buy(size=add_qty)
        self.pyramid_adds_count += 1
        self.last_add_price = current_price
        logger.info(
            f"🔼 PYRAMID ADD {self.symbol} @ ₹{current_price:.2f} | Qty: {add_qty} | "
            f"Step: {self.pyramid_adds_count} | {step.get('name', '')}"
        )

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.bar_executed = len(self)
            # Store the last executed size for the analyzer to pick up
            self.last_executed_size = abs(order.executed.size)
