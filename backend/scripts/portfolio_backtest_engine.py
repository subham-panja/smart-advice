"""
Portfolio Backtest Engine
=========================

Simulates trading multiple stocks simultaneously on a shared timeline
with a single capital pool. Compounding happens across the entire portfolio.

Key features:
- Day-by-day timeline walker across all symbols
- Shared capital pool (default 10 Lakhs)
- Respects max_concurrent_positions and max_position_pct
- Ranks candidates by technical_score (swing signal strength)
- Same-day cash recycling (configurable)
- Pyramid support (configurable position counting)
- Force-close delisted stocks
- Daily portfolio snapshots for equity curve & drawdown
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import talib as ta

from config import PORTFOLIO_BACKTEST_CONFIG, PYRAMID_COUNTS_AS_NEW_POSITION
from scripts.market_regime_detection import MarketRegimeDetection
from scripts.risk_management import RiskManager
from scripts.swing_trading_signals import SwingTradingSignalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class PortfolioPosition:
    """Tracks an open position in the portfolio backtest."""

    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    quantity: int
    stop_loss: float
    current_target_idx: int = 0
    targets_hit: int = 0
    current_stop_loss: float = 0.0
    initial_quantity: int = 0
    adds_count: int = 0
    last_add_price: float = 0.0
    bar_executed: int = 0
    is_scaled_out: bool = False
    status: str = "OPEN"

    def __post_init__(self):
        if self.current_stop_loss == 0.0:
            self.current_stop_loss = self.stop_loss
        if self.last_add_price == 0.0:
            self.last_add_price = self.entry_price
        if self.initial_quantity == 0:
            self.initial_quantity = self.quantity


@dataclass
class PortfolioTrade:
    """Represents a completed trade for persistence."""

    symbol: str
    trade_type: str  # BUY, SELL, PARTIAL_SELL, PYRAMID_ADD
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    quantity: int = 0
    position_value: float = 0.0
    allocation_pct: float = 0.0
    stop_loss: float = 0.0
    target: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: Optional[str] = None
    portfolio_value_at_entry: float = 0.0
    cash_balance_at_entry: float = 0.0
    open_positions_count_at_entry: int = 0


class PortfolioBacktestSession:
    """Core portfolio backtest engine with multi-stock compounding."""

    def __init__(self, strategy_config: Dict[str, Any], capital_config: Optional[Dict] = None):
        self.strategy_config = strategy_config
        cfg = capital_config if capital_config else PORTFOLIO_BACKTEST_CONFIG

        self.initial_capital = cfg["initial_capital"]
        self.brokerage = cfg["brokerage_charges"]
        self.ranking_method = cfg.get("ranking_method", "combined_score")
        self.save_snapshots = cfg.get("save_daily_snapshots", True)
        self.same_day_recycling = cfg.get("same_day_cash_recycling", True)
        self.force_close_delisted = cfg.get("force_close_delisted", True)

        # Risk params from strategy's risk_management section (matches live trading)
        risk_cfg = strategy_config.get("risk_management", {})
        self.risk_per_trade = risk_cfg.get("risk_per_trade_pct", 2.0) / 100.0
        self.max_position_pct = risk_cfg.get("max_position_pct", 10.0) / 100.0
        self.max_positions = risk_cfg.get("max_positions", 15)

        # Global pyramid flag (matches live trading)
        self.pyramid_counts_as_new = PYRAMID_COUNTS_AS_NEW_POSITION

        # State
        self.cash = self.initial_capital
        self.peak_value = self.initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        self.trades: List[PortfolioTrade] = []
        self.daily_snapshots: List[dict] = []
        self.bar_count = 0

        # Daily loss tracking (max_daily_loss_pct circuit breaker)
        exit_rules = strategy_config.get("exit_rules", {})
        self.max_daily_loss_pct = exit_rules.get("max_daily_loss_pct", 0.0) / 100.0
        self.daily_loss = 0.0
        self.daily_loss_reset_date = None

        # Pause buying if bear market regime
        regime_cfg = strategy_config.get("market_regime_config", {})
        self.pause_buying_if_bearish = regime_cfg.get("pause_buying_if_bearish", False)
        self.adaptive_mode = regime_cfg.get("adaptive_mode", False)

        # Tools
        self.swing_analyzer = SwingTradingSignalAnalyzer()
        self.risk_manager = RiskManager(account_balance=self.initial_capital)

        # Market regime detection (from strategy config)
        self.regime_detector = MarketRegimeDetection()
        self.regime_config = strategy_config.get("market_regime_config", {})
        self.regime_enabled = strategy_config.get("analysis_config", {}).get("market_regime_detection", False)
        self._regime_status = "UNKNOWN"  # BULL or BEAR
        self._regime_check_date = None
        self._regime_check_cache = {}  # cache regime check per date

        # Results
        self.session_id: Any = None
        self.start_date: Optional[pd.Timestamp] = None
        self.end_date: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        sim_start_date: Optional[pd.Timestamp] = None,
        sim_end_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Run the portfolio backtest across all provided symbols.

        Args:
            symbols_data: Dict mapping symbol -> DataFrame (5yr OHLCV)
            sim_start_date: Optional start date for simulation (uses full data if None)
            sim_end_date: Optional end date for simulation

        Returns:
            Dict with session summary, trades, and per-stock metrics.
        """
        if not symbols_data:
            raise ValueError("No symbols data provided for portfolio backtest")

        logger.info(f"🚀 Starting portfolio backtest for strategy: {self.strategy_config['name']}")
        logger.info(f"   Capital: ₹{self.initial_capital:,.0f} | Max Positions: {self.max_positions}")

        # 1. Build common timeline
        common_dates = self._get_common_dates(symbols_data)
        if len(common_dates) < 100:
            raise ValueError(f"Insufficient common trading days: {len(common_dates)}")

        # Filter to simulation date range if provided
        if sim_start_date is not None:
            common_dates = common_dates[common_dates >= sim_start_date]
        if sim_end_date is not None:
            common_dates = common_dates[common_dates <= sim_end_date]

        if len(common_dates) < 100:
            raise ValueError(f"Insufficient common trading days after date filter: {len(common_dates)}")

        self.start_date = common_dates[0]
        self.end_date = common_dates[-1]

        logger.info(
            f"   Simulation range: {self.start_date.date()} → {self.end_date.date()} ({len(common_dates)} days)"
        )

        # 2. Pre-compute last available date per symbol (for delisted detection)
        self._last_dates = {sym: df.index[-1] for sym, df in symbols_data.items()}

        # 3. Day-by-day simulation
        for i, date in enumerate(common_dates):
            self.bar_count = i
            self._simulate_day(date, symbols_data)

        # 4. Force-close any remaining open positions at last price
        self._force_close_all_at_end(symbols_data)

        # 5. Calculate final metrics
        metrics = self._calculate_metrics(common_dates)

        logger.info(
            f"🏁 Portfolio backtest complete: Final Value ₹{metrics['final_portfolio_value']:,.0f} "
            f"| CAGR {metrics['cagr']:.1f}% | Max DD {metrics['max_drawdown_pct']:.1f}%"
        )

        return {
            "strategy_name": self.strategy_config["name"],
            "status": "completed",
            "initial_capital": self.initial_capital,
            "final_portfolio_value": metrics["final_portfolio_value"],
            "cash_remaining": self.cash,
            "total_return_pct": metrics["total_return_pct"],
            "cagr": metrics["cagr"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "expectancy": metrics["expectancy"],
            "avg_positions_held": metrics["avg_positions_held"],
            "trades": self.trades,
            "daily_snapshots": self.daily_snapshots,
            "date_range": {"start_date": str(self.start_date.date()), "end_date": str(self.end_date.date())},
        }

    def run_with_signals(
        self, symbols_data: Dict[str, pd.DataFrame], precomputed_signals: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Run portfolio backtest using pre-computed signals (from multiprocessing workers).

        Args:
            symbols_data: Dict mapping symbol -> DataFrame (OHLCV)
            precomputed_signals: Dict[symbol, Dict[date, {score, swing_result}]]

        Returns:
            Dict with session summary, trades, and per-stock metrics.
        """
        if not symbols_data:
            raise ValueError("No symbols data provided for portfolio backtest")

        logger.info(f"🚀 Starting portfolio backtest for strategy: {self.strategy_config['name']}")
        logger.info(f"   Capital: ₹{self.initial_capital:,.0f} | Max Positions: {self.max_positions}")
        logger.info(f"   Using pre-computed signals for {len(precomputed_signals)} symbols")

        # 1. Build common timeline
        common_dates = self._get_common_dates(symbols_data)
        if len(common_dates) < 100:
            raise ValueError(f"Insufficient common trading days: {len(common_dates)}")

        self.start_date = common_dates[0]
        self.end_date = common_dates[-1]

        logger.info(
            f"   Simulation range: {self.start_date.date()} → {self.end_date.date()} ({len(common_dates)} days)"
        )

        # 2. Pre-compute last available date per symbol
        self._last_dates = {sym: df.index[-1] for sym, df in symbols_data.items()}

        # 3. Day-by-day simulation using pre-computed signals
        for i, date in enumerate(common_dates):
            self.bar_count = i
            self._simulate_day_with_signals(date, symbols_data, precomputed_signals)

        # 4. Force-close any remaining open positions
        self._force_close_all_at_end(symbols_data)

        # 5. Calculate final metrics
        metrics = self._calculate_metrics(common_dates)

        logger.info(
            f"🏁 Portfolio backtest complete: Final Value ₹{metrics['final_portfolio_value']:,.0f} "
            f"| CAGR {metrics['cagr']:.1f}% | Max DD {metrics['max_drawdown_pct']:.1f}%"
        )

        return {
            "strategy_name": self.strategy_config["name"],
            "status": "completed",
            "initial_capital": self.initial_capital,
            "final_portfolio_value": metrics["final_portfolio_value"],
            "cash_remaining": self.cash,
            "total_return_pct": metrics["total_return_pct"],
            "cagr": metrics["cagr"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "total_trades": metrics["total_trades"],
            "win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "expectancy": metrics["expectancy"],
            "avg_positions_held": metrics["avg_positions_held"],
            "trades": self.trades,
            "daily_snapshots": self.daily_snapshots,
            "date_range": {"start_date": str(self.start_date.date()), "end_date": str(self.end_date.date())},
        }

    # ------------------------------------------------------------------
    # Simulation Core
    # ------------------------------------------------------------------

    def _simulate_day(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]):
        """Process a single trading day: exits → entries → pyramiding → snapshot."""
        # Reset daily loss tracker at start of each day
        date_str = str(date.date()) if hasattr(date, "date") else str(date)
        if self.daily_loss_reset_date != date_str:
            self.daily_loss = 0.0
            self.daily_loss_reset_date = date_str

        # --- Phase 1: Process Exits (before entries to free cash) ---
        exits_today = self._process_exits(date, symbols_data)
        _ = sum(e.pnl for e in exits_today if e.pnl > 0)  # cash freed, may be used for recycling logic

        # --- Phase 2: Scan for New Signals ---
        candidates = self._scan_for_signals(date, symbols_data)

        # --- Phase 3: Rank and Execute Buys ---
        # Check daily loss circuit breaker
        daily_loss_blocked = False
        if self.max_daily_loss_pct > 0:
            if (self.daily_loss / self.initial_capital) >= self.max_daily_loss_pct:
                daily_loss_blocked = True
                logger.debug(
                    f"Daily loss circuit breaker triggered: {self.daily_loss/self.initial_capital*100:.2f}% >= {self.max_daily_loss_pct*100:.1f}%"
                )

        # Check regime-based buying pause
        regime_blocked = False
        if self.pause_buying_if_bearish and self._regime_status == "BEAR":
            regime_blocked = True
            logger.debug("Bear market regime: buying paused per pause_buying_if_bearish config")

        if candidates and not daily_loss_blocked and not regime_blocked:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            for cand in candidates:
                if not self._can_open_new_position(cand["symbol"]):
                    continue
                self._execute_buy(cand, date, symbols_data)

        # --- Phase 4: Pyramiding ---
        self._process_pyramiding(date, symbols_data)

        # --- Phase 5: Record Snapshot ---
        if self.save_snapshots:
            self._record_snapshot(date, symbols_data)

    def _simulate_day_with_signals(
        self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame], precomputed_signals: Dict[str, Dict]
    ):
        """Process a single trading day using pre-computed signals (from multiprocessing workers)."""
        # --- Phase 1: Process Exits ---
        exits_today = self._process_exits(date, symbols_data)
        _ = sum(e.pnl for e in exits_today if e.pnl > 0)

        # --- Phase 2: Use pre-computed signals for this date ---
        candidates = []
        for symbol, date_signals in precomputed_signals.items():
            if symbol in self.positions:
                continue
            if date in date_signals:
                sig_data = date_signals[date]
                candidates.append(
                    {
                        "symbol": symbol,
                        "score": sig_data["score"],
                        "swing_result": sig_data["swing_result"],
                    }
                )

        # --- Phase 3: Rank and Execute Buys ---
        # Check daily loss circuit breaker
        daily_loss_blocked = False
        if self.max_daily_loss_pct > 0:
            if (self.daily_loss / self.initial_capital) >= self.max_daily_loss_pct:
                daily_loss_blocked = True
                logger.debug(
                    f"Daily loss circuit breaker triggered: {self.daily_loss/self.initial_capital*100:.2f}% >= {self.max_daily_loss_pct*100:.1f}%"
                )

        # Check regime-based buying pause
        regime_blocked = False
        if self.pause_buying_if_bearish and self._regime_status == "BEAR":
            regime_blocked = True
            logger.debug("Bear market regime: buying paused per pause_buying_if_bearish config")

        if candidates and not daily_loss_blocked and not regime_blocked:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            for cand in candidates:
                if not self._can_open_new_position(cand["symbol"]):
                    continue
                self._execute_buy(cand, date, symbols_data)

        # --- Phase 4: Pyramiding ---
        self._process_pyramiding(date, symbols_data)

        # --- Phase 5: Record Snapshot ---
        if self.save_snapshots:
            self._record_snapshot(date, symbols_data)

    def _process_exits(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> List[PortfolioTrade]:
        """Check all open positions for SL, target, time-stop, O'Neil rules, or delisted.

        Regime-adaptive: exit parameters change based on market regime (bull/bear).
        """
        exits = []
        symbols_to_remove = []
        exit_cfg = self.strategy_config.get("exit_rules", {})
        regime_adaptive = exit_cfg.get("regime_adaptive_exits", {})

        # Get current regime
        regime = self._regime_status.lower() if self._regime_status != "UNKNOWN" else "bull"
        regime_params = regime_adaptive.get(regime, {})

        for symbol, pos in list(self.positions.items()):
            df = symbols_data.get(symbol)
            if df is None or date not in df.index:
                if self.force_close_delisted:
                    last_price = df["Close"].iloc[-1] if df is not None else pos.entry_price
                    trade = self._close_position(symbol, date, last_price, "DELISTED")
                    if trade:
                        exits.append(trade)
                continue

            current_price = df.loc[date, "Close"]
            atr = self._calculate_atr(df, date)
            bars_held = self.bar_count - pos.bar_executed
            days_held = (date - pos.entry_date).days if hasattr(date, "__sub__") else bars_held
            weeks_held = days_held / 7.0

            # Get regime-specific parameters (fall back to base config)
            time_stop = regime_params.get("time_stop_bars", exit_cfg.get("time_stop_bars", 0))

            # ATR-based stop loss check
            atr_stop_mult = regime_params.get("atr_stop_multiplier", exit_cfg.get("atr_stop_multiplier", 0))
            if atr_stop_mult > 0 and atr > 0:
                current_stop = pos.entry_price - (atr * atr_stop_mult)
                if pos.current_stop_loss > 0:
                    current_stop = max(current_stop, pos.current_stop_loss)
                if current_price <= current_stop:
                    trade = self._close_position(symbol, date, current_price, "ATR_STOP")
                    if trade:
                        exits.append(trade)
                    continue

            # Target exits (ATR-based, support 1-N targets)
            gain_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            targets = exit_cfg.get("targets", [])
            leader_cfg = exit_cfg.get("leader_exception", {})

            # Leader exception: significant gain in short time = hold and trail
            is_leader = False
            days_held = (date - pos.entry_date).days if hasattr(date, "__sub__") else bars_held
            weeks_held = days_held / 7.0
            if leader_cfg.get("enabled", False) and gain_pct >= leader_cfg.get("min_gain_pct", 20.0):
                if weeks_held <= leader_cfg.get("max_weeks", 8):
                    is_leader = True

            # Check targets in order
            if not is_leader and pos.current_target_idx < len(targets):
                target = targets[pos.current_target_idx]
                target_atr_mult = target.get("atr_multiplier", 0)
                if target_atr_mult > 0 and atr > 0:
                    target_price = pos.entry_price + (atr * target_atr_mult)
                    if current_price >= target_price:
                        sell_qty = int(pos.quantity * target["sell_percentage"])
                        if sell_qty > 0:
                            trade = self._partial_sell(
                                symbol, date, current_price, sell_qty, pos, f"T{pos.current_target_idx+1}_ATR"
                            )
                            if trade:
                                exits.append(trade)
                            pos.quantity -= sell_qty
                            pos.current_target_idx += 1
                            pos.is_scaled_out = True
                            # If this was the last target, close remaining
                            if pos.current_target_idx >= len(targets):
                                trade = self._close_position(symbol, date, current_price, "FINAL_TARGET")
                                if trade:
                                    exits.append(trade)
                                continue
                            if exit_cfg.get("breakeven_at_target_1"):
                                pos.current_stop_loss = pos.entry_price

            # Time stop
            if time_stop > 0 and bars_held >= time_stop:
                trade = self._close_position(symbol, date, current_price, "TIME_STOP")
                if trade:
                    exits.append(trade)
                continue

            # Trailing Stop (regime-adaptive: MA or ATR-based)
            trail_type = regime_params.get("trail_stop_type", exit_cfg.get("trail_stop_type", "atr"))
            if trail_type == "ma":
                ma_period = regime_params.get("trail_stop_ma_period", exit_cfg.get("trail_stop_ma_period", 20))
                if len(df) >= ma_period:
                    ma_value = df["Close"].rolling(ma_period).mean().iloc[-1]
                    if ma_value > pos.current_stop_loss:
                        pos.current_stop_loss = ma_value
            elif atr > 0:
                trail_mult = regime_params.get(
                    "trail_stop_atr_multiplier", exit_cfg.get("trail_stop_atr_multiplier", 2.0)
                )
                if trail_mult > 0:
                    new_sl = current_price - (atr * trail_mult)
                    if new_sl > pos.current_stop_loss:
                        pos.current_stop_loss = new_sl

        # Cleanup closed positions
        for sym in symbols_to_remove:
            if sym in self.positions:
                del self.positions[sym]

        return exits

    def _scan_for_signals(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> List[dict]:
        """Scan stocks without open positions for BUY signals."""
        candidates = []

        # Check regime for adaptive behavior (but don't block entries)
        self._check_market_regime(date, symbols_data)

        for symbol, df in symbols_data.items():
            if symbol in self.positions:
                continue
            if date not in df.index:
                continue

            # Truncate data up to current date (no look-ahead)
            hist = df.loc[:date]
            if len(hist) < 50:
                continue

            try:
                swing = self.swing_analyzer.analyze_swing_opportunity(
                    symbol, hist, strategy_config=self.strategy_config
                )

                if swing.get("all_gates_passed") and swing.get("recommendation") == "BUY":
                    score = swing.get("technical_score", 0.0)
                    candidates.append(
                        {
                            "symbol": symbol,
                            "score": score,
                            "swing_result": swing,
                        }
                    )
            except Exception as e:
                logger.debug(f"Signal scan error for {symbol} on {date}: {e}")

        return candidates

    def _execute_buy(self, candidate: dict, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]):
        """Execute a BUY order with portfolio-aware position sizing."""
        symbol = candidate["symbol"]
        df = symbols_data[symbol]
        current_price = df.loc[date, "Close"]

        # Calculate risk params using initial capital (not inflated portfolio value)
        portfolio_value = self._current_portfolio_value(symbols_data, date)
        self.risk_manager.balance = self.initial_capital

        hist = df.loc[:date]
        risk = self.risk_manager.calculate_risk_params(hist, current_price, self.strategy_config, self._regime_status)

        if not risk.get("risk_reward_ok"):
            return

        size = risk["position_size"]
        if size <= 0:
            return

        # Respect max_position_pct
        max_by_pct = int((portfolio_value * self.max_position_pct) / current_price)
        size = min(size, max_by_pct)

        # Respect available cash
        cost = size * current_price
        if cost > self.cash:
            max_by_cash = int(self.cash / current_price)
            size = min(size, max_by_cash)

        if size <= 0:
            return

        cost = size * current_price
        brokerage_cost = cost * self.brokerage
        total_cost = cost + brokerage_cost

        if total_cost > self.cash:
            return

        # Deduct cash
        self.cash -= total_cost

        # Create position
        position = PortfolioPosition(
            symbol=symbol,
            entry_date=date,
            entry_price=current_price,
            quantity=size,
            stop_loss=risk["stop_loss"],
            current_stop_loss=risk["stop_loss"],
            bar_executed=self.bar_count,
        )
        self.positions[symbol] = position

        trade = PortfolioTrade(
            symbol=symbol,
            trade_type="BUY",
            entry_date=str(date.date()),
            entry_price=current_price,
            quantity=size,
            position_value=cost,
            allocation_pct=(cost / self.initial_capital) * 100,
            stop_loss=risk["stop_loss"],
            target=risk["targets"].get("T1") if risk.get("targets") else None,
            portfolio_value_at_entry=portfolio_value,
            cash_balance_at_entry=self.cash,
            open_positions_count_at_entry=len(self.positions) - 1,
        )
        self.trades.append(trade)

        logger.info(
            f"🟢 BUY {symbol} @ ₹{current_price:.2f} | Qty: {size} | "
            f"SL: ₹{risk['stop_loss']:.2f} | Cash left: ₹{self.cash:,.0f}"
        )

    def _process_pyramiding(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]):
        """Check existing positions for pyramid add triggers.

        Regime-controlled: pyramiding disabled in bear markets.
        """
        pyramid_cfg = self.strategy_config.get("pyramiding", {})
        if not pyramid_cfg.get("enabled", False):
            return

        # Regime control: skip pyramiding in bear markets
        if pyramid_cfg.get("regime_controlled", False):
            regime = self._regime_status.lower() if self._regime_status != "UNKNOWN" else "bull"
            regime_adaptive = self.strategy_config.get("exit_rules", {}).get("regime_adaptive_exits", {})
            regime_params = regime_adaptive.get(regime, {})
            if not regime_params.get("pyramiding_allowed", True):
                return  # No pyramiding in bear regime

        steps = pyramid_cfg.get("steps", [])

        for symbol, pos in list(self.positions.items()):
            if pos.adds_count >= len(steps):
                continue

            df = symbols_data.get(symbol)
            if df is None or date not in df.index:
                continue

            current_price = df.loc[date, "Close"]

            # Trigger check: profit percentage or ATR
            hist = df.loc[:date]
            atr = self._calculate_atr(hist, date)
            step = steps[pos.adds_count]
            trigger_profit_pct = step.get("trigger_profit_pct", None)
            trigger_mult = step.get("trigger_step_atr", 1.5)

            if trigger_profit_pct:
                # Profit-based trigger (Minervini style)
                gain_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                if gain_pct < trigger_profit_pct:
                    continue
            else:
                # ATR-based trigger
                required_price = pos.last_add_price + (trigger_mult * atr)
                if current_price < required_price:
                    continue

            # Calculate pyramid quantity based on original position size
            base_qty = pos.initial_quantity
            add_pct = step.get("add_size_pct", 0.5)
            add_qty = max(int(base_qty * add_pct), 1)

            # Check if pyramid counts as new position
            if self.pyramid_counts_as_new:
                if len(self.positions) >= self.max_positions:
                    continue

            # Check cash
            cost = add_qty * current_price
            brokerage_cost = cost * self.brokerage
            total_cost = cost + brokerage_cost
            if total_cost > self.cash:
                continue

            # Check max position pct
            portfolio_value = self._current_portfolio_value(symbols_data, date)
            current_position_value = pos.quantity * current_price
            new_position_value = current_position_value + cost
            if (new_position_value / portfolio_value) > self.max_position_pct:
                continue

            # Execute pyramid
            self.cash -= total_cost
            pos.quantity += add_qty
            pos.adds_count += 1
            pos.last_add_price = current_price

            trade = PortfolioTrade(
                symbol=symbol,
                trade_type="PYRAMID_ADD",
                entry_date=str(date.date()),
                entry_price=current_price,
                quantity=add_qty,
                position_value=cost,
                portfolio_value_at_entry=portfolio_value,
                cash_balance_at_entry=self.cash,
                open_positions_count_at_entry=len(self.positions),
            )
            self.trades.append(trade)

            logger.info(
                f"⬆️ PYRAMID {symbol} | Added {add_qty} @ ₹{current_price:.2f} | "
                f"New Qty: {pos.quantity} | Step: {step.get('name', 'Add')}"
            )

    def _close_position(self, symbol: str, date: pd.Timestamp, price: float, reason: str) -> Optional[PortfolioTrade]:
        """Close a position fully and record the trade."""
        pos = self.positions.get(symbol)
        if not pos:
            return None

        gross_value = pos.quantity * price
        brokerage_cost = gross_value * self.brokerage
        net_value = gross_value - brokerage_cost

        pnl = net_value - (pos.quantity * pos.entry_price)
        pnl_pct = (pnl / (pos.quantity * pos.entry_price)) * 100 if pos.entry_price > 0 else 0

        self.cash += net_value

        trade = PortfolioTrade(
            symbol=symbol,
            trade_type="SELL",
            entry_date=str(pos.entry_date.date()),
            entry_price=pos.entry_price,
            exit_date=str(date.date()),
            exit_price=price,
            quantity=pos.quantity,
            position_value=gross_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            portfolio_value_at_entry=self._current_portfolio_value_at_date(date),
            cash_balance_at_entry=self.cash - net_value,
            open_positions_count_at_entry=len(self.positions),
        )
        self.trades.append(trade)

        del self.positions[symbol]

        emoji = "🟢" if pnl >= 0 else "🔴"
        logger.info(
            f"{emoji} SELL {symbol} @ ₹{price:.2f} | Reason: {reason} | "
            f"PnL: ₹{pnl:+,.0f} ({pnl_pct:+.2f}%) | Cash: ₹{self.cash:,.0f}"
        )

        return trade

    def _partial_sell(
        self, symbol: str, date: pd.Timestamp, price: float, qty: int, pos: PortfolioPosition, reason: str
    ) -> PortfolioTrade:
        """Sell a portion of a position."""
        gross_value = qty * price
        brokerage_cost = gross_value * self.brokerage
        net_value = gross_value - brokerage_cost

        cost_basis = qty * pos.entry_price
        pnl = net_value - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        self.cash += net_value

        trade = PortfolioTrade(
            symbol=symbol,
            trade_type="PARTIAL_SELL",
            entry_date=str(pos.entry_date.date()),
            entry_price=pos.entry_price,
            exit_date=str(date.date()),
            exit_price=price,
            quantity=qty,
            position_value=gross_value,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )
        self.trades.append(trade)

        logger.info(f"📤 PARTIAL SELL {symbol} | {qty} @ ₹{price:.2f} | Reason: {reason}")

        return trade

    def _force_close_all_at_end(self, symbols_data: Dict[str, pd.DataFrame]):
        """Close any remaining open positions at the last available price."""
        for symbol, pos in list(self.positions.items()):
            df = symbols_data.get(symbol)
            last_price = df["Close"].iloc[-1] if df is not None else pos.entry_price
            self._close_position(symbol, self.end_date, last_price, "SIMULATION_END")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_market_regime(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> str:
        """Check market regime (BULL/BEAR) using NIFTY 50 index.

        Returns 'BULL' or 'BEAR'. Caches result per date to avoid redundant checks.
        """
        if not self.regime_enabled or not self.regime_config:
            return "BULL"  # If regime detection disabled, assume BULL

        # Cache by date
        if date == self._regime_check_date:
            return self._regime_status

        try:
            # Get NIFTY data from symbols_data if available, otherwise fetch
            index_symbol = self.regime_config.get("index", "^NSEI")
            if index_symbol in symbols_data:
                index_df = symbols_data[index_symbol]
                index_hist = index_df.loc[:date]
            else:
                # Fetch index data on-the-fly
                from scripts.data_fetcher import get_historical_data

                full_data = get_historical_data(index_symbol, period="5y")
                index_hist = full_data.loc[:date]

            if len(index_hist) < 250:
                return "BULL"  # Not enough data for SMA(200), assume BULL

            # Parse the bull_market_rule (e.g., "latest close > sma(50)")
            import re

            rule = self.regime_config.get("bull_market_rule", "latest close > sma(50)")
            sma_match = re.search(r"sma\((\d+)\)", rule)
            if not sma_match:
                return "BULL"

            sma_period = int(sma_match.group(1))
            current_price = index_hist["Close"].iloc[-1]
            sma_value = index_hist["Close"].rolling(sma_period).mean().iloc[-1]

            is_bull = current_price > sma_value
            self._regime_status = "BULL" if is_bull else "BEAR"
            self._regime_check_date = date

            if not is_bull:
                logger.info(f"🔴 MACRO REGIME: BEARISH on {date.date()} - {index_symbol} below SMA({sma_period})")

            return self._regime_status

        except Exception as e:
            logger.warning(f"Market regime check error on {date}: {e}")
            return "BULL"  # Default to BULL on error

    def _get_common_dates(self, symbols_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Build a union of all trading dates across symbols."""
        all_dates = set()
        for df in symbols_data.values():
            all_dates.update(df.index)
        return pd.DatetimeIndex(sorted(all_dates))

    def _current_portfolio_value(self, symbols_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        """Calculate total portfolio value = cash + market value of open positions."""
        market_value = 0.0
        for symbol, pos in self.positions.items():
            df = symbols_data.get(symbol)
            if df is not None and date in df.index:
                market_value += pos.quantity * df.loc[date, "Close"]
            else:
                market_value += pos.quantity * pos.entry_price
        return self.cash + market_value

    def _current_portfolio_value_at_date(self, date: pd.Timestamp) -> float:
        """Return cached portfolio value if available, else approximate."""
        # Used during exit recording; approximate is fine
        return self.cash + sum(pos.quantity * pos.entry_price for pos in self.positions.values())

    def _calculate_atr(self, df: pd.DataFrame, date: pd.Timestamp, period: int = 14) -> float:
        """Calculate ATR using data up to given date."""
        try:
            atr_series = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=period)
            return float(atr_series.loc[date]) if date in atr_series.index else float(atr_series.iloc[-1])
        except Exception:
            return 0.0

    def _can_open_new_position(self, symbol: str) -> bool:
        """Check if we can open a new position (regime-adaptive)."""
        if symbol in self.positions:
            return False

        # Regime-adaptive max positions
        regime = self._regime_status.lower() if self._regime_status != "UNKNOWN" else "bull"
        regime_risk_cfg = self.strategy_config.get("risk_management", {}).get("regime_adaptive_risk", {})
        max_pos = self.max_positions  # default
        if regime in regime_risk_cfg:
            max_pos = regime_risk_cfg[regime].get("max_positions", self.max_positions)

        current_slot_count = len(self.positions)
        return current_slot_count < max_pos

    def _record_snapshot(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]):
        """Save daily portfolio state."""
        portfolio_value = self._current_portfolio_value(symbols_data, date)
        market_value = portfolio_value - self.cash

        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        drawdown = portfolio_value - self.peak_value
        drawdown_pct = (drawdown / self.peak_value) * 100 if self.peak_value > 0 else 0

        self.daily_snapshots.append(
            {
                "date": str(date.date()),
                "portfolio_value": round(portfolio_value, 2),
                "cash_balance": round(self.cash, 2),
                "market_value": round(market_value, 2),
                "open_positions_count": len(self.positions),
                "open_positions": list(self.positions.keys()),
                "drawdown_from_peak": round(drawdown, 2),
                "drawdown_from_peak_pct": round(drawdown_pct, 2),
            }
        )

    def _calculate_metrics(self, common_dates: pd.DatetimeIndex) -> Dict[str, Any]:
        """Calculate final portfolio performance metrics."""
        final_value = self.daily_snapshots[-1]["portfolio_value"] if self.daily_snapshots else self.cash
        total_return_pct = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # CAGR
        days = (common_dates[-1] - common_dates[0]).days
        years = days / 365.25
        cagr = ((final_value / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0.1 else 0.0

        # Max Drawdown
        max_dd_pct = min((s["drawdown_from_peak_pct"] for s in self.daily_snapshots), default=0)

        # Trade metrics
        completed_trades = [t for t in self.trades if t.trade_type == "SELL"]
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        total_trades = len(completed_trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl <= 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / (total_trades - len(winning_trades)) if total_trades > len(winning_trades) else 1
        expectancy = ((win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)) if total_trades > 0 else 0.0

        # Sharpe ratio (from daily returns)
        daily_returns = []
        for i in range(1, len(self.daily_snapshots)):
            prev = self.daily_snapshots[i - 1]["portfolio_value"]
            curr = self.daily_snapshots[i]["portfolio_value"]
            daily_returns.append((curr - prev) / prev if prev > 0 else 0)

        sharpe = 0.0
        if daily_returns and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)

        avg_positions = (
            sum(s["open_positions_count"] for s in self.daily_snapshots) / len(self.daily_snapshots)
            if self.daily_snapshots
            else 0
        )

        return {
            "final_portfolio_value": round(final_value, 2),
            "total_return_pct": round(total_return_pct, 2),
            "cagr": round(cagr, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "sharpe_ratio": round(sharpe, 2),
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "expectancy": round(expectancy, 2),
            "avg_positions_held": round(avg_positions, 2),
        }
