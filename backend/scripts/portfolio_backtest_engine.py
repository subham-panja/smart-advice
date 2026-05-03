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

from config import PORTFOLIO_BACKTEST_CONFIG
from scripts.market_regime_detection import MarketRegimeDetection
from scripts.risk_management import RiskManager
from scripts.swing_trading_signals import SwingTradingSignalAnalyzer

PYRAMID_COUNTS_AS_NEW_POSITION = False  # Global default

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

        # Backtest config from config.py (strategy-level risk comes from strategy_config)
        cfg = capital_config if capital_config else PORTFOLIO_BACKTEST_CONFIG

        self.initial_capital = cfg.get("initial_capital", 100000.0)
        self.brokerage = cfg.get("brokerage_charges", 0.0020)
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
        # Check max daily loss circuit breaker
        exit_cfg = self.strategy_config.get("exit_rules", {})
        max_daily_loss_pct = exit_cfg.get("max_daily_loss_pct", None)
        if max_daily_loss_pct and self.daily_snapshots:
            prev_value = self.daily_snapshots[-1]["portfolio_value"]
            curr_value = self._current_portfolio_value(symbols_data, date)
            daily_loss_pct = ((curr_value - prev_value) / prev_value) * 100
            if daily_loss_pct <= -max_daily_loss_pct:
                logger.warning(
                    f"🛑 MAX DAILY LOSS TRIGGERED on {date.date()}: Loss {daily_loss_pct:.2f}% >= {max_daily_loss_pct}% threshold. "
                    f"Force closing all positions."
                )
                # Force close all positions
                for symbol in list(self.positions.keys()):
                    df = symbols_data.get(symbol)
                    last_price = df["Close"].iloc[-1] if df is not None else self.positions[symbol].entry_price
                    self._close_position(symbol, date, last_price, "MAX_DAILY_LOSS")
                return  # Skip rest of the day

        # --- Phase 1: Process Exits (before entries to free cash) ---
        exits_today = self._process_exits(date, symbols_data)
        _ = sum(e.pnl for e in exits_today if e.pnl > 0)  # cash freed, may be used for recycling logic

        # --- Phase 2: Scan for New Signals ---
        candidates = self._scan_for_signals(date, symbols_data)

        # --- Phase 3: Rank and Execute Buys ---
        if candidates:
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
        # Check max daily loss circuit breaker
        exit_cfg = self.strategy_config.get("exit_rules", {})
        max_daily_loss_pct = exit_cfg.get("max_daily_loss_pct", None)
        if max_daily_loss_pct and self.daily_snapshots:
            prev_value = self.daily_snapshots[-1]["portfolio_value"]
            curr_value = self._current_portfolio_value(symbols_data, date)
            daily_loss_pct = ((curr_value - prev_value) / prev_value) * 100
            if daily_loss_pct <= -max_daily_loss_pct:
                logger.warning(
                    f"🛑 MAX DAILY LOSS TRIGGERED on {date.date()}: Loss {daily_loss_pct:.2f}% >= {max_daily_loss_pct}% threshold."
                )
                for symbol in list(self.positions.keys()):
                    df = symbols_data.get(symbol)
                    last_price = df["Close"].iloc[-1] if df is not None else self.positions[symbol].entry_price
                    self._close_position(symbol, date, last_price, "MAX_DAILY_LOSS")
                return

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
        if candidates:
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
            t1_pct = regime_params.get("t1_sell_percentage", exit_cfg["targets"][0]["sell_percentage"])
            time_stop = regime_params.get("time_stop_bars", exit_cfg.get("time_stop_bars", 20))
            stop_loss_pct = regime_params.get("stop_loss_pct", None)  # O'Neil fixed % stop

            # 0. O'Neil Fixed Stop Loss (7-8% in bull, 5-6% in bear)
            if stop_loss_pct:
                oneil_stop = pos.entry_price * (1 - stop_loss_pct / 100.0)
                if current_price <= oneil_stop:
                    trade = self._close_position(symbol, date, current_price, f"ONEIL_STOP_{stop_loss_pct}%")
                    if trade:
                        exits.append(trade)
                    continue

            # 1. O'Neil Absolute Profit Target (20-25% in bull, 15% in bear)
            oneil_target_pct = regime_params.get("oneil_target_pct", 25.0)
            gain_pct = (current_price - pos.entry_price) / pos.entry_price * 100
            leader_cfg = exit_cfg.get("leader_exception", {})

            # Check 8-week leader exception
            is_leader = False
            if leader_cfg.get("enabled", False) and gain_pct >= leader_cfg.get("min_gain_pct", 20.0):
                if weeks_held <= leader_cfg.get("max_weeks", 8):
                    is_leader = True

            if gain_pct >= oneil_target_pct and not is_leader:
                # Sell at O'Neil target
                sell_qty = pos.quantity
                trade = self._close_position(symbol, date, current_price, f"ONEIL_TARGET_{oneil_target_pct:.0f}%")
                if trade:
                    exits.append(trade)
                continue

            # 8-week leader: hold and trail instead of selling
            if is_leader and leader_cfg.get("action") == "hold_and_trail":
                # Skip normal targets, just trail aggressively
                pass
            else:
                # 2. Time Stop (regime-adaptive)
                if bars_held >= time_stop:
                    trade = self._close_position(symbol, date, current_price, "TIME_STOP")
                    if trade:
                        exits.append(trade)
                    continue

                # 3. ATR Stop Loss
                if current_price <= pos.current_stop_loss:
                    trade = self._close_position(symbol, date, current_price, "STOP_LOSS")
                    if trade:
                        exits.append(trade)
                    continue

                # 4. ATR Targets (with regime-adaptive T1 %)
                targets = exit_cfg.get("targets", [])
                if pos.current_target_idx < len(targets):
                    target_cfg = targets[pos.current_target_idx].copy()
                    # Override sell percentage with regime-adaptive value for T1
                    if pos.current_target_idx == 0:
                        target_cfg["sell_percentage"] = t1_pct

                    target_price = pos.entry_price + (target_cfg["atr_multiplier"] * atr)

                    if current_price >= target_price:
                        sell_pct = target_cfg["sell_percentage"]
                        sell_qty = int(pos.quantity * sell_pct)

                        if sell_qty > 0 and sell_pct < 1.0:
                            trade = self._partial_sell(symbol, date, current_price, sell_qty, pos, target_cfg["name"])
                            if trade:
                                exits.append(trade)

                            pos.quantity -= sell_qty
                            pos.current_target_idx += 1
                            pos.is_scaled_out = True

                            if pos.current_target_idx == 1 and exit_cfg.get("breakeven_at_target_1"):
                                pos.current_stop_loss = pos.entry_price
                                logger.info(f"🛡️ {symbol}: SL moved to breakeven ₹{pos.entry_price:.2f}")

                            if pos.quantity <= 0:
                                symbols_to_remove.append(symbol)
                            continue
                        elif sell_pct >= 1.0:
                            trade = self._close_position(symbol, date, current_price, f"FINAL_{target_cfg['name']}")
                            if trade:
                                exits.append(trade)
                            continue

            # 5. Trailing Stop (regime-adaptive: ATR or MA-based)
            trail_type = regime_params.get("trail_stop_type", "atr")
            if trail_type == "ma" and len(df) >= regime_params.get("trail_stop_ma_period", 20):
                # Use moving average as trailing stop (Minervini style)
                ma_period = regime_params.get("trail_stop_ma_period", 20)
                ma_value = df["Close"].rolling(ma_period).mean().iloc[-1]
                if ma_value > pos.current_stop_loss:
                    pos.current_stop_loss = ma_value
                    logger.info(f"📉 {symbol}: MA{ma_period} trail SL updated to ₹{ma_value:.2f}")
            elif atr > 0:
                # ATR-based trailing stop (regime-adaptive multiplier)
                trail_mult = regime_params.get("trail_stop_atr_multiplier", exit_cfg.get("trail_stop_atr", 2.0))
                new_sl = current_price - (atr * trail_mult)
                if new_sl > pos.current_stop_loss:
                    logger.info(f"📉 {symbol}: Trailing SL updated {pos.current_stop_loss:.2f} → {new_sl:.2f}")
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

            # ATR trigger check
            hist = df.loc[:date]
            atr = self._calculate_atr(hist, date)
            step = steps[pos.adds_count]
            trigger_mult = step.get("trigger_step_atr", 1.5)
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
        Raises RuntimeError if regime detection is disabled — this strategy requires it.
        """
        if not self.regime_enabled or not self.regime_config:
            raise RuntimeError(
                "Market regime detection is DISABLED but required by this strategy. "
                "Set 'market_regime_detection': true in analysis_config of your strategy JSON. "
                "Regime detection controls exit behavior (stops, targets, pyramiding) — running without it is unsafe."
            )

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
                raise RuntimeError(
                    f"Insufficient index data for regime detection: {len(index_hist)} days < 250 required. "
                    f"Need at least 250 days of {index_symbol} data to calculate SMA for regime check."
                )

            # Parse the bull_market_rule (e.g., "latest close > sma(50)")
            import re

            rule = self.regime_config.get("bull_market_rule", "latest close > sma(50)")
            sma_match = re.search(r"sma\((\d+)\)", rule)
            if not sma_match:
                raise RuntimeError(
                    f"Invalid bull_market_rule in market_regime_config: '{rule}'. "
                    "Expected format: 'latest close > sma(N)' where N is the SMA period."
                )

            sma_period = int(sma_match.group(1))
            current_price = index_hist["Close"].iloc[-1]
            sma_value = index_hist["Close"].rolling(sma_period).mean().iloc[-1]

            is_bull = current_price > sma_value
            self._regime_status = "BULL" if is_bull else "BEAR"
            self._regime_check_date = date

            if not is_bull:
                logger.info(f"🔴 MACRO REGIME: BEARISH on {date.date()} - {index_symbol} below SMA({sma_period})")

            return self._regime_status

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"Market regime detection failed on {date}: {e}. "
                "This strategy requires regime detection to function. "
                "Check that market_regime_config.index is valid and data is accessible."
            ) from e

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
