"""Portfolio Backtest Engine — Simulates multi-stock trading on a shared capital pool timeline."""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import talib as ta

from config import PORTFOLIO_BACKTEST_CONFIG
from scripts.backtest_metrics import (
    calculate_portfolio_metrics,
    check_market_breadth,
    check_market_regime,
    record_daily_snapshot,
)
from scripts.backtest_position_manager import (
    PortfolioPosition,
    PortfolioTrade,
)
from scripts.market_regime_detection import MarketRegimeDetection
from scripts.risk_management import RiskManager
from scripts.swing_trading_signals import SwingTradingSignalAnalyzer

try:
    import scripts.execution_costs  # noqa: F401

    _HAS_EXECUTION_COSTS = True
except ImportError:
    _HAS_EXECUTION_COSTS = False

PYRAMID_COUNTS_AS_NEW_POSITION = False  # Global default

logger = logging.getLogger(__name__)


class PortfolioBacktestSession:
    """Core portfolio backtest engine with multi-stock compounding."""

    def __init__(
        self,
        strategy_config: Dict[str, Any],
        capital_config: Optional[Dict] = None,
        excluded_date_ranges: Optional[List[tuple]] = None,
    ):
        self.strategy_config = strategy_config
        self.excluded_date_ranges = excluded_date_ranges or []

        # Backtest config from config.py (strategy-level risk comes from strategy_config)
        cfg = capital_config if capital_config else PORTFOLIO_BACKTEST_CONFIG

        self.initial_capital = cfg.get("initial_capital", 100000.0)
        self.brokerage = cfg.get("brokerage_charges", 0.0020)
        self.slippage = cfg.get("slippage_pct", 0.0005)  # 0.05% slippage on execution
        self.ranking_method = cfg.get("ranking_method", "combined_score")
        self.save_snapshots = cfg.get("save_daily_snapshots", True)
        self.same_day_recycling = cfg.get("same_day_cash_recycling", True)
        self.force_close_delisted = cfg.get("force_close_delisted", True)

        # Execution realism: gap risk seed for reproducibility
        self.gap_seed = cfg.get("gap_risk_seed", 42)

        # Use execution_costs module if available (realistic Indian market costs)
        self.use_realistic_costs = _HAS_EXECUTION_COSTS

        # Risk params from strategy's risk_management section (matches live trading)
        risk_cfg = strategy_config.get("risk_management", {})
        self.risk_per_trade = risk_cfg.get("risk_per_trade_pct", 2.0) / 100.0
        self.max_position_pct = risk_cfg.get("max_position_pct", 10.0) / 100.0
        self.max_positions = risk_cfg.get("max_positions", 15)

        # Global pyramid flag (matches live trading)
        self.pyramid_counts_as_new = PYRAMID_COUNTS_AS_NEW_POSITION

        # Multi-timeframe execution configuration
        mtf_cfg = strategy_config.get("multitimeframe_execution", {})
        self.mtf_enabled = mtf_cfg.get("enabled", False)
        self.mtf_lower_tf = mtf_cfg.get("lower_timeframe", "60m")
        self.mtf_shortlist_n = mtf_cfg.get("shortlist_top_n", 10)

        # State
        self.cash = self.initial_capital
        self.peak_value = self.initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        self.trades: List[PortfolioTrade] = []
        self.daily_snapshots: List[dict] = []
        self.bar_count = 0

        self._cached_pv = None
        self._cached_pv_date = None
        self._dd_pause_active = False
        self._dd_pause_start_bar = None
        self._dd_pause_cfg = risk_cfg.get("drawdown_pause", {})
        self._close_prices = {}
        self._atr_cache = {}  # {symbol: {date: atr_value}} — pre-computed ATR fallback
        self._date_idx = {}  # {symbol: {date: int_position}} — for iloc slicing

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

        # Pre-fetched index data override (avoids union-date regime detection issues)
        self._index_data_override: Optional[pd.DataFrame] = None

        # Results
        self.session_id: Any = None
        self.start_date: Optional[pd.Timestamp] = None
        self.end_date: Optional[pd.Timestamp] = None

        # Pre-computed indicator store (optional, for vectorbt acceleration)
        self._indicator_store: Any = None

        # Per-date stock prefilter Boolean DataFrame (dates x symbols)
        self._stock_prefilter: Any = None

    def set_indicator_store(self, store: Any) -> None:
        """Set a pre-computed indicator store for accelerated signal generation."""
        self._indicator_store = store

    # Public API

    def run(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        sim_start_date: Optional[pd.Timestamp] = None,
        sim_end_date: Optional[pd.Timestamp] = None,
        excluded_date_ranges: Optional[List[tuple]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run the portfolio backtest across all provided symbols.

        Args:
            symbols_data: Dict mapping symbol -> DataFrame (5yr OHLCV)
            sim_start_date: Optional start date for simulation (uses full data if None)
            sim_end_date: Optional end date for simulation
            excluded_date_ranges: Optional list of (start, end) date tuples to exclude

        Returns:
            Dict with session summary, trades, and per-stock metrics.
        """
        if not symbols_data:
            raise ValueError("No symbols data provided for portfolio backtest")

        logger.info(f"🚀 Starting portfolio backtest for strategy: {self.strategy_config['name']}")
        logger.info(f"   Capital: ₹{self.initial_capital:,.0f} | Max Positions: {self.max_positions}")

        # 1. Build common timeline
        common_dates = self._get_common_dates(symbols_data)
        if len(common_dates) < 60:
            raise ValueError(f"Insufficient common trading days: {len(common_dates)}")

        # Filter to simulation date range if provided
        if sim_start_date is not None:
            common_dates = common_dates[common_dates >= sim_start_date]
        if sim_end_date is not None:
            common_dates = common_dates[common_dates <= sim_end_date]

        # Filter out excluded date ranges (e.g. 2020 COVID period)
        excluded = excluded_date_ranges or self.excluded_date_ranges
        if excluded:
            before = len(common_dates)
            for excl_start, excl_end in excluded:
                excl_start = pd.Timestamp(excl_start, tz=common_dates.tz)
                excl_end = pd.Timestamp(excl_end, tz=common_dates.tz)
                common_dates = common_dates[~((common_dates >= excl_start) & (common_dates <= excl_end))]
            if len(common_dates) < before:
                logger.info(f"   Excluded {before - len(common_dates)} dates from {len(excluded)} range(s)")

        if len(common_dates) < 60:
            raise ValueError(f"Insufficient common trading days after date filter: {len(common_dates)}")

        self.start_date = common_dates[0]
        self.end_date = common_dates[-1]

        logger.info(
            f"   Simulation range: {self.start_date.date()} → {self.end_date.date()} ({len(common_dates)} days)"
        )

        # 2. Pre-compute last available date per symbol (for delisted detection)
        self._last_dates = {sym: df.index[-1] for sym, df in symbols_data.items()}

        # Pre-compute close prices for O(1) lookup
        self._close_prices = {}
        for sym, df in symbols_data.items():
            self._close_prices[sym] = dict(zip(df.index, df["Close"]))

        # Pre-compute ATR for all symbols (eliminates _calculate_atr fallback path)
        self._atr_cache = {}
        for sym, df in symbols_data.items():
            try:
                atr_series = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
                self._atr_cache[sym] = dict(zip(df.index, atr_series))
            except Exception:
                pass

        # Pre-build date-to-index-position map (for iloc slicing)
        self._date_idx = {}
        for sym, df in symbols_data.items():
            self._date_idx[sym] = {dt: i for i, dt in enumerate(df.index)}

        # 3. Day-by-day simulation
        total_days = len(common_dates)
        import time as _time

        _t0 = _time.time()
        for i, date in enumerate(common_dates):
            self.bar_count = i
            self._simulate_day(date, symbols_data)
            # Print progress every 250 days
            if (i + 1) % 250 == 0 or i == total_days - 1:
                pct = (i + 1) / total_days * 100
                elapsed = _time.time() - _t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_days - i - 1) / speed if speed > 0 else 0
                print(
                    f"  Progress: {i+1}/{total_days} ({pct:.0f}%) | {self.cash:,.0f} cash | {len(self.positions)} pos | {elapsed:.0f}s elapsed, ~{eta:.0f}s left"
                )

        # 4. Force-close any remaining open positions at last price
        self._force_close_all_at_end(symbols_data)

        # 5. Calculate final metrics
        metrics = self._calculate_metrics(common_dates)

        logger.info(
            f"🏁 Portfolio backtest complete: Final Value ₹{metrics['final_portfolio_value']:,.0f} "
            f"| CAGR {metrics['cagr']:.1f}% | Max DD {metrics['max_drawdown_pct']:.1f}%"
        )

        return self._format_session_results(metrics)

    def run_with_signals(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        precomputed_signals: Dict[str, Dict],
        sim_start_date: Optional[pd.Timestamp] = None,
        sim_end_date: Optional[pd.Timestamp] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run portfolio backtest using pre-computed signals (from multiprocessing workers)."""
        if not symbols_data:
            raise ValueError("No symbols data provided for portfolio backtest")

        if verbose:
            logger.info(f"🚀 Starting portfolio backtest for strategy: {self.strategy_config['name']}")
            logger.info(f"   Capital: ₹{self.initial_capital:,.0f} | Max Positions: {self.max_positions}")
            logger.info(f"   Using pre-computed signals for {len(precomputed_signals)} symbols")

        # 1. Build common timeline
        common_dates = self._get_common_dates(symbols_data)
        if len(common_dates) < 100:
            raise ValueError(f"Insufficient common trading days: {len(common_dates)}")

        if sim_start_date is not None:
            common_dates = common_dates[common_dates >= sim_start_date]
        if sim_end_date is not None:
            common_dates = common_dates[common_dates <= sim_end_date]

        if len(common_dates) < 5:
            raise ValueError(f"Insufficient common trading days after date filter: {len(common_dates)}")

        self.start_date = common_dates[0]
        self.end_date = common_dates[-1]

        if verbose:
            logger.info(
                f"   Simulation range: {self.start_date.date()} → {self.end_date.date()} ({len(common_dates)} days)"
            )

        # 2. Pre-compute last available date per symbol
        self._last_dates = {sym: df.index[-1] for sym, df in symbols_data.items()}

        self._close_prices = {}
        for sym, df in symbols_data.items():
            self._close_prices[sym] = dict(zip(df.index, df["Close"]))

        self._signals_by_date = {}
        for symbol, date_signals in precomputed_signals.items():
            if symbol not in symbols_data:
                continue
            for dt, sig_data in date_signals.items():
                dt_key = dt.tz_localize(None) if hasattr(dt, "tzinfo") and dt.tzinfo is not None else dt
                if dt_key not in self._signals_by_date:
                    self._signals_by_date[dt_key] = {}
                self._signals_by_date[dt_key][symbol] = sig_data

        # Pre-compute ATR for all symbols (eliminates _calculate_atr fallback path)
        self._atr_cache = {}
        for sym, df in symbols_data.items():
            try:
                atr_series = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=14)
                self._atr_cache[sym] = dict(zip(df.index, atr_series))
            except Exception:
                pass

        # Pre-build date-to-index-position map (for iloc slicing)
        self._date_idx = {}
        for sym, df in symbols_data.items():
            self._date_idx[sym] = {dt: i for i, dt in enumerate(df.index)}

        # 3. Day-by-day simulation using pre-computed signals
        total_days = len(common_dates)
        import time as _time

        _t0 = _time.time()
        for i, date in enumerate(common_dates):
            self.bar_count = i
            self._simulate_day(date, symbols_data, use_precomputed_signals=True)
            if verbose and ((i + 1) % 250 == 0 or i == total_days - 1):
                pct = (i + 1) / total_days * 100
                elapsed = _time.time() - _t0
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total_days - i - 1) / speed if speed > 0 else 0
                print(
                    f"  Progress: {i+1}/{total_days} ({pct:.0f}%) | {self.cash:,.0f} cash | {len(self.positions)} pos | {elapsed:.0f}s elapsed, ~{eta:.0f}s left"
                )

        # 4. Force-close any remaining open positions
        self._force_close_all_at_end(symbols_data)

        # 5. Calculate final metrics
        metrics = self._calculate_metrics(common_dates)

        return self._format_session_results(metrics)

    def _format_session_results(self, metrics: dict) -> Dict[str, Any]:
        """Format final session result dictionary."""
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

    def run_with_indicator_store(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        indicator_store: Any,
        sim_start_date: Optional[pd.Timestamp] = None,
        sim_end_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Run portfolio backtest using pre-computed indicator store for fast signal lookups."""
        self.set_indicator_store(indicator_store)
        return self.run(symbols_data, sim_start_date, sim_end_date)

    # Drawdown Pause Circuit Breaker

    def _check_drawdown_pause(self, symbols_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> bool:
        """Check/update portfolio drawdown pause state. Returns True if entries should be paused."""
        cfg = self._dd_pause_cfg
        if not cfg.get("enabled", False):
            return False

        curr_value = self._current_portfolio_value(symbols_data, date)
        if curr_value > self.peak_value:
            self.peak_value = curr_value

        dd_pct = ((self.peak_value - curr_value) / self.peak_value) * 100
        pause_threshold = cfg.get("pause_threshold_pct", 15.0)
        resume_threshold = cfg.get("resume_threshold_pct", 10.0)
        pause_days_min = cfg.get("pause_days_min", 5)

        if not self._dd_pause_active:
            if dd_pct >= pause_threshold:
                self._dd_pause_active = True
                self._dd_pause_start_bar = self.bar_count
                logger.info(
                    f"⏸️ DRAWDOWN PAUSE activated on {date.date()}: DD {dd_pct:.1f}% >= {pause_threshold}% threshold"
                )
                return True
            return False
        else:
            bars_paused = self.bar_count - (self._dd_pause_start_bar or self.bar_count)
            cooldown_bars = cfg.get("cooldown_bars", cfg.get("pause_days_min", 20))
            if (dd_pct <= resume_threshold or bars_paused >= cooldown_bars) and bars_paused >= pause_days_min:
                self._dd_pause_active = False
                self._dd_pause_start_bar = None
                self.peak_value = curr_value
                logger.info(f"▶️ DRAWDOWN PAUSE lifted on {date.date()}: DD at {dd_pct:.1f}%, paused {bars_paused} bars")
                return False
            return True

    # --- Simulation Core ---

    def _simulate_day(
        self,
        date: pd.Timestamp,
        symbols_data: Dict[str, pd.DataFrame],
        use_precomputed_signals: bool = False,
    ):
        """Process a single trading day: exits → entries → pyramiding → snapshot."""
        # Check max daily loss circuit breaker
        exit_cfg = self.strategy_config.get("exit_rules", {})
        max_daily_loss_pct = exit_cfg.get("max_daily_loss_pct", None)
        if max_daily_loss_pct and self.daily_snapshots and self.positions:
            prev_value = self.daily_snapshots[-1]["portfolio_value"]
            curr_value = self._current_portfolio_value(symbols_data, date)
            daily_loss_pct = ((curr_value - prev_value) / prev_value) * 100
            if daily_loss_pct <= -max_daily_loss_pct:
                logger.info(
                    f"🛑 MAX DAILY LOSS TRIGGERED on {date.date()}: Loss {daily_loss_pct:.2f}% >= {max_daily_loss_pct}% threshold."
                )
                for symbol in list(self.positions.keys()):
                    df = symbols_data.get(symbol)
                    pos = self.positions[symbol]
                    cur_price = self._close_prices.get(symbol, {}).get(date)
                    if cur_price is None:
                        cur_price = df.loc[date, "Close"] if (df is not None and date in df.index) else pos.entry_price
                    self._close_position(symbol, date, cur_price, "MAX_DAILY_LOSS")
                return

        # --- Phase 1: Process Exits ---
        exits_today = self._process_exits(date, symbols_data)
        _ = sum(e.pnl for e in exits_today if e.pnl > 0)

        # --- Phase 1b: Daily Active Rebalancing (if enabled) ---
        rebal_cfg = self.strategy_config.get("exit_rules", {}).get("daily_active_rebalancing", {})
        if rebal_cfg.get("enabled", False) and self.positions:
            self._daily_active_rebalance(date, symbols_data, rebal_cfg)

        # --- Phase 2: Scan or fetch pre-computed signals ---
        entries_paused = self._check_drawdown_pause(symbols_data, date)

        # Check pause_buying_if_bearish from strategy config
        if not entries_paused and self.regime_enabled:
            regime_status = self._check_market_regime(date, symbols_data)
            pause_buying = self.regime_config.get("pause_buying_if_bearish", True)
            if regime_status == "BEAR" and pause_buying:
                logger.info(f"[{self.strategy_config['name']}] Market regime BEAR: pausing new buys")
                entries_paused = True

        candidates = []
        if not entries_paused:
            if use_precomputed_signals:
                if self._check_market_breadth(date, symbols_data):
                    date_key = date.tz_localize(None) if date.tzinfo is not None else date
                    for symbol, sig_data in self._signals_by_date.get(date_key, {}).items():
                        if symbol not in self.positions:
                            candidates.append(
                                {"symbol": symbol, "score": sig_data["score"], "swing_result": sig_data["swing_result"]}
                            )
            else:
                candidates = self._scan_for_signals(date, symbols_data)

        # --- Phase 3: Rank and Execute Buys ---
        if candidates:
            candidates.sort(key=lambda c: c["score"], reverse=True)
            for cand in candidates:
                if cand["symbol"] not in symbols_data:
                    continue
                if not self._can_open_new_position(cand["symbol"]):
                    continue
                self._execute_buy(cand, date, symbols_data)

        # --- Phase 4: Pyramiding ---
        self._process_pyramiding(date, symbols_data)

        # --- Phase 4b: Dynamic Cash Deployment (Liquid ETF Yield on unallocated cash) ---
        cash_cfg = self.strategy_config.get("cash_deployment", {})
        if cash_cfg.get("enabled", True) and self.cash > 0:
            idle_thresh = cash_cfg.get("idle_threshold_positions", 2)
            regime = self._regime_status.lower() if self._regime_status != "UNKNOWN" else "bull"
            if len(self.positions) <= idle_thresh or regime == "bear":
                annual_yield = cash_cfg.get("annual_liquid_yield_pct", 6.0) / 100.0
                self.cash += self.cash * (annual_yield / 252.0)

        # --- Phase 5: Record Snapshot ---
        if self.save_snapshots:
            self._record_snapshot(date, symbols_data)

    def _process_exits(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> List[PortfolioTrade]:
        """Check all open positions for SL, target, time-stop, O'Neil rules, or delisted."""
        from scripts.backtest_position_manager import check_position_exit_signal

        exits = []
        symbols_to_remove = []
        exit_cfg = self.strategy_config.get("exit_rules", {})
        regime_adaptive = exit_cfg.get("regime_adaptive_exits", {})
        regime_enabled = regime_adaptive.get("enabled", True) and self.regime_enabled
        regime = self._regime_status.lower() if self._regime_status != "UNKNOWN" else "bull"
        regime_params = regime_adaptive.get(regime, {}) if regime_enabled else {}

        for symbol, pos in list(self.positions.items()):
            df = symbols_data.get(symbol)
            if df is None or date not in df.index:
                if self.force_close_delisted:
                    hist_up_to_date = df.loc[:date] if df is not None else None
                    last_price = (
                        hist_up_to_date["Close"].iloc[-1]
                        if (hist_up_to_date is not None and not hist_up_to_date.empty)
                        else pos.entry_price
                    )
                    trade = self._close_position(symbol, date, last_price, "DELISTED")
                    if trade:
                        exits.append(trade)
                continue

            current_price = self._close_prices.get(symbol, {}).get(date, df.loc[date, "Close"])
            atr = self._calculate_atr_from_store(symbol, date) or self._calculate_atr(df, date)
            bars_held = self.bar_count - pos.bar_executed
            days_held = (date - pos.entry_date).days if hasattr(date, "__sub__") else bars_held

            date_idx_dict = self._date_idx.get(symbol, {})
            exit_info = check_position_exit_signal(
                pos,
                date,
                current_price,
                atr,
                bars_held,
                days_held,
                regime_params,
                exit_cfg,
                df,
                date_idx_dict,
            )

            if exit_info:
                reason, exit_price, sell_qty_override = exit_info
                if sell_qty_override and sell_qty_override > 0:
                    trade = self._partial_sell(symbol, date, exit_price, sell_qty_override, pos, reason)
                    if trade:
                        exits.append(trade)
                    pos.quantity -= sell_qty_override
                    if pos.current_target_idx == 0 and exit_cfg.get("breakeven_at_target_1", False):
                        pos.current_stop_loss = max(pos.current_stop_loss, pos.entry_price)
                        logger.info(f"🛡️ BREAKEVEN STOP: Moved stop loss on {symbol} to entry ₹{pos.entry_price:.2f}")
                    pos.current_target_idx += 1
                    if pos.quantity <= 0:
                        symbols_to_remove.append(symbol)
                else:
                    trade = self._close_position(symbol, date, exit_price, reason)
                    if trade:
                        exits.append(trade)

        for sym in symbols_to_remove:
            if sym in self.positions:
                del self.positions[sym]

        return exits

    def _daily_active_rebalance(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame], rebal_cfg: dict):
        """Daily active rebalancing: sell top gainers, buy bottom losers."""
        from scripts.backtest_position_manager import daily_active_rebalance

        daily_active_rebalance(
            self.positions,
            self._close_prices,
            symbols_data,
            date,
            rebal_cfg,
            self._partial_sell,
            self.brokerage,
            logger,
        )

    def _scan_for_signals(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> List[dict]:
        """Scan stocks without open positions for BUY signals."""
        candidates = []

        # Check regime for adaptive behavior (but don't block entries)
        self._check_market_regime(date, symbols_data)

        # Market breadth filter: advance/decline ratio across the universe
        market_breadth_ok = self._check_market_breadth(date, symbols_data)

        for symbol, df in symbols_data.items():
            if symbol in self.positions:
                continue
            if date not in df.index:
                continue

            if self._stock_prefilter is not None:
                pf = self._stock_prefilter
                if date not in pf.index or symbol not in pf.columns:
                    continue
                if not pf.loc[date, symbol]:
                    continue

            # Truncate data up to current date (no look-ahead)
            idx = self._date_idx.get(symbol, {}).get(date)
            if idx is None:
                idx = df.index.searchsorted(date)
            hist = df.iloc[: idx + 1]
            if len(hist) < 50:
                continue

            try:
                swing = self.swing_analyzer.analyze_swing_opportunity(
                    symbol,
                    hist,
                    strategy_config=self.strategy_config,
                    indicator_store=self._indicator_store,
                    market_breadth_ok=market_breadth_ok,
                )

                if swing.get("all_gates_passed") and swing.get("recommendation") == "BUY":
                    c_now = hist["Close"].iloc[-1]
                    c_past = hist["Close"].iloc[-63] if len(hist) >= 63 else hist["Close"].iloc[0]
                    mom_3m = (c_now / c_past) if c_past > 0 else 1.0
                    h52 = hist["High"].tail(252).max() if len(hist) >= 20 else c_now
                    prox_score = (c_now / h52) if h52 > 0 else 1.0
                    tech_score = swing.get("technical_score", 0.0)
                    score = (mom_3m * 2.0) + prox_score + tech_score

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
        if symbol not in symbols_data:
            return
        df = symbols_data[symbol]
        swing_result = candidate.get("swing_result", {})
        reason = swing_result.get("reason", "")
        patterns = re.findall(r"Pattern\((\w+)\)", reason)
        entry_pattern = patterns[0] if patterns else "unknown"
        close_price = self._close_prices.get(symbol, {}).get(date, df.loc[date, "Close"])

        # Apply slippage: pay slightly more than close on buys
        exec_price = close_price * (1 + self.slippage)

        # Size positions using current portfolio value (true geometric compounding)
        portfolio_value = self._current_portfolio_value(symbols_data, date)
        self.risk_manager.balance = max(self.initial_capital * 0.5, portfolio_value)

        idx = self._date_idx.get(symbol, {}).get(date)
        if idx is None:
            idx = df.index.searchsorted(date)
        hist = df.iloc[: idx + 1]
        risk = self.risk_manager.calculate_risk_params(hist, exec_price, self.strategy_config, self._regime_status)

        if not risk.get("risk_reward_ok"):
            return

        size = risk["position_size"]
        if size <= 0:
            return

        # Respect max_position_pct
        max_by_pct = int((portfolio_value * self.max_position_pct) / exec_price)
        size = min(size, max_by_pct)

        # Calculate realistic buy cost
        from scripts.backtest_position_manager import calculate_buy_order_cost

        cost, total_cost = calculate_buy_order_cost(size, exec_price, self.use_realistic_costs, self.brokerage)

        # Respect available cash
        if total_cost > self.cash:
            max_by_cash = int(self.cash / exec_price)
            size = min(size, max_by_cash)
            if size > 0:
                cost, total_cost = calculate_buy_order_cost(size, exec_price, self.use_realistic_costs, self.brokerage)

        if size <= 0 or total_cost > self.cash:
            return

        # Deduct cash
        self.cash -= total_cost
        self._cached_pv = None

        # Create position
        entry_atr = self._calculate_atr_from_store(symbol, date) or self._calculate_atr(df, date)
        position = PortfolioPosition(
            symbol=symbol,
            entry_date=date,
            entry_price=exec_price,
            quantity=size,
            stop_loss=risk["stop_loss"],
            current_stop_loss=risk["stop_loss"],
            bar_executed=self.bar_count,
            entry_pattern=entry_pattern,
            entry_atr=entry_atr if entry_atr else 0.0,
        )
        self.positions[symbol] = position

        trade = PortfolioTrade(
            symbol=symbol,
            trade_type="BUY",
            entry_date=str(date.date()),
            entry_price=exec_price,
            quantity=size,
            position_value=cost,
            allocation_pct=(cost / self.initial_capital) * 100,
            stop_loss=risk["stop_loss"],
            target=risk["targets"].get("T1") if risk.get("targets") else None,
            portfolio_value_at_entry=portfolio_value,
            cash_balance_at_entry=self.cash,
            open_positions_count_at_entry=len(self.positions) - 1,
            entry_pattern=entry_pattern,
        )
        self.trades.append(trade)

        logger.info(
            f"🟢 BUY {symbol} @ ₹{exec_price:.2f} | Qty: {size} | "
            f"SL: ₹{risk['stop_loss']:.2f} | Cash left: ₹{self.cash:,.0f}"
        )

    def _process_pyramiding(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]):
        """Check existing positions for pyramid add triggers."""
        from scripts.backtest_position_manager import evaluate_pyramid_step

        pyramid_cfg = self.strategy_config.get("pyramiding", {})
        if not pyramid_cfg.get("enabled", False):
            return

        if pyramid_cfg.get("regime_controlled", False) and self.regime_enabled:
            regime = self._regime_status.lower() if self._regime_status != "UNKNOWN" else "bull"
            regime_adaptive = self.strategy_config.get("exit_rules", {}).get("regime_adaptive_exits", {})
            if regime_adaptive.get("enabled", True):
                regime_params = regime_adaptive.get(regime, {})
                if not regime_params.get("pyramiding_allowed", True):
                    return

        for symbol, pos in list(self.positions.items()):
            df = symbols_data.get(symbol)
            if df is None or date not in df.index:
                continue

            current_price = self._close_prices.get(symbol, {}).get(date, df.loc[date, "Close"])
            atr = self._calculate_atr_from_store(symbol, date) or self._calculate_atr(df.loc[:date], date)
            portfolio_value = self._current_portfolio_value(symbols_data, date)

            pyramid_res = evaluate_pyramid_step(
                pos,
                date,
                current_price,
                atr,
                pyramid_cfg,
                self.pyramid_counts_as_new,
                self.max_positions,
                len(self.positions),
                self.cash,
                self.brokerage,
                portfolio_value,
                self.max_position_pct,
            )

            if not pyramid_res:
                continue

            add_qty, cost, total_cost = pyramid_res
            self.cash -= total_cost
            self._cached_pv = None
            pos.quantity += add_qty
            pos.adds_count += 1
            pos.last_add_price = current_price
            pos.current_stop_loss = max(pos.current_stop_loss, pos.entry_price)
            logger.info(f"🛡️ PYRAMID STOP: Moved stop loss on {symbol} to entry ₹{pos.entry_price:.2f}")

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

            logger.info(f"⬆️ PYRAMID {symbol} | Added {add_qty} @ ₹{current_price:.2f} | " f"New Qty: {pos.quantity}")

    def _close_position(self, symbol: str, date: pd.Timestamp, price: float, reason: str) -> Optional[PortfolioTrade]:
        """Close a position fully and record the trade."""
        from scripts.backtest_position_manager import execute_close_position

        pos = self.positions.get(symbol)
        if not pos:
            return None

        atr_val = self._calculate_atr_from_store(symbol, date) or (price * 0.02)
        fill_price, gross_value, net_value, pnl = execute_close_position(
            pos,
            date,
            price,
            reason,
            self.use_realistic_costs,
            self.gap_seed,
            self.bar_count,
            self.brokerage,
            atr_val,
        )

        pnl_pct = (pnl / (pos.quantity * pos.entry_price)) * 100 if pos.entry_price > 0 else 0
        self.cash += net_value
        self._cached_pv = None

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
            entry_pattern=pos.entry_pattern,
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
        from scripts.backtest_position_manager import execute_close_position

        atr_val = self._calculate_atr_from_store(symbol, date) or (price * 0.02)
        # Temporary position for partial math
        temp_pos = PortfolioPosition(
            symbol=symbol,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            quantity=qty,
            stop_loss=pos.stop_loss,
        )
        fill_price, gross_value, net_value, pnl = execute_close_position(
            temp_pos,
            date,
            price,
            reason,
            self.use_realistic_costs,
            self.gap_seed,
            self.bar_count,
            self.brokerage,
            atr_val,
        )

        cost_basis = qty * pos.entry_price
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
            entry_pattern=pos.entry_pattern,
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

    # Helpers

    def _check_market_breadth(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> bool:
        """Calculate market breadth across the stock universe."""
        return check_market_breadth(
            self.strategy_config,
            self._indicator_store,
            date,
            symbols_data,
            self._date_idx,
        )

    def _check_market_regime(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> str:
        """Check market regime (BULL/BEAR) using NIFTY 50 index."""
        if date == self._regime_check_date:
            return self._regime_status

        self._regime_status = check_market_regime(
            self.regime_enabled,
            self.regime_config,
            date,
            self._index_data_override,
            symbols_data,
            self._date_idx,
            logger,
        )
        self._regime_check_date = date
        return self._regime_status

    def _get_common_dates(self, symbols_data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Build a union of all trading dates across symbols."""
        all_dates = set().union(*(df.index for df in symbols_data.values())) if symbols_data else set()
        return pd.DatetimeIndex(sorted(all_dates))

    def _current_portfolio_value(self, symbols_data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        if self._cached_pv is not None and self._cached_pv_date == date:
            return self._cached_pv
        market_value = sum(
            pos.quantity * self._close_prices.get(sym, {}).get(date, pos.entry_price)
            for sym, pos in self.positions.items()
        )
        self._cached_pv = self.cash + market_value
        self._cached_pv_date = date
        return self._cached_pv

    def _current_portfolio_value_at_date(self, date: pd.Timestamp) -> float:
        return self.cash + sum(pos.quantity * pos.entry_price for pos in self.positions.values())

    def _calculate_atr(self, df: pd.DataFrame, date: pd.Timestamp, period: int = 14) -> float:
        s = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=period)
        return float(s.loc[date]) if date in s.index else float(s.iloc[-1])

    def _calculate_atr_from_store(self, symbol: str, date: pd.Timestamp, period: int = 14) -> Optional[float]:
        if self._indicator_store is not None:
            v = self._indicator_store.get(symbol, "atr_14", date)
            if v is not None and not np.isnan(v):
                return float(v)
        v = self._atr_cache.get(symbol, {}).get(date)
        return float(v) if v is not None and not np.isnan(v) else None

    def _can_open_new_position(self, symbol: str) -> bool:
        if symbol in self.positions:
            return False
        regime = self._regime_status.lower() if self._regime_status != "UNKNOWN" else "bull"
        regime_risk_cfg = self.strategy_config.get("risk_management", {}).get("regime_adaptive_risk", {})
        max_pos = regime_risk_cfg.get(regime, {}).get("max_positions", self.max_positions)
        return len(self.positions) < max_pos

    def _record_snapshot(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]):
        pv = self._current_portfolio_value(symbols_data, date)
        self.peak_value = record_daily_snapshot(
            self.daily_snapshots, date, pv, self.cash, self.peak_value, self.positions
        )

    def _calculate_metrics(self, common_dates: pd.DatetimeIndex) -> Dict[str, Any]:
        return calculate_portfolio_metrics(
            self.daily_snapshots, self.trades, self.initial_capital, self.cash, common_dates
        )
