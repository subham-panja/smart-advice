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
        self.risk_per_trade = cfg["risk_per_trade"]
        self.max_position_pct = cfg["max_position_pct"]
        self.max_positions = cfg["max_concurrent_positions"]
        self.ranking_method = cfg.get("ranking_method", "combined_score")
        self.save_snapshots = cfg.get("save_daily_snapshots", True)
        self.pyramid_counts_as_new = cfg.get("pyramid_counts_as_new_position", False)
        self.same_day_recycling = cfg.get("same_day_cash_recycling", True)
        self.force_close_delisted = cfg.get("force_close_delisted", True)

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

        # Results
        self.session_id: Any = None
        self.start_date: Optional[pd.Timestamp] = None
        self.end_date: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run the portfolio backtest across all provided symbols.

        Args:
            symbols_data: Dict mapping symbol -> DataFrame (5yr OHLCV)

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

    # ------------------------------------------------------------------
    # Simulation Core
    # ------------------------------------------------------------------

    def _simulate_day(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]):
        """Process a single trading day: exits → entries → pyramiding → snapshot."""
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

    def _process_exits(self, date: pd.Timestamp, symbols_data: Dict[str, pd.DataFrame]) -> List[PortfolioTrade]:
        """Check all open positions for SL, target, time-stop, or delisted."""
        exits = []
        symbols_to_remove = []

        for symbol, pos in list(self.positions.items()):
            df = symbols_data.get(symbol)
            if df is None or date not in df.index:
                # Delisted or no data — force close
                if self.force_close_delisted:
                    last_price = df["Close"].iloc[-1] if df is not None else pos.entry_price
                    trade = self._close_position(symbol, date, last_price, "DELISTED")
                    if trade:
                        exits.append(trade)
                continue

            current_price = df.loc[date, "Close"]
            exit_cfg = self.strategy_config.get("exit_rules", {})
            time_stop_bars = exit_cfg.get("time_stop_bars", 45)

            # 1. Time Stop
            bars_held = self.bar_count - pos.bar_executed
            if bars_held >= time_stop_bars:
                trade = self._close_position(symbol, date, current_price, "TIME_STOP")
                if trade:
                    exits.append(trade)
                continue

            # 2. Stop Loss
            if current_price <= pos.current_stop_loss:
                trade = self._close_position(symbol, date, current_price, "STOP_LOSS")
                if trade:
                    exits.append(trade)
                continue

            # 3. Targets
            targets = exit_cfg.get("targets", [])
            if pos.current_target_idx < len(targets):
                target_cfg = targets[pos.current_target_idx]
                # Calculate ATR for target
                atr = self._calculate_atr(df, date)
                target_price = pos.entry_price + (target_cfg["atr_multiplier"] * atr)

                if current_price >= target_price:
                    sell_pct = target_cfg["sell_percentage"]
                    sell_qty = int(pos.quantity * sell_pct)

                    if sell_qty > 0 and sell_pct < 1.0:
                        # Partial sell
                        trade = self._partial_sell(symbol, date, current_price, sell_qty, pos, target_cfg["name"])
                        if trade:
                            exits.append(trade)

                        # Update position
                        pos.quantity -= sell_qty
                        pos.current_target_idx += 1
                        pos.is_scaled_out = True

                        # Breakeven logic
                        if pos.current_target_idx == 1 and exit_cfg.get("breakeven_at_target_1"):
                            pos.current_stop_loss = pos.entry_price
                            logger.info(f"🛡️ {symbol}: SL moved to breakeven ₹{pos.entry_price:.2f}")

                        if pos.quantity <= 0:
                            symbols_to_remove.append(symbol)
                        continue
                    elif sell_pct >= 1.0:
                        # Final target — full exit
                        trade = self._close_position(symbol, date, current_price, f"FINAL_{target_cfg['name']}")
                        if trade:
                            exits.append(trade)
                        continue

            # 4. Trailing Stop Loss Update
            if atr > 0:
                trail_mult = exit_cfg.get("trail_stop_atr", 2.5)
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

        # Calculate risk params using current portfolio value
        portfolio_value = self._current_portfolio_value(symbols_data, date)
        self.risk_manager.balance = portfolio_value

        hist = df.loc[:date]
        risk = self.risk_manager.calculate_risk_params(hist, current_price, self.strategy_config)

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
        """Check existing positions for pyramid add triggers."""
        pyramid_cfg = self.strategy_config.get("pyramiding", {})
        if not pyramid_cfg.get("enabled", False):
            return

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

            # Calculate pyramid quantity
            base_qty = (
                pos.quantity
                if pos.adds_count == 0
                else pos.initial_quantity
                if hasattr(pos, "initial_quantity")
                else pos.quantity
            )
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
        """Check if we can open a new position."""
        if symbol in self.positions:
            return False

        current_slot_count = len(self.positions)
        return current_slot_count < self.max_positions

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
