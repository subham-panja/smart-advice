import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import pandas as pd


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
    entry_pattern: str = "unknown"
    entry_atr: float = 0.0

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
    entry_pattern: Optional[str] = None


def can_open_new_position(positions: dict, strategy_config: dict, symbol: str) -> bool:
    """Check if new position slot is available."""
    risk_cfg = strategy_config.get("risk_management", {})
    max_pos = risk_cfg.get("max_positions", 15)
    return len(positions) < max_pos


def calculate_position_size(
    cash: float,
    portfolio_value: float,
    price: float,
    stop_loss: float,
    strategy_config: dict,
) -> int:
    """Calculate share quantity based on risk-per-trade and position allocation caps."""
    if price <= 0 or price <= stop_loss:
        return 0

    risk_cfg = strategy_config.get("risk_management", {})
    risk_per_trade_pct = risk_cfg.get("risk_per_trade_pct", 2.0) / 100.0
    max_position_pct = risk_cfg.get("max_position_pct", 10.0) / 100.0

    max_capital_for_pos = portfolio_value * max_position_pct
    max_capital_affordable = min(cash, max_capital_for_pos)

    risk_amount = portfolio_value * risk_per_trade_pct
    per_share_risk = price - stop_loss

    if per_share_risk <= 0:
        return 0

    qty_by_risk = math.floor(risk_amount / per_share_risk)
    qty_by_cap = math.floor(max_capital_affordable / price)

    return max(0, min(qty_by_risk, qty_by_cap))


def daily_active_rebalance(
    positions: dict,
    close_prices: dict,
    symbols_data: dict,
    date: pd.Timestamp,
    rebal_cfg: dict,
    partial_sell_fn: Any,
    brokerage_pct: float,
    logger_obj: Any,
):
    """Daily active rebalancing: sell top gainers, buy bottom losers."""
    top_gainer_pct = rebal_cfg.get("top_gainer_threshold_pct", 3.0)
    sell_pct = rebal_cfg.get("sell_amount_of_remaining_position_pct", 10.0) / 100.0
    bottom_loser_pct = rebal_cfg.get("bottom_loser_threshold_pct", -3.0)
    buy_pct = rebal_cfg.get("buy_amount_with_freed_capital_pct", 10.0) / 100.0

    position_pnl = []
    for symbol, pos in positions.items():
        df = symbols_data.get(symbol)
        if df is None or date not in df.index:
            continue
        current_price = close_prices.get(symbol, {}).get(date, df.loc[date, "Close"])
        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
        position_pnl.append((symbol, pos, current_price, pnl_pct))

    position_pnl.sort(key=lambda x: x[3], reverse=True)
    freed_capital = 0.0

    for symbol, pos, current_price, pnl_pct in position_pnl:
        if pnl_pct >= top_gainer_pct:
            sell_qty = max(1, int(pos.quantity * sell_pct))
            if sell_qty < pos.quantity:
                trade = partial_sell_fn(symbol, date, current_price, sell_qty, pos, "REBALANCE_SELL")
                if trade:
                    freed_capital += trade.exit_price * sell_qty
                    logger_obj.info(
                        f"🔄 REBALANCE: Sold {sell_qty} of {symbol} (+{pnl_pct:.1f}%) | Freed ₹{freed_capital:.0f}"
                    )
                pos.quantity -= sell_qty

    position_pnl.sort(key=lambda x: x[3])
    for symbol, pos, current_price, pnl_pct in position_pnl:
        if pnl_pct <= bottom_loser_pct and freed_capital > 0:
            buy_amount = freed_capital * buy_pct
            if buy_amount > 0 and current_price > 0:
                buy_qty = int(buy_amount / current_price)
                if buy_qty > 0:
                    cost = buy_qty * current_price
                    total_cost = cost + (cost * brokerage_pct)
                    if total_cost <= freed_capital:
                        pos.quantity += buy_qty
                        total_investment = pos.entry_price * (pos.quantity - buy_qty) + cost
                        pos.entry_price = total_investment / pos.quantity
                        freed_capital -= total_cost
                        logger_obj.info(
                            f"🔄 REBALANCE: Added {buy_qty} to {symbol} ({pnl_pct:.1f}%) | Avg entry: ₹{pos.entry_price:.2f}"
                        )


def check_position_exit_signal(
    pos: Any,
    date: pd.Timestamp,
    current_price: float,
    atr: float,
    bars_held: int,
    days_held: int,
    regime_params: dict,
    exit_cfg: dict,
    df: pd.DataFrame,
    date_idx_dict: dict,
) -> Optional[Tuple[str, float, Optional[int]]]:
    """Evaluates exit conditions for a single position.

    Returns None if no exit, or (reason, exit_price, sell_qty_override).
    """
    t1_pct = regime_params.get("t1_sell_percentage", exit_cfg.get("targets", [{}])[0].get("sell_percentage", 1.0))
    time_stop = regime_params.get("time_stop_bars", exit_cfg.get("time_stop_bars", 20))
    stop_loss_pct = regime_params.get("stop_loss_pct", None)

    # 0. O'Neil Fixed Stop Loss
    if stop_loss_pct:
        oneil_stop = pos.entry_price * (1 - stop_loss_pct / 100.0)
        if current_price <= oneil_stop:
            return f"ONEIL_STOP_{stop_loss_pct}%", current_price, None

    # 1. O'Neil Absolute Profit Target
    oneil_target_pct = regime_params.get("oneil_target_pct", 25.0)
    gain_pct = (current_price - pos.entry_price) / pos.entry_price * 100
    leader_cfg = exit_cfg.get("leader_exception", {})
    weeks_held = days_held / 7.0

    is_leader = False
    if leader_cfg.get("enabled", False) and gain_pct >= leader_cfg.get("min_gain_pct", 20.0):
        if weeks_held <= leader_cfg.get("max_weeks", 8):
            is_leader = True

    if gain_pct >= oneil_target_pct and not is_leader:
        return f"ONEIL_TARGET_{oneil_target_pct:.0f}%", current_price, None

    if not (is_leader and leader_cfg.get("action") == "hold_and_trail"):
        # 2. Time Stop - only exit if trade is stagnant/losing (fails to launch, never hit T1)
        if bars_held >= time_stop and getattr(pos, "current_target_idx", 0) == 0 and gain_pct < 2.0:
            return "TIME_STOP", current_price, None

        # 3. Stop Loss
        stop_loss_type = exit_cfg.get("stop_loss_type", "ATR")
        if stop_loss_type == "swing_low":
            lookback = max(10, min(20, bars_held))
            idx = date_idx_dict.get(date)
            if idx is not None:
                start = max(0, idx - lookback + 1)
                recent_lows = df["Low"].iloc[start : idx + 1]
                swing_low = recent_lows.min()
                swing_stop = swing_low * 0.98
                if current_price <= swing_stop:
                    return f"SWING_LOW_STOP@{swing_low:.2f}", current_price, None
                if swing_stop > pos.current_stop_loss:
                    pos.current_stop_loss = swing_stop
        elif current_price <= pos.current_stop_loss:
            return "STOP_LOSS", current_price, None

        # 4. Targets
        targets = exit_cfg.get("targets", [])
        if targets and targets[0].get("type") == "swing_structure" and pos.current_target_idx == 0:
            lookback = max(10, min(20, bars_held + 10))
            idx = date_idx_dict.get(date)
            if idx is not None:
                start = max(0, idx - lookback + 1)
                recent_highs = df["High"].iloc[start : idx + 1]
                swing_high = recent_highs.max()
                if current_price >= swing_high:
                    sell_pct = targets[0].get("sell_percentage", 1.0)
                    sell_qty = int(pos.quantity * sell_pct)
                    if sell_pct < 1.0:
                        return "SWING_HIGH", current_price, sell_qty
                    return "SWING_HIGH_TARGET", current_price, None
        elif pos.current_target_idx < len(targets):
            target_cfg = targets[pos.current_target_idx].copy()
            if pos.current_target_idx == 0:
                target_cfg["sell_percentage"] = t1_pct

            pos_atr = pos.entry_atr if getattr(pos, "entry_atr", 0.0) > 0 else atr
            target_price = pos.entry_price + (target_cfg["atr_multiplier"] * pos_atr)
            if current_price >= target_price:
                sell_pct = target_cfg["sell_percentage"]
                sell_qty = int(pos.quantity * sell_pct)
                if sell_pct < 1.0:
                    return target_cfg["name"], current_price, sell_qty
                return f"FINAL_{target_cfg['name']}", current_price, None

    # Trailing Stop update (Gated: only trail after Target 1 is hit or min gain reached)
    trail_only_after_t1 = exit_cfg.get("trail_only_after_t1", True)
    trail_min_gain_pct = exit_cfg.get("trail_min_gain_pct", 5.0)
    targets_hit = getattr(pos, "targets_hit", getattr(pos, "current_target_idx", 0))

    can_trail = True
    if trail_only_after_t1 and targets_hit == 0 and gain_pct < trail_min_gain_pct:
        can_trail = False

    if can_trail:
        trail_type = regime_params.get("trail_stop_type", "atr")
        if trail_type == "ma" and len(df) >= regime_params.get("trail_stop_ma_period", 20):
            ma_period = regime_params.get("trail_stop_ma_period", 20)
            ma_value = df["Close"].rolling(ma_period).mean().iloc[-1]
            if ma_value > pos.current_stop_loss:
                pos.current_stop_loss = ma_value
        elif atr > 0:
            trail_mult = regime_params.get("trail_stop_atr_multiplier", exit_cfg.get("trail_stop_atr", 2.8))
            new_sl = current_price - (atr * trail_mult)
            if new_sl > pos.current_stop_loss:
                pos.current_stop_loss = new_sl

    return None


def execute_close_position(
    pos: Any,
    date: pd.Timestamp,
    price: float,
    reason: str,
    use_realistic_costs: bool,
    gap_seed: int,
    bar_count: int,
    brokerage_pct: float,
    atr_val: float,
) -> Tuple[float, float, float, float]:
    """Calculates exit fill price, gross value, net proceeds, and PnL for a closed position.

    Returns (fill_price, gross_value, net_proceeds, pnl).
    """
    if use_realistic_costs:
        from scripts.execution_costs import apply_gap_risk, calculate_sell_cost

        fill_price = apply_gap_risk(price, atr_val or (price * 0.02), reason, seed=gap_seed + bar_count)
        gross_value = pos.quantity * fill_price
        costs = calculate_sell_cost(fill_price, pos.quantity)
        net_value = costs.net_proceeds
    else:
        fill_price = price
        gross_value = pos.quantity * fill_price
        net_value = gross_value - (gross_value * brokerage_pct)

    cost_basis = pos.quantity * pos.entry_price
    pnl = net_value - cost_basis
    return fill_price, gross_value, net_value, pnl


def calculate_buy_order_cost(
    size: int,
    exec_price: float,
    use_realistic_costs: bool,
    brokerage_pct: float,
) -> Tuple[float, float]:
    """Calculates position cost and net total cost (including fees).

    Returns (cost, total_cost).
    """
    cost = size * exec_price
    if use_realistic_costs:
        from scripts.execution_costs import calculate_buy_cost

        costs = calculate_buy_cost(exec_price, size)
        total_cost = costs.net_cost
    else:
        total_cost = cost + (cost * brokerage_pct)
    return cost, total_cost


def evaluate_pyramid_step(
    pos: Any,
    date: pd.Timestamp,
    current_price: float,
    atr: float,
    pyramid_cfg: dict,
    pyramid_counts_as_new: bool,
    max_positions: int,
    current_positions_count: int,
    cash: float,
    brokerage_pct: float,
    portfolio_value: float,
    max_position_pct: float,
) -> Optional[Tuple[int, float, float]]:
    """Evaluates if a position qualifies for a pyramid add step.

    Returns None or (add_qty, cost, total_cost).
    """
    steps = pyramid_cfg.get("steps", [])
    if pos.adds_count >= len(steps):
        return None

    step = steps[pos.adds_count]
    trigger_mult = step.get("trigger_step_atr", 1.5)
    required_price = pos.last_add_price + (trigger_mult * atr)

    if current_price < required_price:
        return None

    base_qty = pos.initial_quantity
    add_pct = step.get("add_size_pct", 0.5)
    add_qty = max(int(base_qty * add_pct), 1)

    if pyramid_counts_as_new and current_positions_count >= max_positions:
        return None

    cost = add_qty * current_price
    total_cost = cost + (cost * brokerage_pct)
    if total_cost > cash:
        return None

    current_position_value = pos.quantity * current_price
    new_position_value = current_position_value + cost
    max_pyramid_cap = max_position_pct * pyramid_cfg.get("max_position_multiplier", 1.6)
    if (new_position_value / portfolio_value) > max_pyramid_cap:
        return None

    return add_qty, cost, total_cost
