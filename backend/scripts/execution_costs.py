"""
Execution Cost Model for Indian Equity Delivery Trades
======================================================

Realistic transaction cost calculator including:
- Brokerage (discount broker rates)
- STT (Securities Transaction Tax)
- Exchange transaction charges
- SEBI charges
- Stamp duty
- GST on charges
- Slippage (buy and sell)
- Gap risk modeling for stop-loss fills

All rates as of 2024-2025 for NSE equity delivery trades.
"""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class CostBreakdown:
    """Detailed breakdown of all transaction costs."""

    trade_value: float
    brokerage: float
    stt: float
    exchange_charge: float
    sebi_charge: float
    stamp_duty: float
    gst: float
    slippage: float
    total_charges: float
    net_proceeds: float  # For sells: what you actually receive
    net_cost: float  # For buys: what you actually pay


# Indian market cost rates (2024-2025)
BROKERAGE_RATE = 0.0003  # 0.03% (discount broker like Zerodha/Groww)
STT_RATE = 0.001  # 0.1% on sell side only (equity delivery)
EXCHANGE_CHARGE_NSE = 0.0000325  # 0.00325%
SEBI_CHARGE = 0.000001  # 0.0001% (Rs 10 per crore)
STAMP_DUY_RATE = 0.00015  # 0.015% on buy side only (Maharashtra)
GST_RATE = 0.18  # 18% on all charges (brokerage + exchange + SEBI)

# Slippage rates
BUY_SLIPPAGE = 0.0015  # 0.15% - you pay slightly more
SELL_SLIPPAGE = 0.0015  # 0.15% - you receive slightly less

# Gap risk parameters
GAP_DOWN_PROBABILITY = 0.30  # 30% chance of gap beyond SL trigger
GAP_MIN_SEVERITY_ATR = 0.5  # Minimum gap = 0.5x ATR beyond trigger
GAP_MAX_SEVERITY_ATR = 2.0  # Maximum gap = 2.0x ATR beyond trigger
CIRCUIT_BREAKER_LIMIT = 0.20  # Max gap = 20% (circuit filter)

# Gap-up parameters (for target fills)
GAP_UP_PROBABILITY = 0.15  # 15% chance of gap beyond target
GAP_UP_MIN_SEVERITY_ATR = 0.1
GAP_UP_MAX_SEVERITY_ATR = 0.5


def calculate_buy_cost(
    price: float,
    quantity: int,
    slippage_pct: float = BUY_SLIPPAGE,
) -> CostBreakdown:
    """Calculate all costs for a BUY order.

    Args:
        price: Signal price (close price of signal day)
        quantity: Number of shares
        slippage_pct: Buy slippage rate (default 0.15%)

    Returns:
        CostBreakdown with all charges
    """
    exec_price = price * (1 + slippage_pct)
    trade_value = exec_price * quantity

    # Brokerage
    brokerage = trade_value * BROKERAGE_RATE

    # STT: not applicable on buy (equity delivery)
    stt = 0.0

    # Exchange transaction charge
    exchange_charge = trade_value * EXCHANGE_CHARGE_NSE

    # SEBI charges
    sebi_charge = trade_value * SEBI_CHARGE

    # Stamp duty (on buy side)
    stamp_duty = trade_value * STAMP_DUY_RATE

    # GST: 18% on (brokerage + exchange + SEBI)
    gst_base = brokerage + exchange_charge + sebi_charge
    gst = gst_base * GST_RATE

    total_charges = brokerage + stt + exchange_charge + sebi_charge + stamp_duty + gst
    net_cost = trade_value + total_charges

    return CostBreakdown(
        trade_value=trade_value,
        brokerage=brokerage,
        stt=stt,
        exchange_charge=exchange_charge,
        sebi_charge=sebi_charge,
        stamp_duty=stamp_duty,
        gst=gst,
        slippage=exec_price - price,
        total_charges=total_charges,
        net_proceeds=0.0,
        net_cost=net_cost,
    )


def calculate_sell_cost(
    price: float,
    quantity: int,
    slippage_pct: float = SELL_SLIPPAGE,
) -> CostBreakdown:
    """Calculate all costs for a SELL order.

    Args:
        price: Signal price (close price of exit day)
        quantity: Number of shares
        slippage_pct: Sell slippage rate (default 0.15%)

    Returns:
        CostBreakdown with all charges
    """
    exec_price = price * (1 - slippage_pct)
    trade_value = exec_price * quantity

    # Brokerage
    brokerage = trade_value * BROKERAGE_RATE

    # STT: 0.1% on sell side (equity delivery)
    stt = trade_value * STT_RATE

    # Exchange transaction charge
    exchange_charge = trade_value * EXCHANGE_CHARGE_NSE

    # SEBI charges
    sebi_charge = trade_value * SEBI_CHARGE

    # Stamp duty: not applicable on sell
    stamp_duty = 0.0

    # GST: 18% on (brokerage + exchange + SEBI)
    gst_base = brokerage + exchange_charge + sebi_charge
    gst = gst_base * GST_RATE

    total_charges = brokerage + stt + exchange_charge + sebi_charge + stamp_duty + gst
    net_proceeds = trade_value - total_charges

    return CostBreakdown(
        trade_value=trade_value,
        brokerage=brokerage,
        stt=stt,
        exchange_charge=exchange_charge,
        sebi_charge=sebi_charge,
        stamp_duty=stamp_duty,
        gst=gst,
        slippage=price - exec_price,
        total_charges=total_charges,
        net_proceeds=net_proceeds,
        net_cost=0.0,
    )


def apply_gap_risk(
    trigger_price: float,
    atr: float,
    exit_reason: str,
    seed: Optional[int] = None,
) -> float:
    """Apply realistic gap risk on exit fills.

    Models the fact that stop losses don't always fill at the trigger price.
    During gap-down opens, stops fill at the next available price (often much worse).

    Args:
        trigger_price: The price at which the exit signal triggered
        atr: Current ATR(14) value for the stock
        exit_reason: Reason for exit (STOP_LOSS, TARGET, TIME_STOP, etc.)
        seed: Random seed for reproducible results

    Returns:
        Realistic fill price accounting for gap risk
    """
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random

    # Stop loss exits are vulnerable to gap-down
    if "STOP" in exit_reason or "LOSS" in exit_reason or "ONEIL_STOP" in exit_reason:
        if rng.random() < GAP_DOWN_PROBABILITY:
            gap_severity = rng.uniform(GAP_MIN_SEVERITY_ATR, GAP_MAX_SEVERITY_ATR) * atr
            fill_price = trigger_price - gap_severity
            # Floor at circuit breaker limit (max 20% gap)
            fill_price = max(fill_price, trigger_price * (1 - CIRCUIT_BREAKER_LIMIT))
            return max(fill_price, 1.0)  # Floor at Rs 1

    # Target exits can gap up slightly in your favor
    elif "TARGET" in exit_reason or "SWING_HIGH" in exit_reason:
        if rng.random() < GAP_UP_PROBABILITY:
            gap_bonus = rng.uniform(GAP_UP_MIN_SEVERITY_ATR, GAP_UP_MAX_SEVERITY_ATR) * atr
            fill_price = trigger_price + gap_bonus
            return fill_price

    # Time stops, delisted, simulation end: fill at trigger price
    return trigger_price


def round_trip_cost_pct(price: float, quantity: int) -> float:
    """Calculate total round-trip cost as a percentage of trade value.

    Returns:
        Total cost percentage (e.g., 0.005 = 0.5%)
    """
    buy = calculate_buy_cost(price, quantity)
    sell = calculate_sell_cost(price, quantity)

    total_cost = buy.total_charges + sell.total_charges
    trade_value = price * quantity

    return total_cost / trade_value if trade_value > 0 else 0.0


def estimate_slippage_from_atr(atr: float, price: float) -> float:
    """Estimate realistic slippage based on ATR and price.

    Higher ATR stocks tend to have more slippage.

    Returns:
        Slippage percentage (e.g., 0.002 = 0.2%)
    """
    if price <= 0:
        return BUY_SLIPPAGE

    # Base slippage scales with volatility
    if atr / price > 0.04:  # Very volatile (>4% ATR)
        return 0.003  # 0.3%
    elif atr / price > 0.02:  # Moderately volatile
        return 0.002  # 0.2%
    else:  # Low volatility
        return 0.001  # 0.1%
