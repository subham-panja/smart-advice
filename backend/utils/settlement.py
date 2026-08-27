"""
Settlement Ledger — T+1 Indian Market Settlement Logic
File: utils/settlement.py

Mimics SEBI T+1 settlement rules for Indian equity markets:

1. NORMAL DELIVERY (shares already in demat, held >= 2 market days):
   - 80% of sale proceeds available immediately (Early Pay-In credit).
   - 20% locked until 1 market day passes (T+1).

2. BTST — Buy Today Sell Tomorrow (held < 2 market days, shares NOT in demat):
   - 0% available immediately (broker cannot do Early Pay-In for unsettled shares).
   - 100% locked until the ORIGINAL BUY trade settles (next market day).
   - This is a strict SEBI Peak Margin rule enforced by ALL Indian brokers
     (Dhan, Zerodha, Groww, Upstox, Angel One, etc.).

This module manages a `settlement_ledger` collection in MongoDB.
"""

import logging
from datetime import datetime, timedelta, timezone

import config
from utils.trading_clock import trading_now

logger = logging.getLogger(__name__)

# SEBI T+1 settlement splits
# Normal Delivery: shares settled in demat (held >= 2 market days)
NORMAL_IMMEDIATE_PCT = 0.80
NORMAL_UNSETTLED_PCT = 0.20

# BTST: shares NOT yet in demat (held < 2 market days)
BTST_IMMEDIATE_PCT = 0.00
BTST_UNSETTLED_PCT = 1.00


def _get_db():
    """Get MongoDB connection (standalone, no Flask context needed)."""
    from database import get_mongodb

    return get_mongodb()


def is_market_day(dt):
    """Check if a given date is a market day (Monday–Friday).

    NOTE: This does not account for Indian market holidays (Diwali, Republic Day, etc.).
    For production use, integrate an NSE holiday calendar.
    """
    if isinstance(dt, datetime):
        dt = dt.date()
    return dt.weekday() < 5  # 0=Mon, 4=Fri


def get_next_market_day(from_date):
    """Get the next market day after from_date."""
    if isinstance(from_date, datetime):
        from_date = from_date.date()
    d = from_date + timedelta(days=1)
    while not is_market_day(d):
        d += timedelta(days=1)
    return d


def count_market_days_between(start_date, end_date):
    """Count the number of market days between two dates (exclusive of start, inclusive of end).

    Examples:
        Buy Monday, Sell Tuesday  → 1 market day  → BTST
        Buy Monday, Sell Wednesday → 2 market days → Normal Delivery
        Buy Friday, Sell Monday   → 1 market day  → BTST
    """
    if isinstance(start_date, datetime):
        start_date = start_date.date()
    if isinstance(end_date, datetime):
        end_date = end_date.date()

    if end_date <= start_date:
        return 0

    count = 0
    d = start_date + timedelta(days=1)
    while d <= end_date:
        if is_market_day(d):
            count += 1
        d += timedelta(days=1)
    return count


def is_btst_trade(entry_date, exit_date):
    """Determine if a trade is BTST (shares not yet settled in demat).

    BTST = sold before the original buy has settled.
    Under T+1 settlement, shares settle 1 market day after purchase.
    If market_days_between(entry, exit) < 2, shares haven't settled → BTST.

    Args:
        entry_date: When the stock was bought.
        exit_date: When the stock was sold.

    Returns:
        True if BTST (0% immediate credit), False if Normal Delivery (80% credit).
    """
    gap = count_market_days_between(entry_date, exit_date)
    return gap < 2


def record_sale_proceeds(symbol, gross_amount, exit_date, exit_reason="", entry_date=None):
    """Record a sale's proceeds in the settlement ledger.

    Automatically detects BTST vs Normal Delivery:
    - BTST (entry→exit < 2 market days): 0% immediate, 100% settles T+1
    - Normal Delivery (≥ 2 market days): 80% immediate, 20% settles T+1

    Args:
        symbol: Stock symbol that was sold.
        gross_amount: Total net sale value (exit_price * quantity - brokerage).
        exit_date: When the exit occurred.
        exit_reason: Reason for exit (SL hit, target, etc.).
        entry_date: When the stock was originally bought. If None, assumes Normal Delivery.

    Returns:
        dict with 'immediate', 'unsettled' amounts, 'settlement_date', and 'is_btst'.
    """
    db = _get_db()
    col = db[config.MONGODB_COLLECTIONS["settlement_ledger"]]

    # Detect BTST vs Normal Delivery
    btst = False
    if entry_date is not None:
        btst = is_btst_trade(entry_date, exit_date)

    if btst:
        immediate_pct = BTST_IMMEDIATE_PCT
        unsettled_pct = BTST_UNSETTLED_PCT
        trade_type = "BTST"
    else:
        immediate_pct = NORMAL_IMMEDIATE_PCT
        unsettled_pct = NORMAL_UNSETTLED_PCT
        trade_type = "NORMAL_DELIVERY"

    immediate = round(gross_amount * immediate_pct, 2)
    unsettled = round(gross_amount * unsettled_pct, 2)

    if isinstance(exit_date, datetime):
        settlement_date = get_next_market_day(exit_date)
    else:
        settlement_date = get_next_market_day(exit_date)

    # Convert settlement_date to datetime for MongoDB storage
    settlement_dt = datetime(settlement_date.year, settlement_date.month, settlement_date.day)

    entry = {
        "symbol": symbol,
        "gross_amount": round(gross_amount, 2),
        "immediate_amount": immediate,
        "unsettled_amount": unsettled,
        "trade_type": trade_type,
        "is_btst": btst,
        "entry_date": entry_date
        if isinstance(entry_date, datetime)
        else (datetime.combine(entry_date, datetime.min.time()) if entry_date else None),
        "exit_date": exit_date if isinstance(exit_date, datetime) else datetime.combine(exit_date, datetime.min.time()),
        "settlement_date": settlement_dt,
        "exit_reason": exit_reason,
        "status": "UNSETTLED",
        "created_at": trading_now(timezone.utc).replace(tzinfo=None),
    }

    col.insert_one(entry)

    if btst:
        logger.important(
            f"⚠️ BTST SALE: {symbol} | "
            f"Gross: ₹{gross_amount:.2f} | "
            f"0% Available Today (shares not in demat) | "
            f"100% Locked: ₹{unsettled:.2f} | "
            f"Settles on: {settlement_date}"
        )
    else:
        logger.info(
            f"💰 Settlement recorded: {symbol} | "
            f"Gross: ₹{gross_amount:.2f} | "
            f"80% Available: ₹{immediate:.2f} | "
            f"20% Locked: ₹{unsettled:.2f} | "
            f"Settles on: {settlement_date}"
        )

    return {
        "immediate": immediate,
        "unsettled": unsettled,
        "settlement_date": settlement_date,
        "is_btst": btst,
        "trade_type": trade_type,
    }


def settle_pending_funds():
    """Release unsettled funds that have passed their T+1 settlement date.

    Should be called at the start of each trading cycle.
    Returns the total amount of funds that were just released.
    """
    db = _get_db()
    col = db[config.MONGODB_COLLECTIONS["settlement_ledger"]]

    now = trading_now(timezone.utc).replace(tzinfo=None)
    today = now.date() if isinstance(now, datetime) else now

    # Find all unsettled entries where settlement_date <= today
    today_dt = datetime(today.year, today.month, today.day, 23, 59, 59)
    pending = list(col.find({"status": "UNSETTLED", "settlement_date": {"$lte": today_dt}}))

    total_released = 0.0
    for entry in pending:
        amount = entry.get("unsettled_amount", 0)
        total_released += amount
        col.update_one({"_id": entry["_id"]}, {"$set": {"status": "SETTLED", "settled_at": now}})
        logger.info(f"✅ Settlement released: {entry['symbol']} | " f"₹{amount:.2f} now available (T+1 complete)")

    if total_released > 0:
        logger.important(f"💰 T+1 Settlement: ₹{total_released:.2f} released from {len(pending)} sale(s)")

    return total_released


def get_total_unsettled():
    """Get the total amount of funds still locked in T+1 settlement.

    Returns:
        float: Total unsettled amount across all pending entries.
    """
    db = _get_db()
    col = db[config.MONGODB_COLLECTIONS["settlement_ledger"]]

    unsettled_entries = list(col.find({"status": "UNSETTLED"}))
    total = sum(e.get("unsettled_amount", 0) for e in unsettled_entries)
    return round(total, 2)


def get_unsettled_entries():
    """Get all unsettled settlement ledger entries.

    Returns:
        list of dicts with symbol, amounts, and settlement dates.
    """
    db = _get_db()
    col = db[config.MONGODB_COLLECTIONS["settlement_ledger"]]
    return list(col.find({"status": "UNSETTLED"}))


def get_available_cash(initial_capital, open_positions, closed_positions):
    """Calculate available cash accounting for T+1 settlement.

    This replaces the simple `initial_capital - total_invested` calculation.
    Available = initial_capital + realized_pnl - total_invested - unsettled_20%

    Args:
        initial_capital: Starting capital.
        open_positions: List of open position dicts.
        closed_positions: List of closed position dicts.

    Returns:
        dict with 'available_cash', 'unsettled_funds', 'total_cash'.
    """
    total_invested = sum(p.get("total_investment", p["quantity"] * p["entry_price"]) for p in open_positions)

    realized_pnl = sum(
        (p.get("exit_price", 0) - p.get("entry_price", 0)) * p.get("quantity", 0) for p in closed_positions
    )

    # Total cash without settlement restriction
    total_cash = initial_capital + realized_pnl - total_invested

    # Subtract unsettled 20% that hasn't cleared T+1 yet
    unsettled = get_total_unsettled()
    available_cash = total_cash - unsettled

    return {
        "available_cash": round(available_cash, 2),
        "unsettled_funds": round(unsettled, 2),
        "total_cash": round(total_cash, 2),
    }
