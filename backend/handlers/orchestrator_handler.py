import logging
import queue as queue_module
import threading

from flask import Response

from utils.logger import log_queue, set_verbose

logger = logging.getLogger(__name__)

orchestrator_progress = {"status": "idle", "message": ""}


def run(req_data: dict):
    if orchestrator_progress["status"] == "running":
        return None, "Trading cycle already running."

    mode = req_data.get("mode", "live")
    verbose = req_data.get("verbose", False)

    def run_thread():
        try:
            set_verbose(verbose)
            orchestrator_progress["status"] = "running"
            orchestrator_progress["message"] = "Trading cycle started..."

            if mode == "replay":
                from main_orchestrator import run_replay

                days = req_data.get("replay_days", 5)
                run_replay(int(days))
            elif mode == "date":
                from main_orchestrator import run_single_date

                date_str = req_data.get("date")
                if not date_str:
                    raise ValueError("date is required for date mode")
                run_single_date(date_str)
            else:
                from main_orchestrator import run_trading_cycle

                run_trading_cycle()

            orchestrator_progress["status"] = "completed"
            orchestrator_progress["message"] = "Trading cycle completed."
        except Exception as e:
            orchestrator_progress["status"] = "error"
            orchestrator_progress["message"] = str(e)
            logger.error(f"Orchestrator error: {e}")

    thread = threading.Thread(target=run_thread)
    thread.daemon = True
    thread.start()

    return mode, None


def get_status():
    return orchestrator_progress


def stream_logs():
    def generate():
        while True:
            try:
                msg = log_queue.get(timeout=15)
                yield f"data: {msg}\n\n"
            except queue_module.Empty:
                yield ": keep-alive\n\n"

    return Response(generate(), mimetype="text/event-stream")


def get_pending_exits():
    """Return all positions awaiting exit price confirmation."""
    from database import get_pending_exit_confirmations

    pending = get_pending_exit_confirmations()
    result = []
    for p in pending:
        result.append(
            {
                "symbol": p.get("symbol", ""),
                "system_exit_price": p.get("system_exit_price", 0),
                "exit_reason": p.get("exit_reason", ""),
                "quantity": p.get("quantity", 0),
                "entry_price": p.get("entry_price", 0),
                "pnl_pct": p.get("pnl_pct", 0),
                "exit_date": p.get("exit_date", "").isoformat()
                if hasattr(p.get("exit_date", ""), "isoformat")
                else str(p.get("exit_date", "")),
                "created_at": p.get("created_at", "").isoformat()
                if hasattr(p.get("created_at", ""), "isoformat")
                else str(p.get("created_at", "")),
            }
        )
    return result


def confirm_exit(symbol: str, confirmed_price: float):
    """Confirm or correct the exit price for a closed position.

    This:
    1. Updates the closed position's exit_price and recalculates pnl_pct.
    2. Records sale proceeds in the settlement ledger (80/20 split).
    3. Removes the pending confirmation entry.
    4. Emits an SSE log confirming the update.

    Args:
        symbol: Stock symbol.
        confirmed_price: The user-confirmed (or corrected) exit price.

    Returns:
        dict with status and details, or None if no pending confirmation found.
    """
    import config
    from database import (
        get_mongodb,
        get_pending_exit_confirmations,
        resolve_pending_exit_confirmation,
    )
    from utils.settlement import record_sale_proceeds

    # Find the pending confirmation
    pending = get_pending_exit_confirmations()
    matching = [p for p in pending if p.get("symbol") == symbol]
    if not matching:
        return None

    pending_entry = matching[0]
    system_price = pending_entry.get("system_exit_price", 0)
    entry_price = pending_entry.get("entry_price", 0)
    quantity = pending_entry.get("quantity", 0)
    exit_date = pending_entry.get("exit_date")

    # Recalculate PnL with confirmed price
    new_pnl_pct = ((confirmed_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0

    # Update the closed position in DB with the confirmed price
    db = get_mongodb()
    col_name = config.MONGODB_COLLECTIONS["positions"]
    db[col_name].update_one(
        {"symbol": symbol, "status": "CLOSED"},
        {
            "$set": {
                "exit_price": confirmed_price,
                "pnl_pct": round(new_pnl_pct, 2),
                "exit_confirmed": True,
                "exit_confirmed_at": __import__("datetime")
                .datetime.now(__import__("datetime").timezone.utc)
                .replace(tzinfo=None),
            }
        },
        # Sort by exit_date descending to update the most recent closed position
    )

    # Record sale proceeds in settlement ledger (80% immediate, 20% T+1, or 0/100 for BTST)
    brokerage_pct = config.TRADING_OPTIONS.get("brokerage_charges", 0.0020)
    gross_proceeds = confirmed_price * quantity
    net_proceeds = gross_proceeds * (1 - brokerage_pct)
    entry_date = pending_entry.get("entry_date")
    settlement_info = record_sale_proceeds(
        symbol, net_proceeds, exit_date, pending_entry.get("exit_reason", ""), entry_date
    )

    # Mark the pending confirmation as resolved
    resolve_pending_exit_confirmation(symbol, confirmed_price)

    price_diff = confirmed_price - system_price
    diff_str = f"+₹{price_diff:.2f}" if price_diff >= 0 else f"-₹{abs(price_diff):.2f}"

    if settlement_info.get("is_btst"):
        avail_str = "0% Available (BTST)"
        lock_str = f"100% Settles: {settlement_info['settlement_date']}"
    else:
        avail_str = f"80% Available: ₹{settlement_info['immediate']:.2f}"
        lock_str = f"20% Settles: {settlement_info['settlement_date']}"

    if abs(price_diff) > 0.01:
        logger.important(
            f"✅ EXIT CONFIRMED: {symbol} | "
            f"System: ₹{system_price:.2f} → Actual: ₹{confirmed_price:.2f} ({diff_str}) | "
            f"PnL: {new_pnl_pct:+.2f}% | "
            f"{avail_str} | {lock_str}"
        )
    else:
        logger.important(
            f"✅ EXIT CONFIRMED: {symbol} @ ₹{confirmed_price:.2f} | "
            f"PnL: {new_pnl_pct:+.2f}% | "
            f"{avail_str} | {lock_str}"
        )

    return {
        "symbol": symbol,
        "confirmed_price": confirmed_price,
        "system_price": system_price,
        "price_diff": round(price_diff, 2),
        "pnl_pct": round(new_pnl_pct, 2),
        "settlement": {
            "immediate": settlement_info["immediate"],
            "unsettled": settlement_info["unsettled"],
            "settlement_date": str(settlement_info["settlement_date"]),
        },
    }
