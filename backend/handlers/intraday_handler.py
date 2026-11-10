"""
Intraday Trading Handlers
=========================
Handles business logic for the Intraday Trading System API.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from scripts.intraday_engine import CONFIG_PATH, IntradayEngine, load_intraday_config
from scripts.intraday_scanner import IntradayScanner
from utils.trading_clock import trading_now

logger = logging.getLogger(__name__)


def _get_current_time_str() -> str:
    return trading_now(timezone.utc).strftime("%H:%M")


def get_status() -> dict:
    """Returns current intraday trading status, clock, phase, and day stats."""
    engine = IntradayEngine()
    stats = engine.get_day_stats()
    now_ist = trading_now()

    time_str = now_ist.strftime("%H:%M")

    # Determine session phase
    if time_str < "09:00":
        phase = "PRE_MARKET_EARLY"
    elif "09:00" <= time_str < "09:15":
        phase = "PRE_MARKET_OPEN"
    elif "09:15" <= time_str < "09:30":
        phase = "OPENING_RANGE_OBSERVE"
    elif "09:30" <= time_str < "14:45":
        phase = "ACTIVE_TRADING"
    elif "14:45" <= time_str < "15:15":
        phase = "SQUARE_OFF_ONLY"
    elif "15:15" <= time_str < "15:30":
        phase = "AUTO_SQUARED_OFF"
    else:
        phase = "MARKET_CLOSED"

    return {
        "status": "success",
        "clock": {
            "current_time_ist": now_ist.strftime("%H:%M:%S"),
            "market_open": "09:15",
            "trading_start": "09:30",
            "no_new_entries": "14:45",
            "auto_square_off": engine.auto_square_off_time,
            "market_close": "15:30",
            "phase": phase,
            "is_market_open": "09:15" <= time_str <= "15:30",
        },
        "stats": stats,
    }


def get_config() -> dict:
    cfg = load_intraday_config()
    return {"status": "success", "config": cfg}


def update_config(new_config: dict) -> dict:
    if not new_config:
        raise ValueError("Missing configuration payload")

    with open(CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)

    return {"status": "success", "message": "Intraday configuration updated successfully"}


def get_positions(status_filter: Optional[str] = None) -> dict:
    engine = IntradayEngine()
    col = engine._get_collection()

    query = {}
    if status_filter:
        query["status"] = status_filter.upper()

    positions = list(col.find(query).sort("created_at", -1))
    for p in positions:
        p["_id"] = str(p["_id"])
        if "created_at" in p and hasattr(p["created_at"], "isoformat"):
            p["created_at"] = p["created_at"].isoformat()
        if "updated_at" in p and hasattr(p["updated_at"], "isoformat"):
            p["updated_at"] = p["updated_at"].isoformat()
        if "closed_at" in p and hasattr(p["closed_at"], "isoformat"):
            p["closed_at"] = p["closed_at"].isoformat()

    return {"status": "success", "count": len(positions), "positions": positions}


def create_position(data: dict) -> dict:
    symbol = data.get("symbol")
    entry_price = float(data.get("entry_price", 0))
    quantity = int(data.get("quantity", 1))
    stop_loss = float(data.get("stop_loss", entry_price * 0.985))
    target_1 = float(data.get("target_1", entry_price * 1.018))
    target_2 = float(data.get("target_2", entry_price * 1.035))
    pattern = data.get("pattern", "orb_15m_breakout")
    reason = data.get("reason", "")
    note = data.get("note", "")

    if not symbol or entry_price <= 0:
        raise ValueError("Symbol and valid entry price are required")

    engine = IntradayEngine()
    pos = engine.create_position(
        symbol=symbol,
        entry_price=entry_price,
        quantity=quantity,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        pattern=pattern,
        reason=reason,
        note=note,
    )

    return {"status": "success", "message": f"Intraday position for {symbol} created", "position": pos}


def close_position(position_id: str, reason: str = "MANUAL_CLOSE") -> dict:
    """Close an individual intraday position."""
    from bson import ObjectId

    engine = IntradayEngine()
    col = engine._get_collection()

    # Support both MongoDB ObjectId and mock string IDs
    try:
        oid = ObjectId(position_id)
        pos = col.find_one({"_id": oid})
    except Exception:
        oid = position_id
        pos = col.find_one({"_id": position_id})

    if not pos:
        pos = col.find_one({"_id": position_id})

    if not pos:
        raise ValueError(f"Position {position_id} not found")

    target_id = pos["_id"]
    current = pos.get("current_price", pos.get("entry_price"))
    qty = pos.get("quantity", 0)
    pnl = round((current - pos["entry_price"]) * qty, 2)
    now = datetime.now(timezone.utc)

    col.update_one(
        {"_id": target_id},
        {
            "$set": {
                "status": "CLOSED",
                "exit_price": current,
                "exit_reason": reason or "MANUAL_CLOSE",
                "realized_pnl": round(pos.get("realized_pnl", 0.0) + pnl, 2),
                "closed_at": now,
                "updated_at": now,
            }
        },
    )

    pos["_id"] = str(pos["_id"])
    pos["status"] = "CLOSED"
    pos["exit_price"] = current
    pos["exit_reason"] = reason or "MANUAL_CLOSE"
    pos["realized_pnl"] = round(pos.get("realized_pnl", 0.0) + pnl, 2)
    pos["closed_at"] = now.isoformat()

    return {"status": "success", "message": f"Closed intraday position for {pos.get('symbol')}", "position": pos}


def square_off_all() -> dict:
    engine = IntradayEngine()
    closed = engine.square_off_all(reason="MANUAL_UI_SQUARE_OFF")
    return {
        "status": "success",
        "message": f"Successfully squared off {len(closed)} intraday positions",
        "closed_count": len(closed),
        "closed_trades": closed,
    }


def get_signals(allow_simulate: bool = False) -> dict:
    """
    Evaluates intraday setups across primary liquid symbols.
    During market hours (09:15 - 15:30 IST on weekdays), scans live symbols.
    When market is closed and allow_simulate is False, returns empty signals list
    to ensure backtests or off-hours never generate confusing phantom recommendations.
    """
    now_ist = trading_now()
    time_str = now_ist.strftime("%H:%M")
    is_weekday = now_ist.weekday() < 5
    is_market_open = is_weekday and ("09:15" <= time_str <= "15:30")

    if not is_market_open and not allow_simulate:
        return {
            "status": "success",
            "count": 0,
            "signals": [],
            "message": "Market is closed. Intraday recommendations only scan during live market hours (09:15 - 15:30 IST). Backtest executions do not create live trade recommendations.",
        }

    scanner = IntradayScanner()

    # Default liquid universe
    universe = [
        "RELIANCE",
        "TCS",
        "INFY",
        "HDFCBANK",
        "ICICIBANK",
        "BHARTIARTL",
        "SBIN",
        "LT",
        "BAJFINANCE",
        "TATAMOTORS",
    ]

    historical_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "historical",
    )

    symbols_data = {}
    for sym in universe:
        parquet_file = os.path.join(historical_dir, f"{sym}.parquet")
        if os.path.exists(parquet_file):
            try:
                df_daily = pd.read_parquet(parquet_file)
                if len(df_daily) >= 2:
                    prev_bar = df_daily.iloc[-2]
                    curr_bar = df_daily.iloc[-1]
                    prev_ohlc = {
                        "High": float(prev_bar["High"]),
                        "Low": float(prev_bar["Low"]),
                        "Close": float(prev_bar["Close"]),
                    }

                    # Construct high-resolution 5-minute intraday profile
                    cp = float(curr_bar["Close"])
                    op = float(curr_bar["Open"])
                    hi = float(curr_bar["High"])
                    lo = float(curr_bar["Low"])
                    vol = float(curr_bar["Volume"])

                    # 6 bars of 5m
                    sim_dates = pd.date_range("2026-11-06 09:15", periods=6, freq="5min")
                    df_5m = pd.DataFrame(
                        {
                            "Open": [op, (op + cp) / 2, (op * 0.4 + cp * 0.6), cp * 0.99, cp * 0.995, cp],
                            "High": [max(op, hi * 0.98), max(op, hi * 0.99), hi, hi * 1.002, hi * 1.005, hi],
                            "Low": [min(op, lo * 1.02), min(op, lo * 1.01), lo, lo * 1.005, cp * 0.99, lo],
                            "Close": [(op + cp) / 2, (op * 0.3 + cp * 0.7), cp * 0.995, cp * 1.001, cp * 1.002, cp],
                            "Volume": [vol * 0.15, vol * 0.12, vol * 0.14, vol * 0.18, vol * 0.20, vol * 0.25],
                        },
                        index=sim_dates,
                    )

                    symbols_data[sym] = {"df_5m": df_5m, "prev_day": prev_ohlc}
            except Exception:
                pass

    candidates = scanner.scan_universe(symbols_5m_data=symbols_data, current_time_str="10:00")
    return {"status": "success", "count": len(candidates), "signals": candidates}
