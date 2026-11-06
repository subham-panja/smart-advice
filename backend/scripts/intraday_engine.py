"""
Intraday Trading Engine & Risk Management
=========================================
Manages intraday position execution, TP1 profit-taking, breakeven stop-trails,
and strict 15:15 IST auto-square-off rules.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

# Strategy config path
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "strategies",
    "intraday_trading.json",
)


def load_intraday_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load intraday config: {e}")
    return {}


class IntradayEngine:
    """Manages active intraday positions and automated exit rules."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_intraday_config()
        self.exit_rules = self.config.get("exit_rules", {})
        self.risk_mgmt = self.config.get("risk_management", {})
        self.time_cfg = self.config.get("timeframe_config", {})
        self.auto_square_off_time = self.time_cfg.get("auto_square_off_time", "15:15")

    def _get_collection(self):
        from database import get_mongodb

        db = get_mongodb()
        return db["intraday_positions"]

    def create_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target_1: float,
        target_2: float,
        pattern: str = "orb_15m_breakout",
        reason: str = "",
        note: str = "",
    ) -> dict:
        """Create and persist an intraday position."""
        now = datetime.now(timezone.utc)
        pos = {
            "symbol": symbol.upper(),
            "quantity": int(quantity),
            "initial_quantity": int(quantity),
            "entry_price": float(entry_price),
            "current_price": float(entry_price),
            "stop_loss": float(stop_loss),
            "target_1": float(target_1),
            "target_2": float(target_2),
            "pattern": pattern,
            "reason": reason,
            "note": note,
            "status": "OPEN",
            "t1_hit": False,
            "total_investment": round(entry_price * quantity, 2),
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "pnl_pct": 0.0,
            "entry_time": now.strftime("%H:%M:%S"),
            "created_at": now,
            "updated_at": now,
        }

        col = self._get_collection()
        res = col.insert_one(pos)
        pos_copy = dict(pos)
        pos_copy["_id"] = str(res.inserted_id or pos.get("_id", ""))
        return pos_copy

    def get_open_positions(self) -> List[dict]:
        col = self._get_collection()
        docs = list(col.find({"status": "OPEN"}))
        for d in docs:
            d["_id"] = str(d["_id"])
        return docs

    def process_price_tick(
        self,
        symbol: str,
        current_price: float,
        current_time_str: str = "11:00",
    ) -> Optional[dict]:
        """
        Processes price tick against an active position.
        Checks:
        1. 15:15 IST Mandatory Square-Off
        2. Stop Loss Hit
        3. Target 1 Hit (Book 50% + Move SL to Breakeven)
        4. Target 2 Hit (Close remainder)
        """
        col = self._get_collection()
        pos = col.find_one({"symbol": symbol.upper(), "status": "OPEN"})
        if not pos:
            return None

        entry = pos["entry_price"]
        sl = pos["stop_loss"]
        t1 = pos["target_1"]
        t2 = pos["target_2"]
        qty = pos["quantity"]
        t1_hit = pos.get("t1_hit", False)

        unrealized = round((current_price - entry) * qty, 2)
        pnl_pct = round(((current_price - entry) / entry) * 100.0, 2) if entry > 0 else 0.0

        update_fields = {
            "current_price": current_price,
            "unrealized_pnl": unrealized,
            "pnl_pct": pnl_pct,
            "updated_at": datetime.now(timezone.utc),
        }

        # 1. Mandatory Auto Square-Off Check at 15:15
        if current_time_str >= self.auto_square_off_time:
            update_fields.update(
                {
                    "status": "CLOSED",
                    "exit_price": current_price,
                    "exit_reason": f"AUTO_SQUARE_OFF_{self.auto_square_off_time}",
                    "realized_pnl": unrealized,
                    "closed_at": datetime.now(timezone.utc),
                }
            )
            col.update_one({"_id": pos["_id"]}, {"$set": update_fields})
            pos.update(update_fields)
            return {"action": "AUTO_SQUARE_OFF", "symbol": symbol, "exit_price": current_price, "pnl": unrealized}

        # 2. Stop Loss Check
        if current_price <= sl:
            update_fields.update(
                {
                    "status": "CLOSED",
                    "exit_price": current_price,
                    "exit_reason": "STOP_LOSS_HIT",
                    "realized_pnl": unrealized,
                    "closed_at": datetime.now(timezone.utc),
                }
            )
            col.update_one({"_id": pos["_id"]}, {"$set": update_fields})
            pos.update(update_fields)
            return {"action": "STOP_LOSS", "symbol": symbol, "exit_price": current_price, "pnl": unrealized}

        # 3. Target 1 Check (Partial Profit Booking + Breakeven SL)
        if not t1_hit and current_price >= t1:
            sell_qty = max(1, qty // 2)
            remaining_qty = qty - sell_qty
            booked_pnl = round((current_price - entry) * sell_qty, 2)

            update_fields.update(
                {
                    "t1_hit": True,
                    "quantity": remaining_qty,
                    "stop_loss": entry,  # Breakeven Stop
                    "realized_pnl": pos.get("realized_pnl", 0.0) + booked_pnl,
                }
            )
            col.update_one({"_id": pos["_id"]}, {"$set": update_fields})
            pos.update(update_fields)
            return {"action": "TARGET_1_HIT", "symbol": symbol, "booked_pnl": booked_pnl, "new_sl": entry}

        # 4. Target 2 Check (Close remaining)
        if t1_hit and current_price >= t2:
            remaining_pnl = round((current_price - entry) * qty, 2)
            total_pnl = pos.get("realized_pnl", 0.0) + remaining_pnl
            update_fields.update(
                {
                    "status": "CLOSED",
                    "exit_price": current_price,
                    "exit_reason": "TARGET_2_HIT",
                    "realized_pnl": round(total_pnl, 2),
                    "closed_at": datetime.now(timezone.utc),
                }
            )
            col.update_one({"_id": pos["_id"]}, {"$set": update_fields})
            pos.update(update_fields)
            return {"action": "TARGET_2_HIT", "symbol": symbol, "exit_price": current_price, "pnl": total_pnl}

        # Just update mark-to-market
        col.update_one({"_id": pos["_id"]}, {"$set": update_fields})
        return None

    def square_off_all(self, reason: str = "MANUAL_EMERGENCY") -> List[dict]:
        """Emergency square-off for all active intraday positions."""
        col = self._get_collection()
        open_positions = list(col.find({"status": "OPEN"}))
        closed_records = []

        now = datetime.now(timezone.utc)
        for p in open_positions:
            current = p.get("current_price", p["entry_price"])
            qty = p.get("quantity", 0)
            pnl = round((current - p["entry_price"]) * qty, 2)

            col.update_one(
                {"_id": p["_id"]},
                {
                    "$set": {
                        "status": "CLOSED",
                        "exit_price": current,
                        "exit_reason": reason,
                        "realized_pnl": p.get("realized_pnl", 0.0) + pnl,
                        "closed_at": now,
                        "updated_at": now,
                    }
                },
            )
            closed_records.append({"symbol": p["symbol"], "exit_price": current, "pnl": pnl})

        return closed_records

    def get_day_stats(self) -> dict:
        """Returns intraday realized P&L, open count, and risk utilization."""
        col = self._get_collection()
        open_pos = list(col.find({"status": "OPEN"}))
        closed_pos = list(col.find({"status": "CLOSED"}))

        realized_pnl = sum(p.get("realized_pnl", 0.0) for p in closed_pos)
        unrealized_pnl = sum(p.get("unrealized_pnl", 0.0) for p in open_pos)
        total_investment = sum(p.get("total_investment", 0.0) for p in open_pos)

        max_positions = self.risk_mgmt.get("max_positions", 3)
        max_daily_loss = self.risk_mgmt.get("max_daily_loss_pct", 2.0)

        return {
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl": round(realized_pnl + unrealized_pnl, 2),
            "open_count": len(open_pos),
            "max_positions": max_positions,
            "positions_available": max(0, max_positions - len(open_pos)),
            "capital_deployed": round(total_investment, 2),
            "circuit_breaker_active": False,
            "max_daily_loss_pct": max_daily_loss,
            "auto_square_off_time": self.auto_square_off_time,
        }
