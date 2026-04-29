"""
Execution Engine
File: scripts/execution_engine.py

Handles trade execution (Buying/Selling) with Paper Trading Mock support.
"""

import logging
from datetime import datetime

from config import TRADING_OPTIONS
from database import close_position, get_open_positions, insert_position

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self):
        self.is_paper_trading = TRADING_OPTIONS.get("is_paper_trading", True)
        self.brokerage_pct = TRADING_OPTIONS.get("brokerage_charges", 0.0005)
        self.initial_capital = TRADING_OPTIONS.get("initial_capital", 100000.0)

    def execute_buy(self, symbol, quantity, price, stop_loss, target, recomm_id=None):
        """Execute a buy order with detailed metadata and position sizing analysis."""
        mode = "PAPER" if self.is_paper_trading else "LIVE"
        if self.is_paper_trading:
            logger.info(f"{mode} TRADING: Executing BUY for {symbol} | Qty: {quantity} | Price: {price}")

            if not TRADING_OPTIONS.get("allow_multiple_positions_same_stock", False):
                open_pos = get_open_positions()
                if any(p["symbol"] == symbol for p in open_pos):
                    logger.warning(f"Skipping {symbol}: Position already exists.")
                    return None

            try:
                total_investment = round(quantity * price, 2)
                allocation_pct = round((total_investment / self.initial_capital) * 100, 2)

                pos_data = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "entry_price": round(price, 2),
                    "total_investment": total_investment,
                    "allocation_pct": allocation_pct,
                    "stop_loss": round(stop_loss, 2),
                    "current_stop_loss": round(stop_loss, 2),
                    "target": round(target, 2),
                    "current_target": round(target, 2),
                    "recomm_id": recomm_id,
                    "entry_date": datetime.now(),
                    "status": "OPEN",
                    "trade_type": "LONG_BUY",
                    "is_paper": self.is_paper_trading,
                    "initial_risk": round((price - stop_loss) * quantity, 2),
                    "risk_pct_of_cap": round(((price - stop_loss) * quantity / self.initial_capital) * 100, 2),
                }
                res = insert_position(pos_data)
                logger.info(f"Position sizing validated: {allocation_pct}% of capital allocated to {symbol}")
                return res
            except Exception as e:
                logger.error(f"Failed to store position for {symbol}: {e}")
                return None
        else:
            logger.warning("Real trading execution not yet implemented.")
            return None

    def execute_sell(self, symbol, price, reason):
        """Execute a sell order (Mocked for Paper Trading)."""
        mode = "PAPER" if self.is_paper_trading else "LIVE"
        if self.is_paper_trading:
            logger.info(f"{mode} TRADING: Executing SELL for {symbol} | Price: {price} | Reason: {reason}")
            try:
                res = close_position(symbol, price, reason)
                if res and (getattr(res, "modified_count", 0) > 0 or getattr(res, "upserted_id", None)):
                    logger.info(f"Successfully closed position for {symbol}")
                    return True
                else:
                    logger.error(f"Could not find open position to close for {symbol}")
                    return False
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
                return False
        else:
            logger.warning("Real trading execution not yet implemented.")
            return False
