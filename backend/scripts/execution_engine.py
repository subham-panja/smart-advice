"""
Execution Engine
File: scripts/execution_engine.py

Handles trade execution (Buying/Selling) with Paper Trading Mock support.
"""

import logging
from datetime import datetime
from config import TRADING_OPTIONS, RISK_MANAGEMENT
from database import insert_position, close_position, get_open_positions

logger = logging.getLogger(__name__)

class ExecutionEngine:
    def __init__(self):
        self.is_paper_trading = TRADING_OPTIONS.get('is_paper_trading', True)
        self.brokerage_pct = TRADING_OPTIONS.get('brokerage_charges', 0.0005)

    def execute_buy(self, symbol, quantity, price, stop_loss, target, recomm_id=None):
        """Execute a buy order (Mocked for Paper Trading)."""
        if self.is_paper_trading:
            logger.info(f"PAPER TRADING: Executing BUY for {symbol} | Qty: {quantity} | Price: {price}")
            
            # Check if we already have an open position for this stock
            if not TRADING_OPTIONS.get('allow_multiple_positions_same_stock', False):
                open_pos = get_open_positions()
                if any(p['symbol'] == symbol for p in open_pos):
                    logger.warning(f"Skipping {symbol}: Position already exists and multiple positions disabled.")
                    return None

            # Calculate total cost including brokerage
            total_cost = quantity * price
            brokerage = total_cost * self.brokerage_pct
            
            # In a real app, we'd deduct from account balance here.
            # For now, we just log it and store it in MongoDB.
            
            try:
                res = insert_position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    stop_loss=stop_loss,
                    target=target,
                    recomm_id=recomm_id
                )
                logger.info(f"Position stored in MongoDB for {symbol}")
                return res
            except Exception as e:
                logger.error(f"Failed to store position for {symbol}: {e}")
                return None
        else:
            # TODO: Add 5paisa real API execution here
            logger.warning("Real trading execution not yet implemented.")
            return None

    def execute_sell(self, symbol, price, reason):
        """Execute a sell order (Mocked for Paper Trading)."""
        if self.is_paper_trading:
            logger.info(f"PAPER TRADING: Executing SELL for {symbol} | Price: {price} | Reason: {reason}")
            
            try:
                res = close_position(symbol, price, reason)
                if res and res.modified_count > 0:
                    logger.info(f"Successfully closed position for {symbol}")
                    return True
                else:
                    logger.error(f"Could not find open position to close for {symbol}")
                    return False
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
                return False
        else:
            # TODO: Add 5paisa real API execution here
            logger.warning("Real trading execution not yet implemented.")
            return False
