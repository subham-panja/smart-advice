"""
Execution Engine
File: scripts/execution_engine_paper.py

Handles trade execution (Buying/Selling) with Paper Trading Mock support.
"""

import logging
from datetime import datetime, timezone

import config
from config import TRADING_OPTIONS
from database import close_position, get_open_positions, insert_position, update_position

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self):
        self.is_paper_trading = TRADING_OPTIONS["is_paper_trading"]
        self.brokerage_pct = TRADING_OPTIONS["brokerage_charges"]
        self.initial_capital = TRADING_OPTIONS["initial_capital"]

    def execute_buy(self, symbol, quantity, price, stop_loss, target, recomm_id=None, strategy_name="UNKNOWN"):
        """Execute a buy order with detailed metadata and pyramiding support."""
        mode = "PAPER" if self.is_paper_trading else "LIVE"
        if self.is_paper_trading:
            logger.info(f"{mode} TRADING: Executing BUY for {symbol} | Qty: {quantity} | Price: {price}")

            open_positions = get_open_positions()
            existing_pos = next((p for p in open_positions if p["symbol"] == symbol), None)

            pyramid_cfg = config.RISK_MANAGEMENT["pyramiding"]
            can_pyramid = pyramid_cfg["enabled"]

            if existing_pos:
                if not can_pyramid:
                    logger.warning(f"Skipping {symbol}: Position already exists and pyramiding disabled.")
                    return None

                adds = existing_pos.get("adds_count", 0)
                pyramid_steps = pyramid_cfg["steps"]

                if adds >= len(pyramid_steps):
                    logger.warning(f"Skipping {symbol}: Max pyramid steps ({len(pyramid_steps)}) reached.")
                    return None

                step_obj = pyramid_steps[adds]
                trigger_atr_mult = step_obj["trigger_step_atr"]

                # PRICE TRIGGER CHECK: Only add if price moved up by X * ATR
                try:
                    import talib as ta

                    from scripts.data_fetcher import get_historical_data

                    hist = get_historical_data(symbol, period="60d")
                    atr = ta.ATR(hist["High"], hist["Low"], hist["Close"], timeperiod=14).iloc[-1]
                    last_price = existing_pos.get("last_add_price", existing_pos["entry_price"])
                    required_price = last_price + (trigger_atr_mult * atr)

                    if price < required_price:
                        msg = f"⏳ {step_obj['name']} deferred for {symbol}: Price ₹{price} < Required ₹{required_price:.2f}"
                        print(msg)
                        logger.info(msg)
                        return None
                except Exception as e:
                    logger.error(f"Critical error in pyramid trigger check for {symbol}: {e}")
                    raise e

                # Calculate Pyramid Quantity
                base_qty = existing_pos.get("initial_quantity", existing_pos["quantity"])
                add_pct = step_obj["add_size_pct"]
                pyramid_qty = max(int(base_qty * add_pct), 1)

                # Update existing position (Pyramiding)
                new_qty = existing_pos["quantity"] + pyramid_qty
                new_total_inv = existing_pos["total_investment"] + (pyramid_qty * price)
                new_avg_price = new_total_inv / new_qty

                update_data = {
                    "quantity": new_qty,
                    "entry_price": round(new_avg_price, 2),
                    "total_investment": round(new_total_inv, 2),
                    "adds_count": adds + 1,
                    "last_add_price": price,
                    "last_add_date": datetime.now(),
                }
                update_position(symbol, update_data)
                msg = f"✅ PYRAMID {step_obj['name']}: {symbol} | Added {pyramid_qty} shares ({int(add_pct*100)}%). New Avg: {new_avg_price:.2f}"
                print(msg)
                logger.info(msg)
                return True

            try:
                total_investment = round(quantity * price, 2)
                allocation_pct = round((total_investment / self.initial_capital) * 100, 2)

                pos_data = {
                    "symbol": symbol,
                    "quantity": quantity,
                    "initial_quantity": quantity,
                    "entry_price": round(price, 2),
                    "total_investment": total_investment,
                    "allocation_pct": allocation_pct,
                    "stop_loss": round(stop_loss, 2),
                    "current_stop_loss": round(stop_loss, 2),
                    "target": round(target, 2),
                    "current_target": round(target, 2),
                    "recomm_id": recomm_id,
                    "strategy_name": strategy_name,
                    "entry_date": datetime.now(timezone.utc).replace(tzinfo=None),
                    "status": "OPEN",
                    "trade_type": "LONG_BUY",
                    "is_paper": self.is_paper_trading,
                    "initial_risk": round((price - stop_loss) * quantity, 2),
                    "risk_pct_of_cap": round(((price - stop_loss) * quantity / self.initial_capital) * 100, 2),
                    "adds_count": 0,
                }
                res = insert_position(pos_data)
                logger.info(f"Position sizing validated: {allocation_pct}% of capital allocated to {symbol}")
                return res
            except Exception as e:
                logger.error(f"Critical execution error for {symbol}: {e}")
                raise e
        else:
            logger.warning("Real trading execution not yet implemented.")
            raise NotImplementedError("Live trading not supported in paper engine.")

    def execute_sell(self, symbol, price, reason, quantity=None):
        """Execute a sell order (Supports Partial or Full exits)."""
        if self.is_paper_trading:
            qty_str = f"Qty: {quantity}" if quantity else "FULL Position"
            logger.info(f"PAPER TRADING: Executing SELL for {symbol} | {qty_str} | Price: {price} | Reason: {reason}")

            try:
                if quantity:
                    logger.info(f"✅ Partial Sell Recorded for {symbol}: {quantity} units at {price}")
                    return True
                else:
                    # Full Exit
                    res = close_position(symbol, price, reason)
                    if res and (getattr(res, "modified_count", 0) > 0 or getattr(res, "upserted_id", None)):
                        logger.info(f"Successfully closed position for {symbol}")
                        return True
                    else:
                        raise ValueError(f"Could not find open position to close for {symbol}")
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")
                raise e
        else:
            logger.warning("Real trading execution not yet implemented.")
            raise NotImplementedError("Live trading not supported in paper engine.")
