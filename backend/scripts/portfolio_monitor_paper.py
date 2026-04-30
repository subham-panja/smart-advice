"""
Portfolio Monitor
File: scripts/portfolio_monitor.py

Responsible for:
1. End-of-day trailing stop loss updates.
2. Hard Target and Stop Loss exit detection.
3. Time-stop checks (sideways fund lock release).
4. Live price updates for PnL tracking.
"""

import logging
from datetime import datetime, timezone

from config import TRADING_OPTIONS
from database import get_open_positions, update_position
from scripts.data_fetcher import get_historical_data
from utils.strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    def __init__(self):
        self.time_stop_days = TRADING_OPTIONS.get("time_stop_days", 15)

    def monitor_all_positions(self):
        """Monitor all open positions for exits or SL updates."""
        positions = get_open_positions()
        if not positions:
            logger.info("No open positions to monitor.")
            return

        logger.info(f"Monitoring {len(positions)} active positions...")
        for pos in positions:
            self._process_single_position(pos)

    def _process_single_position(self, pos):
        symbol = pos["symbol"]
        strat_name = pos.get("strategy_name", "Delayed_EP")
        strategy = StrategyLoader.get_strategy_by_name(strat_name)
        exit_rules = strategy.get("exit_rules", {}) if strategy else {}

        try:
            # 1. Fetch latest data (Live Price Sync)
            data = get_historical_data(symbol, period="1mo", fresh=True)
            if data.empty:
                logger.warning(f"Could not fetch data for {symbol}")
                return

            current_price = round(data["Close"].iloc[-1], 2)
            entry_price = pos["entry_price"]
            entry_date = pos["entry_date"]
            current_sl = pos.get("current_stop_loss", pos.get("stop_loss"))
            target = pos.get("target")

            # Update Live Price in DB for Telegram PnL
            update_position(symbol, {"current_price": current_price})

            from scripts.execution_engine_paper import ExecutionEngine

            engine = ExecutionEngine()

            # 2. Check for Hard Exit: Stop Loss Hit
            if current_price <= current_sl:
                logger.info(f"🛑 STOP LOSS HIT: {symbol} at ₹{current_price:.2f}")
                engine.execute_sell(symbol, current_price, "STOP_LOSS_HIT")
                return

            # 6. Structured Target Monitoring (Scaling Out & Target Hits)
            targets = exit_rules.get("targets", [])
            # Fetch ATR for Target calculations
            import talib as ta

            atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=14).iloc[-1]

            # Find the next un-hit target
            # Track which target we are currently on
            current_target_idx = pos.get("current_target_idx", 0)

            if current_target_idx < len(targets):
                target_obj = targets[current_target_idx]
                target_price = entry_price + (target_obj["atr_multiplier"] * atr)

                if current_price >= target_price:
                    sell_pct = target_obj.get("sell_percentage", 1.0)
                    logger.info(f"🎯 {target_obj['name']} HIT: Price ₹{current_price:.2f} >= ₹{target_price:.2f}")

                    if sell_pct < 1.0:
                        # Partial Sell (Scale Out)
                        sell_qty = int(pos["quantity"] * sell_pct)
                        if sell_qty > 0:
                            engine.execute_sell(
                                symbol, current_price, f"PARTIAL_{target_obj['name']}", quantity=sell_qty
                            )
                            rem_qty = pos["quantity"] - sell_qty
                            update_data = {
                                "quantity": rem_qty,
                                "current_target_idx": current_target_idx + 1,
                                "is_scaled_out": True,
                            }
                            # Auto-Breakeven if enabled and this is Target 1
                            if current_target_idx == 0 and exit_rules.get("breakeven_at_target_1", True):
                                update_data["current_stop_loss"] = entry_price
                                logger.info(f"🛡️ SL moved to Breakeven (₹{entry_price:.2f})")

                            update_position(symbol, update_data)
                            return  # Exit process for this stock today
                    else:
                        # Full Exit (Final Target)
                        engine.execute_sell(symbol, current_price, f"FINAL_{target_obj['name']}")
                        return

            # 7. VTT-Style Trailing SL Update
            from scripts.risk_management import RiskManager

            rm = RiskManager()
            sl_info = rm.calculate_stop_loss(
                data, current_price, method="atr", atr_multiplier=exit_rules.get("trail_stop_atr", 3.0)
            )

            new_sl = sl_info["stop_loss"]
            if new_sl > current_sl:
                logger.info(f"📉 VTT UPDATE: Trailing {symbol} SL from {current_sl:.2f} to {new_sl:.2f}")
                update_position(symbol, {"current_stop_loss": new_sl})
                current_sl = new_sl  # Update local variable for next check

            # 8. Time Stop (Sideways)
            days_held = (datetime.now(timezone.utc).replace(tzinfo=None) - entry_date).days
            if days_held >= self.time_stop_days:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                if pnl_pct < 2.0:
                    logger.info(f"⏳ TIME STOP: {symbol} held for {days_held} days. Exit due to stagnation.")
                    engine.execute_sell(symbol, current_price, "TIME_STOP")
                    return

            # Update days held metadata
            update_position(symbol, {"days_held": days_held})

        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    PortfolioMonitor().monitor_all_positions()
