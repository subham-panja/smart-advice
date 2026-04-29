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
from datetime import datetime

from config import SWING_PATTERNS, TRADING_OPTIONS
from database import get_open_positions, update_position
from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    def __init__(self):
        self.time_stop_days = TRADING_OPTIONS.get("time_stop_days", 15)
        self.exit_rules = SWING_PATTERNS.get("exit_rules", {})

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

            # 3. Check for Hard Exit: Target Hit
            if target and current_price >= target:
                logger.info(f"🎯 TARGET HIT: {symbol} at ₹{current_price:.2f}")
                engine.execute_sell(symbol, current_price, "TARGET_HIT")
                return

            # 4. Check for Time Stop (Sideways)
            days_held = (datetime.now() - entry_date).days
            if days_held >= self.time_stop_days:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                if pnl_pct < 2.0:
                    logger.info(f"⏳ TIME STOP: {symbol} held for {days_held} days. Exit due to stagnation.")
                    engine.execute_sell(symbol, current_price, "TIME_STOP")
                    return

            # 5. VTT-Style Trailing SL Update
            from scripts.risk_management import RiskManager

            rm = RiskManager()
            sl_info = rm.calculate_stop_loss(
                data, current_price, method="atr", atr_multiplier=self.exit_rules.get("atr_stop_multiplier", 1.5)
            )

            new_sl = sl_info["stop_loss"]
            if new_sl > current_sl:
                logger.info(f"📉 VTT UPDATE: Trailing {symbol} SL from {current_sl:.2f} to {new_sl:.2f}")
                update_position(symbol, {"current_stop_loss": new_sl})
                current_sl = new_sl  # Update local variable for next check

            # 6. Breakeven Protection
            if self.exit_rules.get("breakeven_at_target_1", True) and target:
                half_target = entry_price + (target - entry_price) * 0.5
                if current_price >= half_target and current_sl < entry_price:
                    logger.info(f"🛡️ BREAKEVEN: {symbol} reached 50% of target. Moving SL to Entry.")
                    update_position(symbol, {"current_stop_loss": entry_price})

            # Update days held metadata
            update_position(symbol, {"days_held": days_held})

        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    PortfolioMonitor().monitor_all_positions()
