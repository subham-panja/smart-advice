"""
Portfolio Monitor
File: scripts/portfolio_monitor.py

Responsible for:
1. End-of-day trailing stop loss updates.
2. Time-stop checks (sideways fund lock release).
3. Profit target monitoring.
"""

import logging
import pandas as pd
from datetime import datetime
from config import TRADING_OPTIONS, RISK_MANAGEMENT, SWING_PATTERNS
from database import get_open_positions, update_position, close_position
from scripts.data_fetcher import get_historical_data

logger = logging.getLogger(__name__)

class PortfolioMonitor:
    def __init__(self):
        self.time_stop_days = TRADING_OPTIONS.get('time_stop_days', 15)
        self.exit_rules = SWING_PATTERNS.get('exit_rules', {})

    def monitor_all_positions(self):
        """Monitor all open positions for exits or SL updates."""
        positions = get_open_positions()
        if not positions:
            logger.info("No open positions to monitor.")
            return

        logger.info(f"Monitoring {len(positions)} open positions...")
        
        for pos in positions:
            self._process_single_position(pos)

    def _process_single_position(self, pos):
        symbol = pos['symbol']
        try:
            # 1. Fetch latest data
            data = get_historical_data(symbol, period='1mo', fresh=True)
            if data.empty:
                logger.warning(f"Could not fetch data for {symbol}")
                return

            current_price = data['Close'].iloc[-1]
            entry_price = pos['entry_price']
            entry_date = pos['entry_date']
            
            # Calculate days held
            days_held = (datetime.now() - entry_date).days
            
            # 2. Check for Time Stop (Sideways)
            if days_held >= self.time_stop_days:
                # If profit is less than 2% after 15 days, exit
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                if pnl_pct < 2.0:
                    logger.info(f"TIME STOP: {symbol} held for {days_held} days with {pnl_pct:.2f}% PnL. Exiting.")
                    from scripts.execution_engine import ExecutionEngine
                    engine = ExecutionEngine()
                    engine.execute_sell(symbol, current_price, "TIME_STOP")
                    return

            # 3. Calculate New Trailing SL (using ATR)
            from scripts.risk_management import RiskManager
            rm = RiskManager()
            sl_info = rm.calculate_stop_loss(data, current_price, method='atr', 
                                            atr_multiplier=self.exit_rules.get('atr_stop_multiplier', 1.5))
            
            new_sl = sl_info['stop_loss']
            current_sl = pos['current_stop_loss']

            # Move SL up if price has moved in our favor (Trailing)
            # Rule: Never move SL down
            if new_sl > current_sl:
                logger.info(f"TRAILING SL: Updating {symbol} SL from {current_sl:.2f} to {new_sl:.2f}")
                update_position(symbol, {'current_stop_loss': new_sl})
                # TODO: In real execution, update VTT order on 5paisa here

            # 4. Check for Breakeven Logic (Move SL to Entry if T1 hit)
            if self.exit_rules.get('breakeven_at_target_1', False):
                target_1 = pos['current_target'] # Assuming T1 is stored as current_target
                if current_price >= target_1 and current_sl < entry_price:
                    logger.info(f"BREAKEVEN: {symbol} hit T1. Moving SL to Entry: {entry_price}")
                    update_position(symbol, {'current_stop_loss': entry_price})

            # Update days held in DB
            update_position(symbol, {'days_held': days_held})

        except Exception as e:
            logger.error(f"Error monitoring {symbol}: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = PortfolioMonitor()
    monitor.monitor_all_positions()
