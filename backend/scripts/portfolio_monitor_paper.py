"""
Portfolio Monitor
File: scripts/portfolio_monitor_paper.py

Responsible for:
1. End-of-day trailing stop loss updates.
2. Hard Target and Stop Loss exit detection.
3. Time-stop checks (sideways fund lock release).
4. Live price updates for PnL tracking.
"""

import logging
from datetime import timezone

from database import get_open_positions, update_position
from scripts.data_fetcher import get_historical_data
from utils.strategy_loader import StrategyLoader
from utils.trading_clock import trading_now

logger = logging.getLogger(__name__)


class PortfolioMonitor:
    def __init__(self):
        # time_stop_days now comes from strategy trading_config per position
        self.default_time_stop_days = 15

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
        strat_name = pos["strategy_name"]

        try:
            try:
                strategy = StrategyLoader.get_strategy_by_name(strat_name)
            except ValueError:
                logger.warning(
                    f"Strategy '{strat_name}' not found/disabled for {symbol}. Using Swing_Trading fallback."
                )
                strategy = StrategyLoader.get_strategy_by_name("Swing_Trading")

            exit_rules = strategy.get("exit_rules", {})
            trading_cfg = strategy.get("trading_config", {})
            # Canonical source: exit_rules.time_stop_bars (matches backtest).
            # Fallback: trading_config.time_stop_days (legacy). Then default 15.
            time_stop_days = exit_rules.get(
                "time_stop_bars",
                trading_cfg.get("time_stop_days", self.default_time_stop_days),
            )
            # 1. Fetch latest data (Live Price Sync)
            from utils.trading_clock import is_replay

            fetch_period = "5y" if is_replay() else "1mo"
            data = get_historical_data(symbol, period=fetch_period)
            if data is None or data.empty:
                raise ValueError(f"Could not fetch data for {symbol}")

            if is_replay():
                sim_dt = trading_now().replace(tzinfo=None)
                sim_date = sim_dt.date() if hasattr(sim_dt, "date") else sim_dt
                data = data[data.index.date <= sim_date]
                if data.empty:
                    raise ValueError(f"Could not find historical data for {symbol} on/before {sim_date}")

            current_price = round(data["Close"].iloc[-1], 2)
            entry_price = pos["entry_price"]
            entry_date = pos["entry_date"]
            current_sl = pos.get("current_stop_loss", pos["stop_loss"])

            # Correct entry price to opening price ONLY for unadjusted paper positions once
            entry_date_obj = entry_date.date() if hasattr(entry_date, "date") else entry_date
            today_date = trading_now().date()
            if (
                entry_date_obj < today_date
                and not pos.get("entry_adjusted_manually", False)
                and not pos.get("is_entry_corrected", False)
                and pos.get("is_paper", True)
            ):
                entry_row = data[data.index.date == entry_date_obj]
                if not entry_row.empty:
                    open_price = round(entry_row["Open"].iloc[0], 2)
                    if abs(open_price - entry_price) > 0.01:
                        # Recalculate risk with corrected entry price
                        corrected_investment = round(
                            open_price * pos["quantity"] * (1 + 0.0020), 2
                        )  # include brokerage
                        sl_distance = entry_price - pos.get("stop_loss", entry_price)  # original SL distance
                        adjusted_sl = round(open_price - sl_distance, 2)
                        qty = pos["quantity"]
                        new_initial_risk = round((open_price - adjusted_sl) * qty, 2)

                        logger.info(
                            f"📊 ENTRY PRICE CORRECTION: {symbol} | "
                            f"Old: ₹{entry_price:.2f} → New (Open): ₹{open_price:.2f} | "
                            f"SL adjusted: ₹{pos.get('stop_loss', 0):.2f} → ₹{adjusted_sl:.2f}"
                        )
                        update_position(
                            symbol,
                            {
                                "entry_price": open_price,
                                "total_investment": corrected_investment,
                                "stop_loss": adjusted_sl,
                                "current_stop_loss": max(adjusted_sl, current_sl),
                                "initial_risk": new_initial_risk,
                                "risk_pct_of_cap": round((new_initial_risk / 100000.0) * 100, 2),
                                "is_entry_corrected": True,
                            },
                        )
                        entry_price = open_price
                        current_sl = max(adjusted_sl, current_sl)

            # Update Live Price in DB for Telegram PnL
            update_position(symbol, {"current_price": current_price})

            from scripts.execution_engine_paper import ExecutionEngine

            engine = ExecutionEngine(strategy_config=strategy)

            # 2. Check for Hard Exit: Stop Loss Hit
            if current_price <= current_sl:
                logger.info(f"🛑 STOP LOSS HIT: {symbol} at ₹{current_price:.2f}")
                engine.execute_sell(symbol, current_price, "STOP_LOSS_HIT")
                return

            # 6. Structured Target Monitoring
            targets = exit_rules["targets"]
            import talib as ta

            atr = ta.ATR(data["High"], data["Low"], data["Close"], timeperiod=14).iloc[-1]

            current_target_idx = pos.get("current_target_idx", 0)

            if current_target_idx < len(targets):
                target_obj = targets[current_target_idx]
                target_price = entry_price + (target_obj["atr_multiplier"] * atr)

                if current_price >= target_price:
                    sell_pct = target_obj["sell_percentage"]
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
                                "targets_hit": current_target_idx + 1,
                                "is_scaled_out": True,
                            }
                            # Auto-Breakeven if enabled and this is Target 1
                            if current_target_idx == 0 and exit_rules["breakeven_at_target_1"]:
                                update_data["current_stop_loss"] = entry_price
                                logger.info(f"🛡️ SL moved to Breakeven (₹{entry_price:.2f})")

                            update_position(symbol, update_data)
                            return
                    else:
                        # Full Exit (Final Target)
                        engine.execute_sell(symbol, current_price, f"FINAL_{target_obj['name']}")
                        return

            # 7. Trailing SL is managed exclusively by ExecutionEngine.manage_exits() (Phase 1b)

            # 8. Time Stop (Sideways)
            days_held = max(0, (trading_now(timezone.utc).replace(tzinfo=None) - entry_date).days)
            if days_held >= time_stop_days:
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
                if pnl_pct < 2.0:
                    logger.info(f"⏳ TIME STOP: {symbol} held for {days_held} days. Exit due to stagnation.")
                    engine.execute_sell(symbol, current_price, "TIME_STOP")
                    return

            # Update days held metadata
            update_position(symbol, {"days_held": days_held})

        except Exception as e:
            logger.error(f"Critical error monitoring {symbol}: {e}")
            raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    PortfolioMonitor().monitor_all_positions()
