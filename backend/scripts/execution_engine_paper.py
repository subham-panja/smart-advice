"""
Execution Engine
File: scripts/execution_engine_paper.py

Handles trade execution (Buying/Selling) with Paper Trading Mock support.
Pyramiding config is now read from strategy config (not global config.py)
to ensure consistency with portfolio backtesting.
"""

import logging
from datetime import datetime, timezone

import config
from database import close_position, get_open_positions, insert_position, update_position

logger = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self, strategy_config=None):
        """Initialize execution engine.

        Args:
            strategy_config: Strategy config dict from JSON. All trading params come from here.
        """
        self.strategy_config = strategy_config

        # App-level defaults from config.py
        trading_opts = config.TRADING_OPTIONS
        self.is_paper_trading = trading_opts.get("is_paper_trading", True)
        self.brokerage_pct = trading_opts.get("brokerage_charges", 0.0020)
        self.initial_capital = trading_opts.get("initial_capital", 100000.0)
        self.circuit_breaker = trading_opts.get("circuit_breaker", False)

    def _get_pyramid_config(self):
        """Get pyramiding config from strategy config."""
        if self.strategy_config and "pyramiding" in self.strategy_config:
            return self.strategy_config["pyramiding"]
        return {"enabled": False, "steps": []}

    def _check_regime_for_pyramid(self):
        """Check if pyramiding should be allowed based on market regime."""
        if not self.strategy_config:
            return True

        pyramid_cfg = self._get_pyramid_config()
        if not pyramid_cfg.get("regime_controlled", False):
            return True

        # Import regime detector
        try:
            from scripts.market_regime_detection import MarketRegimeDetection

            detector = MarketRegimeDetection()
            regime_config = self.strategy_config.get("market_regime_config", {})
            result = detector.get_simple_regime_check(
                index=regime_config.get("index", "^NSEI"),
                rule=regime_config.get("bull_market_rule", "latest close > sma(200)"),
            )
            # In bear regime, check if pyramiding is allowed
            regime_adaptive = self.strategy_config.get("exit_rules", {}).get("regime_adaptive_exits", {})
            regime = result.get("status", "BULL").lower()
            if regime in regime_adaptive:
                return regime_adaptive[regime].get("pyramiding_allowed", True)
            return True
        except Exception as e:
            logger.warning(f"Regime check for pyramiding failed: {e}")
            return True  # Default to allowing pyramiding if regime check fails

    def execute_buy(self, symbol, quantity, price, stop_loss, target, recomm_id=None, strategy_name="UNKNOWN"):
        """Execute a buy order with detailed metadata and pyramiding support."""
        if self.circuit_breaker:
            logger.warning(f"🛑 CIRCUIT BREAKER: Buy order for {symbol} blocked.")
            return None

        mode = "PAPER" if self.is_paper_trading else "LIVE"
        if self.is_paper_trading:
            logger.info(f"{mode} TRADING: Executing BUY for {symbol} | Qty: {quantity} | Price: {price}")

            open_positions = get_open_positions()
            existing_pos = next((p for p in open_positions if p["symbol"] == symbol), None)

            # Read pyramiding config from strategy (not global config.py)
            pyramid_cfg = self._get_pyramid_config()
            can_pyramid = pyramid_cfg.get("enabled", False)

            if existing_pos:
                if not can_pyramid:
                    logger.warning(f"Skipping {symbol}: Position already exists and pyramiding disabled.")
                    return None

                # Check regime-controlled pyramiding
                if not self._check_regime_for_pyramid():
                    logger.info(f"Skipping {symbol}: Pyramiding disabled in current market regime")
                    return None

                adds = existing_pos.get("adds_count", 0)
                pyramid_steps = pyramid_cfg.get("steps", [])

                if adds >= len(pyramid_steps):
                    logger.warning(f"Skipping {symbol}: Max pyramid steps ({len(pyramid_steps)}) reached.")
                    return None

                step_obj = pyramid_steps[adds]
                trigger_atr_mult = step_obj.get("trigger_step_atr", 1.5)

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
                add_pct = step_obj.get("add_size_pct", 0.5)
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

    def manage_exits(self):
        """Scan all open positions and apply exit logic: stops, targets, time stops, leader exceptions.

        Mirrors the exit logic from portfolio_backtest_engine._process_exits().
        Reads regime_adaptive_exits, leader_exception, trail_stop configs from strategy JSON.
        """
        if self.circuit_breaker:
            logger.warning("🛑 CIRCUIT BREAKER: Exit management blocked.")
            return

        exit_cfg = self.strategy_config.get("exit_rules", {}) if self.strategy_config else {}
        regime_adaptive = exit_cfg.get("regime_adaptive_exits", {})
        leader_cfg = exit_cfg.get("leader_exception", {})
        targets = exit_cfg.get("targets", [])

        if not targets:
            logger.warning("No exit targets configured — skipping exit management.")
            return

        # Determine current regime
        regime = self._detect_regime()
        regime_params = regime_adaptive.get(regime, {})

        import talib as ta

        from scripts.data_fetcher import get_historical_data

        open_positions = get_open_positions()
        for pos in open_positions:
            symbol = pos["symbol"]
            entry_price = pos["entry_price"]
            current_stop_loss = pos.get("current_stop_loss", pos.get("stop_loss"))
            entry_date = pos.get("entry_date")
            quantity = pos["quantity"]
            targets_hit = pos.get("targets_hit", 0)

            # Calculate days/weeks held
            if entry_date:
                days_held = (datetime.now(timezone.utc).replace(tzinfo=None) - entry_date).days
            else:
                days_held = 0
            weeks_held = days_held / 7.0

            # Fetch current price and ATR
            try:
                hist = get_historical_data(symbol, period="60d")
                if hist is None or hist.empty or len(hist) < 14:
                    continue
                current_price = hist["Close"].iloc[-1]
                atr = ta.ATR(hist["High"], hist["Low"], hist["Close"], timeperiod=14).iloc[-1]
            except Exception as e:
                logger.error(f"Failed to fetch data for exit check {symbol}: {e}")
                continue

            gain_pct = (current_price - entry_price) / entry_price * 100

            # --- 0. O'Neil Fixed Stop Loss ---
            stop_loss_pct = regime_params.get("stop_loss_pct", None)
            if stop_loss_pct:
                oneil_stop = entry_price * (1 - stop_loss_pct / 100.0)
                if current_price <= oneil_stop:
                    self.execute_sell(symbol, current_price, f"ONEIL_STOP_{stop_loss_pct}%")
                    continue

            # --- 1. O'Neil Profit Target + Leader Exception ---
            oneil_target_pct = regime_params.get("oneil_target_pct", 25.0)
            is_leader = False
            if leader_cfg.get("enabled", False) and gain_pct >= leader_cfg.get("min_gain_pct", 20.0):
                if weeks_held <= leader_cfg.get("max_weeks", 8):
                    is_leader = True

            if gain_pct >= oneil_target_pct and not is_leader:
                self.execute_sell(symbol, current_price, f"ONEIL_TARGET_{oneil_target_pct:.0f}%")
                continue

            # 8-week leader: hold and trail, skip normal targets/time stops
            if is_leader and leader_cfg.get("action") == "hold_and_trail":
                logger.debug(f"🏆 LEADER: {symbol} | Gain: {gain_pct:.1f}% in {weeks_held:.1f}w | Holding & trailing")
            else:
                # --- 2. Time Stop ---
                time_stop = regime_params.get("time_stop_bars", exit_cfg.get("time_stop_bars", 20))
                if days_held >= time_stop:
                    self.execute_sell(symbol, current_price, "TIME_STOP")
                    continue

                # --- 3. ATR Targets ---
                if targets_hit < len(targets):
                    target_cfg = targets[targets_hit].copy()
                    # Regime-adaptive T1 sell percentage
                    if targets_hit == 0:
                        t1_pct = regime_params.get("t1_sell_percentage", target_cfg["sell_percentage"])
                        target_cfg["sell_percentage"] = t1_pct

                    target_price = entry_price + (target_cfg["atr_multiplier"] * atr)

                    if current_price >= target_price:
                        sell_pct = target_cfg["sell_percentage"]
                        sell_qty = int(quantity * sell_pct)

                        if sell_qty > 0 and sell_pct < 1.0:
                            # Partial sell
                            self.execute_sell(symbol, current_price, f"{target_cfg['name']}", quantity=sell_qty)
                            # Update position: reduce quantity, increment targets_hit
                            new_qty = quantity - sell_qty
                            update_data = {
                                "quantity": new_qty,
                                "targets_hit": targets_hit + 1,
                                "is_scaled_out": True,
                            }
                            if targets_hit == 0 and exit_cfg.get("breakeven_at_target_1"):
                                update_data["current_stop_loss"] = entry_price
                                logger.info(f"🛡️ {symbol}: SL moved to breakeven ₹{entry_price:.2f}")
                            update_position(symbol, update_data)
                        elif sell_pct >= 1.0:
                            # Full exit
                            self.execute_sell(symbol, current_price, f"FINAL_{target_cfg['name']}")
                        continue

            # --- 4. Trailing Stop Update ---
            trail_type = regime_params.get("trail_stop_type", "atr")

            if trail_type == "ma" and len(hist) >= regime_params.get("trail_stop_ma_period", 20):
                ma_period = regime_params.get("trail_stop_ma_period", 20)
                ma_value = hist["Close"].rolling(ma_period).mean().iloc[-1]
                if ma_value > current_stop_loss:
                    logger.info(
                        f"📉 {symbol}: MA{ma_period} trail SL updated ₹{current_stop_loss:.2f} → ₹{ma_value:.2f}"
                    )
                    update_position(symbol, {"current_stop_loss": round(ma_value, 2)})
            elif atr > 0:
                trail_mult = regime_params.get("trail_stop_atr_multiplier", exit_cfg.get("trail_stop_atr", 2.0))
                new_sl = current_price - (atr * trail_mult)
                if new_sl > current_stop_loss:
                    logger.info(f"📉 {symbol}: Trailing SL updated ₹{current_stop_loss:.2f} → ₹{new_sl:.2f}")
                    update_position(symbol, {"current_stop_loss": round(new_sl, 2)})

        logger.info(f"Exit management complete. Checked {len(open_positions)} open positions.")

    def _detect_regime(self):
        """Detect current market regime (BULL/BEAR) using strategy config.

        Returns lowercase 'bull' or 'bear'. Raises RuntimeError if regime detection is disabled or fails.
        """
        if not self.strategy_config:
            raise RuntimeError(
                "Market regime detection is DISABLED but required by this strategy. "
                "Enable market_regime_detection in hybrid_trading.json or provide strategy_config to ExecutionEngine."
            )

        regime_config = self.strategy_config.get("market_regime_config", {})
        if not regime_config:
            raise RuntimeError(
                "Market regime detection is DISABLED but required by this strategy. "
                "Ensure 'market_regime_config' is present in hybrid_trading.json with 'index' and 'bull_market_rule'."
            )

        try:
            from scripts.market_regime_detection import MarketRegimeDetection

            detector = MarketRegimeDetection()
            result = detector.get_simple_regime_check(
                index=regime_config.get("index", "^NSEI"),
                rule=regime_config.get("bull_market_rule", "latest close > sma(200)"),
            )
            return result.get("status", "BULL").lower()
        except Exception as e:
            raise RuntimeError(
                f"Market regime detection failed: {e}. "
                "Check your market_regime_config in hybrid_trading.json or verify index data availability."
            ) from e

    def execute_sell(self, symbol, price, reason, quantity=None):
        """Execute a sell order (Supports Partial or Full exits)."""
        if self.circuit_breaker:
            logger.warning(f"🛑 CIRCUIT BREAKER: Sell order for {symbol} blocked.")
            return None

        if self.is_paper_trading:
            qty_str = f"Qty: {quantity}" if quantity else "FULL Position"
            logger.info(f"PAPER TRADING: Executing SELL for {symbol} | {qty_str} | Price: {price} | Reason: {reason}")

            try:
                if quantity:
                    # Partial Sell — update position quantity
                    open_positions = get_open_positions()
                    pos = next((p for p in open_positions if p["symbol"] == symbol), None)
                    if pos:
                        current_qty = pos["quantity"]
                        new_qty = current_qty - quantity
                        if new_qty <= 0:
                            # Should have been a full exit
                            close_position(symbol, price, reason)
                            logger.info(f"✅ FULL EXIT: {symbol} | Price: {price} | Reason: {reason}")
                        else:
                            update_position(
                                symbol,
                                {
                                    "quantity": new_qty,
                                    "partial_exits": pos.get("partial_exits", [])
                                    + [
                                        {
                                            "date": datetime.now(timezone.utc).replace(tzinfo=None),
                                            "quantity": quantity,
                                            "price": price,
                                            "reason": reason,
                                        }
                                    ],
                                },
                            )
                            logger.info(
                                f"✅ PARTIAL SELL: {symbol} | Sold {quantity} of {current_qty} @ {price} | Reason: {reason}"
                            )
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
            logger.warning("Real trading execution not supported in paper engine.")
            raise NotImplementedError("Live trading not supported in paper engine.")
