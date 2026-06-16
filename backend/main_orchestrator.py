import argparse
import logging
from datetime import datetime, timedelta, timezone

import requests

import config
from database import get_mongodb, get_open_positions
from run_analysis import AutomatedStockAnalysis
from utils.logger import setup_logging
from utils.persistence_handler import PersistenceHandler
from utils.strategy_loader import StrategyLoader
from utils.trading_clock import is_replay, set_simulated_date, trading_now

setup_logging(verbose=True)
logger = logging.getLogger("Orchestrator")


def _send_telegram(message: str):
    if is_replay():
        return
    tg = getattr(config, "TELEGRAM_CONFIG", {})
    if not tg.get("enabled", False):
        logger.warning("Telegram: not enabled")
        return
    token = tg.get("bot_token", "")
    chat_ids = tg.get("allowed_user_ids", [])
    if not token or not chat_ids:
        logger.warning("Telegram: missing token or chat_ids")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram message sent (chat_id=%s)" % chat_id)
            else:
                logger.error("Telegram API error: %s %s" % (resp.status_code, resp.text))
        except Exception as e:
            logger.error("Telegram send failed: %s" % e)


def run_trading_cycle():
    """Main entry point for the unified trading cycle."""
    trading_opts = config.TRADING_OPTIONS

    if is_replay():
        sim = trading_now().strftime("%Y-%m-%d")
    elif trading_opts.get("is_paper_trading", True):
        sim = "PAPER"
    else:
        sim = "LIVE"

    print("")
    print("=" * 50)
    print(f"STARTING TRADING CYCLE [{sim}]")
    print("=" * 50)
    logger.warning(f"=== STARTING TRADING CYCLE [{sim}] ===")

    _send_telegram(f"🔄 <b>Trading Cycle Started [{sim}]</b>\nRunning analysis and execution...")

    if trading_opts.get("circuit_breaker"):
        logger.warning("CIRCUIT BREAKER ACTIVE. Stopping.")
        print("CIRCUIT BREAKER ACTIVE. Stopping.")
        return {"executed": 0, "exits": 0, "positions": 0, "equity": trading_opts.get("initial_capital", 100000.0)}

    PersistenceHandler().clear_old_data(config.DATA_PURGE_DAYS)

    all_strategies = StrategyLoader.load_all_strategies()
    if not all_strategies:
        raise RuntimeError("No enabled strategies found.")

    logger.info("Running trading cycle for %d enabled strategy(ies)" % len(all_strategies))

    is_paper = trading_opts.get("is_paper_trading", True)
    if is_paper:
        from scripts.execution_engine_paper import ExecutionEngine
        from scripts.portfolio_monitor_paper import PortfolioMonitor
    else:
        from scripts.execution_engine import ExecutionEngine
        from scripts.portfolio_monitor import PortfolioMonitor

    total_executed = 0
    total_exits = 0
    analysis_failed = False

    positions_before = get_open_positions()

    for pos in positions_before:
        if pos.get("quantity", 0) <= 0:
            from database import close_position as db_close

            cp = pos.get("current_price", pos["entry_price"])
            db_close(pos["symbol"], cp, "ZERO_QUANTITY_CLEANUP")
            logger.info(f"Auto-closed {pos['symbol']}: quantity was 0")

    positions_before = get_open_positions()
    symbols_before = {p["symbol"] for p in positions_before}

    for strategy in all_strategies:
        strat_name = strategy["name"]

        print(f"\nProcessing Strategy: {strat_name}")
        logger.info("Processing Strategy: " + strat_name)

        # Phase 1: Monitor
        print("Phase 1: Monitoring existing positions...")
        logger.info("Phase 1: Monitoring existing positions...")
        PortfolioMonitor().monitor_all_positions()

        # Phase 1b: Advanced exits
        exit_engine = ExecutionEngine(strategy_config=strategy)
        exit_engine.manage_exits()

        # Phase 2: Analysis (use all NSE symbols in replay mode for per-date local filtering)
        print(f"Phase 2: Running analysis for {strat_name}...")
        logger.info(f"Phase 2: Running analysis for {strat_name}...")
        try:
            analyzer = AutomatedStockAnalysis(verbose=True)
            analyzer.run(strategy_config=strategy, use_all_symbols=is_replay())
            if not analyzer.scanned_symbols_count:
                logger.warning("Analysis scan produced 0 candidates for %s" % strat_name)
                analysis_failed = True
        except Exception as e:
            logger.error("Analysis failed for %s: %s" % (strat_name, e))
            analysis_failed = True

        # Phase 3: Execute
        print(f"Phase 3: Executing recommendations for {strat_name}...")
        logger.info(f"Phase 3: Executing recommendations for {strat_name}...")

        engine = ExecutionEngine(strategy_config=strategy)
        db = get_mongodb()
        today_start = trading_now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).replace(tzinfo=None)

        open_positions = get_open_positions()

        risk_cfg = strategy.get("risk_management", {})
        max_pos = risk_cfg.get("max_positions", 15)

        if len(open_positions) >= max_pos:
            logger.warning("Portfolio Full: %d/%d. Skipping %s" % (len(open_positions), max_pos, strat_name))
            print(f"   Portfolio Full: {len(open_positions)}/{max_pos}")
            continue

        recs = list(
            db.recommended_shares.find(
                {
                    "recommendation_date": {"$gte": today_start},
                    "strategy_name": strat_name,
                    "recommendation_strength": "BUY",
                }
            )
        )

        if not recs:
            logger.info(f"No new recommendations for {strat_name}")
            print(f"   No new recommendations for {strat_name}")
            continue

        if not engine._check_market_breadth_paper(strategy):
            logger.warning(f"MARKET BREADTH WEAK: Skipping new buys for {strat_name}")
            print(f"   Market Breadth Weak: Skipping new buys for {strat_name}")
            continue

        slots_left = max_pos - len(open_positions)
        executed_count = 0

        total_invested = sum(p.get("total_investment", 0) for p in open_positions)
        initial_capital = trading_opts.get("initial_capital", 100000.0)
        remaining_capital = initial_capital - total_invested

        for r in recs:
            if executed_count >= slots_left:
                break

            symbol = r["symbol"]

            if r.get("suggested_quantity", 0) <= 0:
                logger.info(f"Skipping {symbol}: suggested_quantity is 0")
                continue

            trade_cost = r["buy_price"] * r["suggested_quantity"]

            if trade_cost > remaining_capital:
                logger.warning(
                    f"Insufficient capital for {symbol}: need ₹{trade_cost:.2f}, have ₹{remaining_capital:.2f}"
                )
                continue

            if trading_opts.get("auto_execute", True) or is_paper:
                success = engine.execute_buy(
                    symbol,
                    quantity=r["suggested_quantity"],
                    price=r["buy_price"],
                    stop_loss=r["stop_loss"],
                    target=r["sell_price"],
                    recomm_id=r["_id"],
                    strategy_name=strat_name,
                )
                if success:
                    executed_count += 1
                    remaining_capital -= trade_cost

        print(f"   Executed {executed_count} trades for {strat_name}")
        total_executed += executed_count

    # Detect exits that happened this cycle
    positions_after = get_open_positions()
    symbols_after = {p["symbol"] for p in positions_after}
    closed_symbols = symbols_before - symbols_after
    new_symbols = symbols_after - symbols_before
    total_exits = len(closed_symbols)

    # Compute final state
    final_positions = positions_after
    initial_cap = trading_opts.get("initial_capital", 100000.0)
    total_mkt_val = sum(p.get("current_price", p["entry_price"]) * p["quantity"] for p in final_positions)
    total_invested = sum(p.get("total_investment", p["quantity"] * p["entry_price"]) for p in final_positions)

    realized_pnl = sum(
        (p.get("exit_price", 0) - p.get("entry_price", 0)) * p.get("quantity", 0)
        for p in db.positions.find({"status": "CLOSED"})
    )

    cash_left = initial_cap + realized_pnl - total_invested
    total_equity = total_mkt_val + cash_left

    cycle_stats = {
        "date": sim,
        "executed": total_executed,
        "exits": total_exits,
        "closed": list(closed_symbols),
        "opened": list(new_symbols),
        "positions": len(final_positions),
        "equity": total_equity,
        "cash": cash_left,
        "pnl_pct": ((total_equity - initial_cap) / initial_cap) * 100,
    }

    # Send Telegram (skipped automatically in replay mode)
    if not analysis_failed:
        _build_and_send_telegram_summary(final_positions, total_executed, initial_cap, is_paper)
    else:
        _send_telegram("⚠️ <b>Trading Cycle — Scan Failed</b>\nScreener API was unreachable.")

    print(f"\n{'=' * 50}")
    print(
        f"TRADING CYCLE COMPLETE [{sim}] | Executed: {total_executed} | Exits: {total_exits} | Equity: ₹{total_equity:,.2f} ({cycle_stats['pnl_pct']:+.2f}%)"
    )
    print(f"{'=' * 50}\n")

    return cycle_stats


def _build_and_send_telegram_summary(open_positions, total_executed, initial_cap, is_paper):
    """Build and send the Telegram summary message."""
    total_mkt_val = 0
    total_pnl_val = 0
    positions_msg = ""
    sim_date = trading_now()

    for p in open_positions:
        current_p = p.get("current_price", p["entry_price"])
        pnl_val = (current_p - p["entry_price"]) * p["quantity"]
        pnl_pct = ((current_p - p["entry_price"]) / p["entry_price"]) * 100
        total_cost = p.get("total_investment", p["quantity"] * p["entry_price"])
        total_mkt_val += current_p * p["quantity"]
        total_pnl_val += pnl_val
        status_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        allocation = p.get("allocation_pct", (total_cost / initial_cap) * 100)

        active_sl = p.get("current_stop_loss", p.get("stop_loss", 0))
        original_sl = p.get("stop_loss", 0)
        sl_trailed = active_sl > original_sl
        active_target = p.get("current_target", p.get("target", 0))

        sl_line = f"🛑 SL: ₹{active_sl:.2f}"
        if sl_trailed:
            sl_line += f" <i>(trailed from ₹{original_sl:.2f})</i>"

        positions_msg += (
            f"{status_emoji} <b>{p['symbol']}</b> | {p.get('strategy_name', 'Swing_Trading')}\n"
            f"📅 Entered: {p['entry_date'].strftime('%Y-%m-%d') if p.get('entry_date') else 'N/A'}\n"
            f"🔢 Qty: {p['quantity']} @ ₹{p['entry_price']:.2f}\n"
            f"💰 Cost: ₹{total_cost:,.2f} ({allocation:.1f}% Cap)\n"
            f"🎯 Target: ₹{active_target:.2f} | {sl_line}\n"
            f"💸 PnL: ₹{pnl_val:+,.2f} ({pnl_pct:+.2f}%)\n"
        )

        adds = p.get("adds_count", 0)
        if adds > 0:
            positions_msg += f"🔺 Pyramid: {adds} add(s)\n"

        partial_exits = p.get("partial_exits", [])
        for pe in partial_exits:
            positions_msg += f"📤 Sold {pe.get('quantity',0)} @ ₹{pe.get('price',0):.2f} ({pe.get('reason','')})\n"

        updates = p.get("updates", [])
        today_updates = [u for u in updates if u.get("date") and u["date"].date() == sim_date.date()]
        for u in today_updates:
            utype = u.get("type", "")
            if utype == "TRAIL_SL":
                positions_msg += f"📋 Trail: ₹{u.get('prev_sl',0):.2f} → ₹{u.get('current_sl',0):.2f}\n"
            elif utype == "PYRAMID":
                positions_msg += "📋 Pyramid add\n"
            elif utype == "TARGET_HIT":
                positions_msg += "📋 Target hit\n"
            elif utype == "PARTIAL_SELL":
                positions_msg += "📋 Partial sell\n"

        positions_msg += "\n"

    cash_left = initial_cap - sum(p.get("total_investment", p["quantity"] * p["entry_price"]) for p in open_positions)
    total_equity = total_mkt_val + cash_left
    overall_pnl_pct = ((total_equity - initial_cap) / initial_cap) * 100
    header = "📝 Paper Positions" if is_paper else "💼 Live Positions"

    summary = (
        f"✅ <b>Trading Cycle Complete!</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{header} ({len(open_positions)}):\n"
        f"{'━━━━━━━━━━━━━━━━━━━━\n' + positions_msg if positions_msg else '📭 No open positions.\n\n'}"
        f"📊 <b>PORTFOLIO SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 Initial Capital: ₹{initial_cap:,.2f}\n"
        f"📈 Market Value: ₹{total_mkt_val:,.2f}\n"
        f"🏦 Cash Balance: ₹{cash_left:,.2f}\n"
        f"💰 Net Equity: ₹{total_equity:,.2f}\n"
        f"📊 Total PnL: ₹{total_pnl_val:+,.2f} ({overall_pnl_pct:+.2f}%)\n\n"
        f"💰 Trades Executed: {total_executed}"
    )
    _send_telegram(summary)


def _send_replay_summary(all_stats, initial_cap):
    """Send a detailed final summary after a multi-day replay."""
    if not all_stats:
        return

    first_date = all_stats[0]["date"]
    last_date = all_stats[-1]["date"]
    total_days = len(all_stats)
    total_buys = sum(s["executed"] for s in all_stats)
    total_exits = sum(s["exits"] for s in all_stats)
    final_equity = all_stats[-1]["equity"]
    final_pnl_pct = all_stats[-1]["pnl_pct"]
    peak_equity = max(s["equity"] for s in all_stats)
    trough_equity = min(s["equity"] for s in all_stats)
    max_drawdown = ((trough_equity - peak_equity) / peak_equity) * 100 if peak_equity > 0 else 0

    db = get_mongodb()

    # ── Message 1: Performance + Open Positions with full detail ──
    open_positions = get_open_positions()
    pos_msg = ""
    for p in open_positions:
        current_p = p.get("current_price", p["entry_price"])
        pnl_val = (current_p - p["entry_price"]) * p["quantity"]
        pnl_pct = ((current_p - p["entry_price"]) / p["entry_price"]) * 100
        total_cost = p.get("total_investment", p["quantity"] * p["entry_price"])
        status_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        allocation = p.get("allocation_pct", (total_cost / initial_cap) * 100)

        active_sl = p.get("current_stop_loss", p.get("stop_loss", 0))
        original_sl = p.get("stop_loss", 0)
        sl_trailed = active_sl > original_sl
        active_target = p.get("current_target", p.get("target", 0))

        sl_line = f"🛑 SL: ₹{active_sl:.2f}"
        if sl_trailed:
            sl_line += f" <i>(from ₹{original_sl:.2f})</i>"

        pos_msg += (
            f"{status_emoji} <b>{p['symbol']}</b> | {p.get('strategy_name', 'Swing_Trading')}\n"
            f"  📅 {p['entry_date'].strftime('%Y-%m-%d') if p.get('entry_date') else 'N/A'}"
            f" | Qty: {p['quantity']} @ ₹{p['entry_price']:.2f}"
            f" | ₹{total_cost:,.0f} ({allocation:.1f}%)\n"
            f"  🎯 ₹{active_target:.2f} | {sl_line}\n"
            f"  💸 PnL: ₹{pnl_val:+,.2f} ({pnl_pct:+.2f}%)\n"
        )

        # Pyramiding detail
        adds = p.get("adds_count", 0)
        if adds > 0:
            last_add = p.get("last_add_price", 0)
            pos_msg += f"  🔺 Pyramid: {adds} add(s) | last @ ₹{last_add:.2f}\n"

        # Partial exits detail
        for pe in p.get("partial_exits", []):
            pos_msg += (
                f"  📤 Sold {pe.get('quantity',0)} @ ₹{pe.get('price',0):.2f}" f" ({pe.get('reason','partial')})\n"
            )

        # Full update history for this position
        updates = p.get("updates", [])
        if updates:
            trail_count = sum(1 for u in updates if u.get("type") == "TRAIL_SL")
            if trail_count > 0:
                first_sl = updates[0].get("current_sl", original_sl)
                last_sl = updates[-1].get("current_sl", active_sl)
                pos_msg += f"  📋 {trail_count} SL trails: ₹{first_sl:.2f} → ₹{last_sl:.2f}\n"
            pyramid_count = sum(1 for u in updates if u.get("type") == "PYRAMID")
            if pyramid_count > 0:
                pos_msg += f"  📋 {pyramid_count} pyramid add(s)\n"
            partial_count = sum(1 for u in updates if u.get("type") in ("PARTIAL_SELL", "TARGET_HIT"))
            if partial_count > 0:
                pos_msg += f"  📋 {partial_count} partial sell(s)\n"

        pos_msg += "\n"

    # Count update types across all positions
    all_positions = list(db.positions.find({}, {"updates": 1}))
    update_counts = {}
    for p in all_positions:
        for u in p.get("updates", []):
            utype = u.get("type", "UNKNOWN")
            update_counts[utype] = update_counts.get(utype, 0) + 1
    update_summary = " | ".join(f"{k}: {v}" for k, v in sorted(update_counts.items())) if update_counts else "None"

    msg1 = (
        f"📊 <b>REPLAY REPORT ({total_days} days)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"📅 {first_date} → {last_date}\n\n"
        f"💰 <b>Performance</b>\n"
        f"💵 Capital: ₹{initial_cap:,.0f}\n"
        f"💰 Equity: ₹{final_equity:,.2f} (<b>{final_pnl_pct:+.2f}%</b>)\n"
        f"📈 Peak: ₹{peak_equity:,.0f}\n"
        f"📉 Max DD: {max_drawdown:+.1f}%\n\n"
        f"📈 <b>Trade Stats</b>\n"
        f"🔢 Buys: {total_buys} | Exits: {total_exits}\n"
        f"📋 Updates: {update_summary}\n\n"
        f"📝 <b>Open Positions ({len(open_positions)})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"{pos_msg if pos_msg else '📭 No open positions.\n'}"
    )

    # ── Message 2: Closed Trades with full lifecycle ──
    closed_trades = list(
        db.positions.find(
            {"status": "CLOSED", "exit_reason": {"$exists": True}},
            {
                "symbol": 1,
                "entry_price": 1,
                "exit_price": 1,
                "pnl_pct": 1,
                "exit_reason": 1,
                "entry_date": 1,
                "exit_date": 1,
                "quantity": 1,
                "stop_loss": 1,
                "target": 1,
                "updates": 1,
                "partial_exits": 1,
                "adds_count": 1,
                "strategy_name": 1,
            },
        ).sort("exit_date", 1)
    )

    wins = [t for t in closed_trades if t.get("pnl_pct", 0) > 0]
    losses = [t for t in closed_trades if t.get("pnl_pct", 0) <= 0]
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
    avg_win = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

    trades_msg = ""
    for t in closed_trades[-15:]:
        emoji = "🟢" if t.get("pnl_pct", 0) > 0 else "🔴"
        entry_dt = t["entry_date"].strftime("%m/%d") if t.get("entry_date") else "?"
        exit_dt = t["exit_date"].strftime("%m/%d") if t.get("exit_date") else "?"

        trades_msg += (
            f"{emoji} <b>{t['symbol']}</b> | {t.get('pnl_pct', 0):+.1f}%\n"
            f"  {entry_dt}→{exit_dt} | Qty: {t.get('quantity',0)}"
            f" @ ₹{t.get('entry_price',0):.2f} → ₹{t.get('exit_price',0):.2f}\n"
            f"  Exit: {t.get('exit_reason', '?')}\n"
        )

        # Show lifecycle events from updates
        updates = t.get("updates", [])
        lifecycle = []
        for u in updates:
            utype = u.get("type", "")
            if utype == "TARGET_HIT":
                lifecycle.append(f"T{u.get('targets_hit','?')} hit")
            elif utype == "PARTIAL_SELL":
                lifecycle.append(f"Sold {u.get('quantity','?')}")
            elif utype == "TRAIL_SL":
                lifecycle.append(f"SL→₹{u.get('current_sl',0):.2f}")
            elif utype == "PYRAMID":
                lifecycle.append(f"Pyramid +{u.get('quantity','?')}")
            elif utype == "CLOSED":
                pass
        if lifecycle:
            trades_msg += f"  📋 {' | '.join(lifecycle)}\n"

        adds = t.get("adds_count", 0)
        if adds > 0:
            trades_msg += f"  🔺 {adds} pyramid add(s)\n"

        for pe in t.get("partial_exits", []):
            trades_msg += f"  📤 Sold {pe.get('quantity',0)} @ ₹{pe.get('price',0):.2f}" f" ({pe.get('reason','')})\n"

        trades_msg += "\n"

    if len(closed_trades) > 15:
        trades_msg = f"<i>(last 15 of {len(closed_trades)} trades)</i>\n" + trades_msg

    msg2 = (
        f"📋 <b>Closed Trades ({len(closed_trades)})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Wins: {len(wins)} (avg {avg_win:+.1f}%)"
        f" | ❌ Losses: {len(losses)} (avg {avg_loss:+.1f}%)\n"
        f"🎯 Win Rate: <b>{win_rate:.1f}%</b>\n\n"
        f"{trades_msg if trades_msg else 'No closed trades.\n'}"
    )

    # ── Message 3: Daily Equity Curve ──
    equity_lines = []
    for s in all_stats:
        bar = "🟢" if s["pnl_pct"] >= 0 else "🔴"
        events = []
        if s["opened"]:
            events.append(f"+{len(s['opened'])} buy")
        if s["closed"]:
            events.append(f"-{len(s['closed'])} exit")
        event_str = " | " + ", ".join(events) if events else ""
        equity_lines.append(
            f"{bar} {s['date']} | ₹{s['equity']:,.0f}" f" ({s['pnl_pct']:+.1f}%) | {s['positions']}pos{event_str}"
        )

    msg3 = "📈 <b>Daily Equity Curve</b>\n" "━━━━━━━━━━━━━━━━━━━━\n" + "\n".join(equity_lines)

    # Send all 3 messages, splitting further if needed
    for msg in [msg1, msg2, msg3]:
        if len(msg) > 4000:
            chunk = msg[:4000]
            _send_telegram_real(chunk)
            remainder = msg[4000:]
            if remainder.strip():
                _send_telegram_real(remainder[:4000])
        else:
            _send_telegram_real(msg)


def _send_telegram_real(message: str):
    """Send telegram message even during replay (for final summary only)."""
    tg = getattr(config, "TELEGRAM_CONFIG", {})
    if not tg.get("enabled", False):
        return
    token = tg.get("bot_token", "")
    chat_ids = tg.get("allowed_user_ids", [])
    if not token or not chat_ids:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            resp = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram message sent (chat_id=%s)" % chat_id)
            else:
                logger.error("Telegram API error: %s %s" % (resp.status_code, resp.text))
        except Exception as e:
            logger.error("Telegram send failed: %s" % e)


def run_replay(num_days: int):
    """Run the trading cycle for the last N calendar days (skipping weekends)."""
    today = datetime.now().date()
    replay_dates = []
    d = today - timedelta(days=1)
    while len(replay_dates) < num_days:
        if d.weekday() < 5:  # Skip Saturday(5) and Sunday(6)
            replay_dates.append(d)
        d -= timedelta(days=1)
    replay_dates.reverse()

    initial_cap = config.TRADING_OPTIONS.get("initial_capital", 100000.0)

    print(f"\n{'=' * 60}")
    print(f"REPLAY MODE: {num_days} trading days ({replay_dates[0]} → {replay_dates[-1]})")
    print(f"{'=' * 60}\n")

    all_stats = []
    for i, day in enumerate(replay_dates):
        sim_dt = datetime(day.year, day.month, day.day, 15, 30, 0, tzinfo=timezone.utc)
        set_simulated_date(sim_dt)

        print(f"\n{'─' * 40}")
        print(f"[{i+1}/{num_days}] Replaying {day.strftime('%Y-%m-%d (%a)')}")
        print(f"{'─' * 40}")

        try:
            stats = run_trading_cycle()
            stats["date"] = day.strftime("%Y-%m-%d")
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Replay failed for {day}: {e}")
            all_stats.append(
                {
                    "date": day.strftime("%Y-%m-%d"),
                    "executed": 0,
                    "exits": 0,
                    "closed": [],
                    "opened": [],
                    "positions": 0,
                    "equity": all_stats[-1]["equity"] if all_stats else initial_cap,
                    "cash": 0,
                    "pnl_pct": 0,
                }
            )

    # Reset simulated date
    set_simulated_date(None)

    # Print final summary to console
    final = all_stats[-1] if all_stats else {}
    print(f"\n{'=' * 60}")
    print("REPLAY COMPLETE")
    print(f"Period: {replay_dates[0]} → {replay_dates[-1]} ({num_days} days)")
    print(f"Final Equity: ₹{final.get('equity', 0):,.2f} ({final.get('pnl_pct', 0):+.2f}%)")
    print(f"Total Buys: {sum(s['executed'] for s in all_stats)}")
    print(f"Total Exits: {sum(s['exits'] for s in all_stats)}")
    print(f"{'=' * 60}\n")

    # Send Telegram summary
    _send_replay_summary(all_stats, initial_cap)

    return all_stats


def run_single_date(date_str: str):
    """Run the trading cycle for a specific date."""
    day = datetime.strptime(date_str, "%Y-%m-%d").date()
    sim_dt = datetime(day.year, day.month, day.day, 15, 30, 0, tzinfo=timezone.utc)
    set_simulated_date(sim_dt)

    print(f"\nRunning trading cycle for date: {date_str}")
    stats = run_trading_cycle()

    set_simulated_date(None)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading Orchestrator")
    parser.add_argument("--date", type=str, default=None, help="Run for a specific date (YYYY-MM-DD)")
    parser.add_argument("--replay", type=int, default=None, help="Replay last N trading days")
    args = parser.parse_args()

    if args.replay:
        run_replay(args.replay)
    elif args.date:
        run_single_date(args.date)
    else:
        run_trading_cycle()
