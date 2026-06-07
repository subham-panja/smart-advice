import sys
from datetime import datetime, timezone

import telebot
from pymongo import MongoClient
from telebot.types import ReplyKeyboardMarkup

import config
from utils.fivepaisa_client import FivePaisaUtility, get_5paisa_balance, get_5paisa_holdings

if not getattr(config, "TELEGRAM_CONFIG", {}).get("enabled", False):
    sys.exit(0)

TOKEN = config.TELEGRAM_CONFIG.get("bot_token", "")
bot = telebot.TeleBot(TOKEN)
ALLOWED = config.TELEGRAM_CONFIG.get("allowed_user_ids", [])


def check(msg):
    if ALLOWED and msg.from_user.id not in ALLOWED:
        bot.reply_to(msg, "⛔ Unauthorized")
        return False
    return True


def get_kb():
    kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.add("▶️ Run Analysis", "⚡ Run Trading Cycle")
    kb.add("📊 View Recommendations", "📈 View Positions")
    kb.add("💼 Portfolio", "💰 Balance", "❓ Help")
    return kb


# ---------------------------------------------------------------------------
# 5paisa OAuth commands
# ---------------------------------------------------------------------------


@bot.message_handler(commands=["5paisa_login"])
def fivepaisa_login(m):
    """Generate OAuth URL for 5paisa login."""
    if not check(m):
        return
    util = FivePaisaUtility()
    url = util.get_oauth_url()
    if not url:
        bot.send_message(m.chat.id, "❌ 5paisa API keys missing in .env")
        return
    bot.send_message(
        m.chat.id,
        "🔐 <b>5paisa Login</b>\n\n"
        "1. Visit this link:\n{}\n\n"
        "2. Log in and approve the request\n\n"
        "3. You will be redirected to a page with a <code>request_token</code>\n\n"
        "4. Send it back here as:\n<code>/5paisa_token YOUR_REQUEST_TOKEN</code>".format(url),
        parse_mode="HTML",
    )


@bot.message_handler(commands=["5paisa_token"])
def fivepaisa_token(m):
    """Complete OAuth flow with the request token."""
    if not check(m):
        return
    parts = m.text.split()
    if len(parts) < 2:
        bot.send_message(m.chat.id, "❌ Usage: <code>/5paisa_token YOUR_REQUEST_TOKEN</code>", parse_mode="HTML")
        return

    request_token = parts[1]
    util = FivePaisaUtility()
    result = util.login_with_request_token(request_token)

    if result.get("status") == "success":
        access_token = result["access_token"]
        bot.send_message(
            m.chat.id,
            "✅ <b>Login Successful!</b>\n\n"
            "Access Token:\n<code>{}</code>\n\n"
            "Add this to your .env as:\n"
            "<code>FIVEPAISA_ACCESS_TOKEN={}</code>\n\n"
            "Then restart the bot.".format(access_token, access_token),
            parse_mode="HTML",
        )
    else:
        bot.send_message(m.chat.id, "❌ <b>Login Failed</b>\n{}".format(result.get("message")), parse_mode="HTML")


@bot.message_handler(commands=["start", "help"])
@bot.message_handler(func=lambda m: m.text == "❓ Help")
def welcome(m):
    if not check(m):
        return
    bot.send_message(
        m.chat.id,
        "🤖 <b>Smart Advice Bot</b>\n\nUse buttons to manage your trading.",
        parse_mode="HTML",
        reply_markup=get_kb(),
    )


@bot.message_handler(func=lambda m: m.text == "▶️ Run Analysis")
def run_analysis(m):
    if not check(m):
        return
    bot.reply_to(m, "⏳ <b>Starting Analysis...</b>", parse_mode="HTML")
    import os
    import subprocess

    script_path = os.path.join(os.path.dirname(__file__), "run_analysis.py")
    res = subprocess.run([sys.executable, script_path, "--all", "--max-stocks", "50"], capture_output=True, text=True)
    if res.returncode == 0:
        bot.send_message(m.chat.id, "✅ <b>Complete!</b>", parse_mode="HTML")
        view_recs(m, today=True)
    else:
        bot.send_message(m.chat.id, f"❌ <b>Failed</b>\n{res.stderr[-200:]}", parse_mode="HTML")


@bot.message_handler(func=lambda m: m.text == "⚡ Run Trading Cycle")
def run_trading_cycle(m):
    if not check(m):
        return
    # Load strategy to get paper trading mode
    from utils.strategy_loader import StrategyLoader

    strategies = StrategyLoader.load_all_strategies()
    is_paper = True
    if strategies:
        trading_cfg = strategies[0].get("trading_config", {})
        is_paper = trading_cfg.get("is_paper_trading", True)
    mode_text = "(Paper Trading)" if is_paper else "⚠️ (LIVE TRADING)"
    bot.reply_to(m, f"⚡ <b>Executing Trading Cycle {mode_text}...</b>", parse_mode="HTML")
    import os
    import subprocess

    script_path = os.path.join(os.path.dirname(__file__), "main_orchestrator.py")
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if res.returncode == 0:
        bot.send_message(m.chat.id, "✅ <b>Trading Cycle Complete!</b>", parse_mode="HTML")
        view_positions(m)
    else:
        bot.send_message(m.chat.id, f"❌ <b>Execution Failed</b>\n{res.stderr[-200:]}", parse_mode="HTML")


@bot.message_handler(func=lambda m: m.text == "📊 View Recommendations")
def view_recs(m, today=False):
    if not check(m):
        return
    db = MongoClient(f"mongodb://{config.MONGODB_HOST}:{config.MONGODB_PORT}/")[config.MONGODB_DATABASE]
    query = {}
    if today:
        query["recommendation_date"] = {"$gte": datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)}

    recs = list(db.recommended_shares.find(query).sort("combined_score", -1))
    if not recs:
        bot.send_message(m.chat.id, "📭 No signals.")
        return

    for r in recs:
        bt = r.get("backtest_metrics", {})
        score = r.get("combined_score", 0)
        quantity = r.get("suggested_quantity", 1)

        # Calculate sizing for the message
        initial_cap = config.TRADING_OPTIONS.get("initial_capital", 100000.0)
        buy_price = r.get("buy_price", 0)
        sell_price = r.get("sell_price", 0)
        stop_loss = r.get("stop_loss", 0)

        total_cost = quantity * buy_price
        cap_pct = r.get("allocation_pct", (total_cost / initial_cap) * 100)
        rr = r.get("rr_ratio", 0)

        msg = (
            f"📈 <b>{r['symbol']}</b> | <b>{r.get('strategy_name', 'Delayed_EP')}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🎯 Score: <b>{score:.1f}/100</b>\n"
            f"💰 <b>Trade Plan</b>:\n"
            f"• Entry: ₹{buy_price:.2f}\n"
            f"• Target: ₹{sell_price:.2f}\n"
            f"• Stop Loss: ₹{stop_loss:.2f}\n"
            f"• RR Ratio: <b>{rr:.2f}</b>\n\n"
            f"🔢 <b>Sizing (₹{initial_cap/100000:.1f}L Cap)</b>:\n"
            f"• Quantity: <b>{quantity}</b>\n"
            f"• Allocation: <b>₹{total_cost:,.2f} ({cap_pct:.1f}%)</b>\n\n"
            f"📊 <b>Backtest Stats</b>:\n"
            f"• Trades: <b>{bt.get('total_trades', 0)}</b>\n"
            f"• Win Rate: {bt.get('avg_win_rate', 0):.1f}%\n"
            f"• Avg CAGR: {bt.get('avg_cagr', 0):.1f}%\n"
            f"• Expectancy: {bt.get('avg_expectancy', 0.0):.2f}\n\n"
            f"📝 <b>Analysis</b>: {r.get('reason') or 'Technical Momentum Breakout'}"
        )

        bot.send_message(m.chat.id, msg, parse_mode="HTML")


@bot.message_handler(func=lambda m: m.text == "💰 Balance")
def balance(m):
    if not check(m):
        return
    b = get_5paisa_balance()
    if b.get("status") == "success":
        msg = f"💰 <b>Balance</b>\nMargin: ₹{b.get('available_margin', 0):.2f}\nNet: ₹{b.get('net_available', 0):.2f}"
        bot.send_message(m.chat.id, msg, parse_mode="HTML")
    else:
        bot.send_message(m.chat.id, f"❌ Balance Error\n{b.get('message', 'Unknown error')}", parse_mode="HTML")


@bot.message_handler(func=lambda m: m.text == "📈 View Positions")
def view_positions(m):
    if not check(m):
        return
    db = MongoClient(f"mongodb://{config.MONGODB_HOST}:{config.MONGODB_PORT}/")[config.MONGODB_DATABASE]
    positions = list(db.positions.find({"status": "OPEN"}))

    # Load strategy to get paper trading mode
    from utils.strategy_loader import StrategyLoader

    strategies = StrategyLoader.load_all_strategies()
    is_paper = True
    if strategies:
        trading_cfg = strategies[0].get("trading_config", {})
        is_paper = trading_cfg.get("is_paper_trading", True)
    header = "📝 *Active Paper Positions*" if is_paper else "💼 *Active Live Positions*"

    if not positions:
        bot.send_message(m.chat.id, f"📭 {header}\nNo open positions found.")
        return

    total_mkt_val = 0
    total_pnl_val = 0
    initial_cap = config.TRADING_OPTIONS.get("initial_capital", 100000.0)

    for p in positions:
        current_p = p.get("current_price", p["entry_price"])
        pnl_val = (current_p - p["entry_price"]) * p["quantity"]
        pnl_pct = ((current_p - p["entry_price"]) / p["entry_price"]) * 100
        total_cost = p.get("total_investment", p["quantity"] * p["entry_price"])

        total_mkt_val += current_p * p["quantity"]
        total_pnl_val += pnl_val

        status_emoji = "🟢" if pnl_pct >= 0 else "🔴"
        allocation = p.get("allocation_pct", (total_cost / initial_cap) * 100)

        # Use current (trailed) SL, fall back to original
        active_sl = p.get("current_stop_loss", p.get("stop_loss", 0))
        original_sl = p.get("stop_loss", 0)
        sl_trailed = active_sl > original_sl

        active_target = p.get("current_target", p.get("target", 0))

        sl_line = f"🛑 <b>SL</b>: ₹{active_sl:.2f}"
        if sl_trailed:
            sl_line += f" <i>(trailed from ₹{original_sl:.2f})</i>"

        msg = (
            f"{status_emoji} <b>{p['symbol']}</b> | {p.get('strategy_name', 'Swing_Trading')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>Entered</b>: {p['entry_date'].strftime('%Y-%m-%d %H:%M')}\n"
            f"🔢 <b>Quantity</b>: {p['quantity']} @ ₹{p['entry_price']:.2f}\n"
            f"💰 <b>Total Cost</b>: ₹{total_cost:,.2f} (<b>{allocation:.1f}% Cap</b>)\n"
            f"🎯 <b>Target</b>: ₹{active_target:.2f} | {sl_line}\n"
            f"💸 <b>Unrealized PnL</b>: ₹{pnl_val:+,.2f} ({pnl_pct:+.2f}%)\n"
        )

        # Pyramiding info
        adds = p.get("adds_count", 0)
        if adds > 0:
            last_add_price = p.get("last_add_price", 0)
            msg += f"🔺 <b>Pyramid</b>: {adds} add(s) (last at ₹{last_add_price:.2f})\n"

        # Partial exits info
        partial_exits = p.get("partial_exits", [])
        if partial_exits:
            for pe in partial_exits:
                pe_qty = pe.get("quantity", 0)
                pe_price = pe.get("price", 0)
                pe_reason = pe.get("reason", "")
                msg += f"📤 <b>Sold</b> {pe_qty} @ ₹{pe_price:.2f} ({pe_reason})\n"

        # Recent updates from history (last 3)
        updates = p.get("updates", [])
        recent = updates[-3:] if updates else []
        if recent:
            msg += "📋 <b>Recent Updates</b>:\n"
            for u in recent:
                utype = u.get("type", "")
                udate = u.get("date")
                date_str = udate.strftime("%m/%d") if udate else "?"
                if utype == "TRAIL_SL":
                    prev = u.get("prev_sl", 0)
                    curr = u.get("current_sl", 0)
                    msg += f"  • {date_str} Trail: ₹{prev:.2f} → ₹{curr:.2f}\n"
                elif utype == "PYRAMID":
                    msg += f"  • {date_str} Pyramid add\n"
                elif utype == "TARGET_HIT":
                    msg += f"  • {date_str} Target hit\n"
                elif utype == "PARTIAL_SELL":
                    msg += f"  • {date_str} Partial sell\n"
                elif utype == "ENTRY_CORRECTION":
                    msg += f"  • {date_str} Entry corrected\n"

        msg += f"🆔 <b>Ref</b>: <code>{str(p.get('recomm_id', 'N/A'))[-8:]}</code>"

        bot.send_message(m.chat.id, msg, parse_mode="HTML")

    # Portfolio Summary Footer
    cash_left = initial_cap - sum(p.get("total_investment", p["quantity"] * p["entry_price"]) for p in positions)
    total_equity = total_mkt_val + cash_left
    overall_pnl_pct = ((total_equity - initial_cap) / initial_cap) * 100

    summary = (
        f"📊 <b>PORTFOLIO SUMMARY</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💵 <b>Initial Capital</b>: ₹{initial_cap:,.2f}\n"
        f"📈 <b>Market Value</b>: ₹{total_mkt_val:,.2f}\n"
        f"🏦 <b>Cash Balance</b>: ₹{cash_left:,.2f}\n"
        f"💰 <b>Net Equity</b>: ₹{total_equity:,.2f}\n"
        f"📊 <b>Total PnL</b>: ₹{total_pnl_val:+,.2f} ({overall_pnl_pct:+.2f}%)"
    )

    bot.send_message(m.chat.id, summary, parse_mode="HTML")


@bot.message_handler(func=lambda m: m.text == "💼 Portfolio")
def portfolio(m):
    if not check(m):
        return
    p = get_5paisa_holdings()
    if p.get("status") == "success":
        active = [h for h in p.get("data", []) if h.get("Quantity", 0) > 0]
        if not active:
            bot.send_message(m.chat.id, "📭 Empty")
            return

        total_invested = 0
        total_current = 0
        lines = []
        for h in active:
            qty = h.get("Quantity", 0)
            avg = h.get("AvgRate", 0)
            ltp = h.get("CurrentPrice", h.get("LTP", 0))
            invested = qty * avg
            current = qty * ltp
            pnl = current - invested
            pnl_pct = (pnl / invested * 100) if invested > 0 else 0
            total_invested += invested
            total_current += current
            emoji = "🟢" if pnl >= 0 else "🔴"
            lines.append(
                "{} <b>{}</b>\n"
                "  Qty: {} | Avg: ₹{:.2f} | LTP: ₹{:.2f}\n"
                "  Value: ₹{:,.0f} | PnL: ₹{:+,.0f} ({:+.1f}%)".format(
                    emoji, h["Symbol"], qty, avg, ltp, current, pnl, pnl_pct
                )
            )

        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

        # Send top 10 holdings + summary in first message
        batch_size = 10
        first_batch = lines[:batch_size]
        header = (
            "💼 <b>Portfolio</b> ({} holdings)\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            "{}\n\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "💵 Total Invested: ₹{:,.0f}\n"
            "📈 Current Value: ₹{:,.0f}\n"
            "💰 Total PnL: ₹{:+,.0f} ({:+.1f}%)\n\n"
            "<i>Showing top {} of {} holdings.</i>".format(
                len(active),
                "\n\n".join(first_batch),
                total_invested,
                total_current,
                total_pnl,
                total_pnl_pct,
                min(batch_size, len(active)),
                len(active),
            )
        )
        bot.send_message(m.chat.id, header, parse_mode="HTML")

        # Send remaining holdings in batches
        for i in range(batch_size, len(lines), batch_size):
            batch = lines[i : i + batch_size]
            bot.send_message(m.chat.id, "\n\n".join(batch), parse_mode="HTML")
    else:
        bot.send_message(m.chat.id, "❌ Error")


print("Bot running...")
bot.infinity_polling()
