import sys
from datetime import datetime, timezone

import telebot
from pymongo import MongoClient
from telebot.types import ReplyKeyboardMarkup

import config
from utils.fivepaisa_client import get_5paisa_balance, get_5paisa_holdings

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
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if res.returncode == 0:
        bot.send_message(m.chat.id, "✅ <b>Complete!</b>", parse_mode="HTML")
        view_recs(m, today=True)
    else:
        bot.send_message(m.chat.id, f"❌ <b>Failed</b>\n{res.stderr[-200:]}", parse_mode="HTML")


@bot.message_handler(func=lambda m: m.text == "⚡ Run Trading Cycle")
def run_trading_cycle(m):
    if not check(m):
        return
    is_paper = config.TRADING_OPTIONS.get("is_paper_trading", True)
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
        initial_cap = config.TRADING_OPTIONS.get("initial_capital", 1000000.0)
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
        bot.send_message(m.chat.id, "❌ Error")


@bot.message_handler(func=lambda m: m.text == "📈 View Positions")
def view_positions(m):
    if not check(m):
        return
    db = MongoClient(f"mongodb://{config.MONGODB_HOST}:{config.MONGODB_PORT}/")[config.MONGODB_DATABASE]
    positions = list(db.positions.find({"status": "OPEN"}))

    is_paper = config.TRADING_OPTIONS.get("is_paper_trading", True)
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

        msg = (
            f"{status_emoji} <b>{p['symbol']}</b> | {p.get('strategy_name', 'Delayed_EP')}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>Entered</b>: {p['entry_date'].strftime('%Y-%m-%d %H:%M')}\n"
            f"🔢 <b>Quantity</b>: {p['quantity']} @ ₹{p['entry_price']:.2f}\n"
            f"💰 <b>Total Cost</b>: ₹{total_cost:,.2f} (<b>{allocation:.1f}% Cap</b>)\n"
            f"🎯 <b>Target</b>: ₹{p['target']:.2f} | 🛑 <b>SL</b>: ₹{p['stop_loss']:.2f}\n"
            f"💸 <b>Unrealized PnL</b>: ₹{pnl_val:+,.2f} ({pnl_pct:+.2f}%)\n"
            f"🆔 <b>Ref</b>: <code>{str(p.get('recomm_id', 'N/A'))[-8:]}</code>"
        )

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
        msg = "💼 <b>Portfolio</b>\n" + "\n".join(
            [f"• {h['Symbol']}: {h['Quantity']} @ ₹{h['AvgRate']:.2f}" for h in active]
        )
        bot.send_message(m.chat.id, msg, parse_mode="HTML")
    else:
        bot.send_message(m.chat.id, "❌ Error")


print("Bot running...")
bot.infinity_polling()
