import os
import sys
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from datetime import datetime, timezone
from pymongo import MongoClient
import config
from utils.fivepaisa_client import get_5paisa_balance, get_5paisa_holdings

if not getattr(config, 'TELEGRAM_CONFIG', {}).get('enabled', False): sys.exit(0)

TOKEN = config.TELEGRAM_CONFIG.get('bot_token', '')
bot = telebot.TeleBot(TOKEN)
ALLOWED = config.TELEGRAM_CONFIG.get('allowed_user_ids', [])

def check(msg):
    if ALLOWED and msg.from_user.id not in ALLOWED:
        bot.reply_to(msg, "⛔ Unauthorized"); return False
    return True

def get_kb():
    kb = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    kb.add("▶️ Run Analysis", "📊 View Recommendations", "💰 Balance", "💼 Portfolio", "❓ Help")
    return kb

@bot.message_handler(commands=['start', 'help'])
@bot.message_handler(func=lambda m: m.text == "❓ Help")
def welcome(m):
    if not check(m): return
    bot.send_message(m.chat.id, "🤖 *Smart Advice Bot*\n\nUse buttons to manage your trading.", parse_mode='Markdown', reply_markup=get_kb())

@bot.message_handler(func=lambda m: m.text == "▶️ Run Analysis")
def run_analysis(m):
    if not check(m): return
    bot.reply_to(m, "⏳ *Starting Analysis...*", parse_mode='Markdown')
    import subprocess
    res = subprocess.run(['python', 'run_analysis.py'], capture_output=True, text=True)
    if res.returncode == 0:
        bot.send_message(m.chat.id, "✅ *Complete!*")
        view_recs(m, today=True)
    else: bot.send_message(m.chat.id, f"❌ *Failed*\n{res.stderr[-200:]}")

@bot.message_handler(func=lambda m: m.text == "📊 View Recommendations")
def view_recs(m, today=False):
    if not check(m): return
    db = MongoClient(f"mongodb://{config.MONGODB_HOST}:{config.MONGODB_PORT}/")[config.MONGODB_DATABASE]
    query = {'is_recommended': True}
    if today: query['recommendation_date'] = {'$gte': datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)}
    
    recs = list(db.recommended_shares.find(query).sort('combined_score', -1))
    if not recs: bot.send_message(m.chat.id, "📭 No signals."); return
    
    for r in recs:
        msg = (f"📈 *{r['symbol']}*\n"
               f"💰 Entry: ₹{r['buy_price']:.2f} | Target: ₹{r['sell_price']:.2f}\n"
               f"🛑 SL: ₹{r['stop_loss']:.2f} | Score: {r['combined_score']:.2f}\n"
               f"📝 {r['reason']}")
        bot.send_message(m.chat.id, msg, parse_mode='Markdown')

@bot.message_handler(func=lambda m: m.text == "💰 Balance")
def balance(m):
    if not check(m): return
    b = get_5paisa_balance()
    if b.get('status') == 'success':
        msg = f"💰 *Balance*\nMargin: ₹{b.get('available_margin', 0):.2f}\nNet: ₹{b.get('net_available', 0):.2f}"
        bot.send_message(m.chat.id, msg, parse_mode='Markdown')
    else: bot.send_message(m.chat.id, "❌ Error")

@bot.message_handler(func=lambda m: m.text == "💼 Portfolio")
def portfolio(m):
    if not check(m): return
    p = get_5paisa_holdings()
    if p.get('status') == 'success':
        active = [h for h in p.get('data', []) if h.get('Quantity', 0) > 0]
        if not active: bot.send_message(m.chat.id, "📭 Empty"); return
        msg = "💼 *Portfolio*\n" + "\n".join([f"• {h['Symbol']}: {h['Quantity']} @ ₹{h['AvgRate']:.2f}" for h in active])
        bot.send_message(m.chat.id, msg, parse_mode='Markdown')
    else: bot.send_message(m.chat.id, "❌ Error")

print("Bot running...")
bot.infinity_polling()
