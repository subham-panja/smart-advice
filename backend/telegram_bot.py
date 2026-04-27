import os
import sys
import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from datetime import datetime, timedelta
from pymongo import MongoClient

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(backend_dir)

import config

# Check if Telegram is enabled
if not getattr(config, 'TELEGRAM_CONFIG', {}).get('enabled', False):
    print("Telegram bot is disabled in config.py. Exiting.")
    sys.exit(0)

# Get Token
TOKEN = config.TELEGRAM_CONFIG.get('bot_token', '')
if not TOKEN or TOKEN == 'YOUR_TELEGRAM_BOT_TOKEN_HERE':
    print("Please set a valid Telegram bot token in config.py. Exiting.")
    sys.exit(0)

# Initialize Bot
bot = telebot.TeleBot(TOKEN)
ALLOWED_USERS = config.TELEGRAM_CONFIG.get('allowed_user_ids', [])

def check_permission(message):
    """Check if the user is allowed to use the bot."""
    if ALLOWED_USERS and message.from_user.id not in ALLOWED_USERS:
        bot.reply_to(message, "⛔ You are not authorized to use this bot.")
        return False
    return True

# Keyboard Menu
def get_main_keyboard():
    markup = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    btn_run = KeyboardButton("▶️ Run Analysis")
    btn_view = KeyboardButton("📊 View Recommendations")
    markup.add(btn_run, btn_view)
    return markup

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    if not check_permission(message): return
    bot.reply_to(
        message, 
        "🤖 *Smart Advice Trading Dashboard*\n\nWelcome! I am your personal swing trading assistant.", 
        parse_mode='Markdown',
        reply_markup=get_main_keyboard()
    )

@bot.message_handler(func=lambda message: message.text == "▶️ Run Analysis")
def run_analysis_command(message):
    if not check_permission(message): return
    
    bot.reply_to(message, "⏳ *Starting Analysis Pipeline...*\nThis usually takes about 30-60 seconds. I'll notify you when it's done.", parse_mode='Markdown')
    
    try:
        # Run the analysis script via subprocess to isolate memory
        import subprocess
        result = subprocess.run(
            ['python', os.path.join(backend_dir, 'run_analysis.py')], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            bot.send_message(message.chat.id, "✅ *Analysis Complete!*\nFetching results...", parse_mode='Markdown')
            # Trigger view recommendations logic
            view_recommendations_command(message, today_only=True)
        else:
            bot.send_message(message.chat.id, f"❌ *Analysis Failed*\n\n```text\n{result.stderr[-500:]}\n```", parse_mode='Markdown')
            
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ *Error starting analysis:*\n{str(e)}", parse_mode='Markdown')


@bot.message_handler(func=lambda message: message.text == "📊 View Recommendations")
def view_recommendations_command(message, today_only=False):
    if not check_permission(message): return
    
    bot.send_message(message.chat.id, "🔍 Fetching recommendations from Database...")
    
    try:
        # Connect to DB
        client = MongoClient(f"mongodb://{config.MONGODB_HOST}:{config.MONGODB_PORT}/")
        db = client[config.MONGODB_DATABASE]
        
        # Build query
        query = {'is_recommended': True}
        if today_only:
            today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            query['recommendation_date'] = {'$gte': today}
            
        recs = list(db.recommended_shares.find(query).sort('combined_score', -1))
        
        if not recs:
            msg = "📭 No recommendations found for today." if today_only else "📭 No active recommendations found in the database."
            bot.send_message(message.chat.id, msg)
            return
            
        bot.send_message(message.chat.id, f"🎯 Found *{len(recs)}* strong recommendations:", parse_mode='Markdown')
        
        for rec in recs:
            # Format the message nicely
            symbol = rec.get('symbol', 'UNKNOWN')
            company = rec.get('company_name', '')
            strength = rec.get('recommendation_strength', 'BUY')
            buy_price = rec.get('buy_price', 0)
            sell_price = rec.get('sell_price', 0)
            stop_loss = rec.get('stop_loss', 0)
            
            # Scores
            tech = rec.get('technical_score', 0)
            fund = rec.get('fundamental_score', 0)
            combined = rec.get('combined_score', 0)
            
            # Trade Plan Details
            trade_plan = rec.get('trade_plan', {})
            rr_ratio = trade_plan.get('risk_reward_ratio', 0)
            if rr_ratio == 0 and buy_price > 0 and stop_loss > 0 and sell_price > 0:
                rr_ratio = (sell_price - buy_price) / (buy_price - stop_loss)
            
            # Backtest
            bt = rec.get('backtest_metrics', {})
            cagr = bt.get('cagr', 0)
            win_rate = bt.get('win_rate', 0)
            
            msg = (
                f"📈 *{symbol}* - {company}\n"
                f"🔥 Signal: *{strength}*\n"
                f"──────────────\n"
                f"🎯 *Trade Plan:*\n"
                f"• 💰 Entry: ₹{buy_price:.2f}\n"
                f"• 🟢 Target: ₹{sell_price:.2f}\n"
                f"• 🔴 Stop Loss: ₹{stop_loss:.2f}\n"
                f"• ⚖️ R:R Ratio: 1:{rr_ratio:.2f}\n"
                f"──────────────\n"
                f"📊 *Scores:*\n"
                f"• Combined: {combined:.2f}\n"
                f"• Technical: {tech:.2f}\n"
                f"• Fundamental: {fund:.2f}\n"
                f"──────────────\n"
            )
            
            # Smart Money & Sector
            sector_data = rec.get('sector_analysis', {})
            sector_name = sector_data.get('sector', 'Unknown')
            sector_rec = sector_data.get('recommendation', '')
            
            fii_dii = rec.get('market_regime', {}).get('fii_dii_status', {}) # if you saved it here, or just grab from global config, but since it's historical, let's grab from DB if it exists
            
            smart_money = rec.get('detailed_analysis', {}).get('smart_money', {})
            delivery = smart_money.get('delivery_pct', 0)
            
            if sector_name != 'Unknown' or delivery > 0:
                msg += f"🏦 *Smart Money & Sector:*\n"
                if sector_name != 'Unknown':
                    msg += f"• 🏭 Sector: {sector_name}\n"
                    msg += f"• 🧭 Sector Flow: {sector_rec}\n"
                if delivery > 0:
                    msg += f"• 🚚 Delivery Vol: {delivery:.1f}%\n"
                msg += f"──────────────\n"
            
            msg += (
                f"⚙️ *Backtest Performance:*\n"
                f"• Hist. CAGR: {cagr:.1f}%\n"
                f"• Win Rate: {win_rate:.1f}%\n"
                f"• Expectancy: ₹{bt.get('expectancy', 0):.2f}\n"
                f"• Profit Factor: {bt.get('profit_factor', 0):.2f}\n"
            )
            
            # Add strategy details and entry patterns
            tech_details = rec.get('detailed_analysis', {}).get('technical', {})
            
            # Entry Patterns
            patterns = tech_details.get('entry_patterns', {}).get('patterns', {})
            active_patterns = [p_data.get('description', p_name) for p_name, p_data in patterns.items() if p_data.get('detected')]
            if active_patterns:
                msg += f"──────────────\n🧩 *Triggered Patterns:*\n"
                for p in active_patterns:
                    msg += f"• ✅ {p}\n"
                    
            # Individual Strategies
            individual_strats = tech_details.get('individual_strategies', {})
            active_strats = [strat_name for strat_name, data in individual_strats.items() if data.get('signal', 0) > 0]
            if active_strats:
                msg += f"──────────────\n🤖 *Bullish Indicators:*\n"
                for s in active_strats:
                    msg += f"• 📈 {s}\n"
            
            bot.send_message(message.chat.id, msg, parse_mode='Markdown')

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ *Database Error:*\n{str(e)}", parse_mode='Markdown')

print("Telegram Bot is running! Waiting for messages...")
bot.infinity_polling()
