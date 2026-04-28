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
from utils.fivepaisa_client import get_5paisa_balance, get_5paisa_holdings, FivePaisaUtility

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
    btn_balance = KeyboardButton("💰 Wallet Balance")
    btn_portfolio = KeyboardButton("💼 View Portfolio")
    btn_help = KeyboardButton("❓ Help & Commands")
    markup.add(btn_run, btn_view, btn_balance, btn_portfolio, btn_help)
    return markup

@bot.message_handler(commands=['start', 'help'])
@bot.message_handler(func=lambda message: message.text == "❓ Help & Commands")
def send_welcome(message):
    if not check_permission(message): return
    
    help_text = (
        "🤖 *Smart Advice Trading Dashboard*\n\n"
        "Welcome! I am your personal swing trading assistant. Here are my available commands:\n\n"
        "🚀 *Core Actions*\n"
        "• *Run Analysis*: Start the full market scan (Chartink + TA + Backtest).\n"
        "• *View Recommendations*: Show the latest BUY signals from the database.\n\n"
        "🏦 *Account Details*\n"
        "• *Wallet Balance*: Fetch your live margin and ledger balance from 5paisa.\n"
        "• *View Portfolio*: Check your current stock holdings.\n\n"
        "💡 *How to use*\n"
        "Use the buttons below to navigate, or simply type the command name."
    )
    
    bot.send_message(
        message.chat.id, 
        help_text, 
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


@bot.message_handler(func=lambda message: message.text == "💰 Wallet Balance")
def wallet_balance_command(message):
    if not check_permission(message): return
    
    bot.send_message(message.chat.id, "🏦 *Fetching your 5paisa balance...*", parse_mode='Markdown')
    
    try:
        balance = get_5paisa_balance()
        
        if balance.get('status') == 'success':
            data = balance
            # If the utility returned the simplified dict
            if 'available_margin' in data:
                msg = (
                    "💰 *5paisa Wallet Balance*\n"
                    "──────────────\n"
                    f"💳 *Available Margin:* ₹{data['available_margin']:.2f}\n"
                    f"📝 *Ledger Balance:* ₹{data['ledger_balance']:.2f}\n"
                    f"📊 *Utilized Margin:* ₹{data['utilized_margin']:.2f}\n"
                    f"📈 *Net Available:* ₹{data['net_available']:.2f}\n"
                    "──────────────\n"
                    "_Note: Data fetched live from your 5paisa account._"
                )
            else:
                msg = f"✅ *Balance Data:*\n\n```json\n{data}\n```"
            
            bot.send_message(message.chat.id, msg, parse_mode='Markdown')
        else:
            bot.send_message(message.chat.id, f"❌ *Error fetching balance:*\n{balance.get('message', 'Unknown error')}", parse_mode='Markdown')
            
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ *Bot Error:*\n```\n{str(e)}\n```", parse_mode='Markdown')


@bot.message_handler(func=lambda message: message.text == "💼 View Portfolio")
def view_portfolio_command(message):
    if not check_permission(message): return
    
    bot.send_message(message.chat.id, "💼 *Fetching your 5paisa holdings...*", parse_mode='Markdown')
    
    try:
        portfolio = get_5paisa_holdings()
        
        if portfolio.get('status') == 'success':
            holdings = portfolio.get('data', [])
            
            if not holdings:
                bot.send_message(message.chat.id, "📭 Your portfolio is currently empty.", parse_mode='Markdown')
                return
                
            # Filter out zero quantity holdings
            active_holdings = [h for h in holdings if h.get('Quantity', 0) > 0]
            
            if not active_holdings:
                bot.send_message(message.chat.id, "📭 You have no active holdings.", parse_mode='Markdown')
                return
                
            msg_chunks = []
            current_msg = "💼 *Your Active Portfolio*\n──────────────\n"
            total_invested = 0
            total_current = 0
            
            for h in active_holdings:
                symbol = h.get('Symbol', 'UNKNOWN')
                qty = h.get('Quantity', 0)
                avg_price = h.get('AvgRate', 0)
                ltp = h.get('CurrentPrice', avg_price) # Fallback to avg price if LTP is missing
                
                invested = qty * avg_price
                current = qty * ltp
                pnl = current - invested
                pnl_pct = (pnl / invested * 100) if invested > 0 else 0
                
                total_invested += invested
                total_current += current
                
                icon = "🟢" if pnl >= 0 else "🔴"
                stock_text = f"*{symbol}* ({qty} qty)\nAvg: ₹{avg_price:.2f} | LTP: ₹{ltp:.2f}\nP&L: {icon} ₹{pnl:.2f} ({pnl_pct:.2f}%)\n\n"
                
                if len(current_msg) + len(stock_text) > 3000:
                    msg_chunks.append(current_msg)
                    current_msg = ""
                
                current_msg += stock_text
                
            total_pnl = total_current - total_invested
            total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
            total_icon = "🟢" if total_pnl >= 0 else "🔴"
            
            summary = (
                "──────────────\n"
                f"📈 *Total Invested:* ₹{total_invested:.2f}\n"
                f"💰 *Current Value:* ₹{total_current:.2f}\n"
                f"📊 *Total P&L:* {total_icon} ₹{total_pnl:.2f} ({total_pnl_pct:.2f}%)"
            )
            
            if len(current_msg) + len(summary) > 3000:
                msg_chunks.append(current_msg)
                current_msg = summary
            else:
                current_msg += summary
                
            msg_chunks.append(current_msg)
            
            for chunk in msg_chunks:
                bot.send_message(message.chat.id, chunk, parse_mode='Markdown')

                
        else:
            bot.send_message(message.chat.id, f"❌ *Error fetching portfolio:*\n{portfolio.get('message', 'Unknown error')}", parse_mode='Markdown')
            
    except Exception as e:
        bot.send_message(message.chat.id, f"❌ *Bot Error:*\n```\n{str(e)}\n```", parse_mode='Markdown')


@bot.message_handler(commands=['5paisa_login'])
def fivepaisa_login_start(message):
    """Start the 5paisa OAuth login flow."""
    if not check_permission(message): return
    
    utility = FivePaisaUtility()
    oauth_url = utility.get_oauth_url()
    
    if not oauth_url:
        bot.send_message(message.chat.id, "❌ 5paisa credentials not configured in .env")
        return
    
    msg = (
        "🔐 *5paisa OAuth Login*\n\n"
        "Step 1: Click the link below and login with your 5paisa account:\n\n"
        f"`{oauth_url}`\n\n"
        "Step 2: After login, you will be redirected to a URL. "
        "Copy the *RequestToken* from the URL and send it to me as:\n\n"
        "`/5paisa_token YOUR_TOKEN_HERE`"
    )
    bot.send_message(message.chat.id, msg, parse_mode='Markdown')


@bot.message_handler(commands=['5paisa_token'])
def fivepaisa_login_token(message):
    """Complete the 5paisa OAuth flow with the request token."""
    if not check_permission(message): return
    
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        bot.send_message(message.chat.id, "❌ Usage: `/5paisa_token YOUR_TOKEN_HERE`", parse_mode='Markdown')
        return
    
    request_token = parts[1].strip()
    utility = FivePaisaUtility()
    result = utility.login_with_request_token(request_token)
    
    if result.get('status') == 'success':
        access_token = result.get('access_token', '')
        msg = (
            "✅ *5paisa Login Successful!*\n\n"
            f"Your Access Token:\n`{access_token}`\n\n"
            "⚠️ *Save this in your .env file as:*\n"
            f"`FIVEPAISA_ACCESS_TOKEN={access_token}`\n\n"
            "After saving, restart the bot. You won't need to login again."
        )
        bot.send_message(message.chat.id, msg, parse_mode='Markdown')
    else:
        bot.send_message(message.chat.id, f"❌ *Login Failed:*\n{result.get('message')}", parse_mode='Markdown')

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
            from datetime import timezone
            now = datetime.now(timezone.utc)
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
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

        # === DYNAMIC POSITION SIZING (NO ORDER PLACEMENT) ===
        try:
            balance = get_5paisa_balance()
            if balance.get('status') == 'success' and 'net_available' in balance:
                net_available = balance['net_available']
                if net_available > 0:
                    sizing_msg = f"🧮 *Dynamic Position Sizing (Based on ₹{net_available:.2f})*\n"
                    sizing_msg += "_Risking max 2% of available capital per trade_\n──────────────\n"
                    
                    risk_capital = net_available * 0.02 # 2% risk per trade
                    max_allocation = net_available * 0.25 # Max 25% of capital in one stock
                    
                    for rec in recs:
                        symbol = rec.get('symbol', 'UNKNOWN')
                        buy_price = rec.get('buy_price', 0)
                        stop_loss = rec.get('stop_loss', 0)
                        
                        if buy_price > 0 and stop_loss > 0 and buy_price > stop_loss:
                            risk_per_share = buy_price - stop_loss
                            shares_to_buy = int(risk_capital / risk_per_share)
                            
                            # Capital constraint
                            total_cost = shares_to_buy * buy_price
                            if total_cost > max_allocation:
                                shares_to_buy = int(max_allocation / buy_price)
                                total_cost = shares_to_buy * buy_price
                                
                            if shares_to_buy > 0:
                                sizing_msg += f"• *{symbol}*: Buy {shares_to_buy} shares (Cost: ₹{total_cost:.2f})\n"
                            else:
                                sizing_msg += f"• *{symbol}*: Too expensive for current margin.\n"
                    
                    bot.send_message(message.chat.id, sizing_msg, parse_mode='Markdown')
                else:
                    bot.send_message(message.chat.id, f"⚠️ *Insufficient Margin*\nYour Net Available margin is ₹{net_available:.2f}. Cannot generate dynamic position sizing.", parse_mode='Markdown')
        except Exception as e:
            print(f"Error calculating position sizing: {e}")

    except Exception as e:
        bot.send_message(message.chat.id, f"❌ *Database Error:*\n{str(e)}", parse_mode='Markdown')


print("Telegram Bot is running! Waiting for messages...")
bot.infinity_polling()
