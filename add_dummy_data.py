#!/usr/bin/env python3
"""
Add Dummy Data Script
File: add_dummy_data.py

This script adds dummy data to the database in the same format that run_analysis.py saves.
It creates sample stock recommendations and backtest results for testing purposes.
"""

import sys
import os
from datetime import datetime, timedelta
import random

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from database import get_mongodb, insert_backtest_result
from utils.logger import setup_logging

# Initialize logging
logger = setup_logging()

# Sample stock symbols and company names for dummy data
SAMPLE_STOCKS = [
    {'symbol': 'RELIANCE', 'company_name': 'Reliance Industries Limited'},
    {'symbol': 'TCS', 'company_name': 'Tata Consultancy Services Limited'},
    {'symbol': 'HDFCBANK', 'company_name': 'HDFC Bank Limited'},
    {'symbol': 'INFY', 'company_name': 'Infosys Limited'},
    {'symbol': 'ICICIBANK', 'company_name': 'ICICI Bank Limited'},
    {'symbol': 'HINDUNILVR', 'company_name': 'Hindustan Unilever Limited'},
    {'symbol': 'ITC', 'company_name': 'ITC Limited'},
    {'symbol': 'SBIN', 'company_name': 'State Bank of India'},
    {'symbol': 'BHARTIARTL', 'company_name': 'Bharti Airtel Limited'},
    {'symbol': 'ASIANPAINT', 'company_name': 'Asian Paints Limited'},
    {'symbol': 'MARUTI', 'company_name': 'Maruti Suzuki India Limited'},
    {'symbol': 'LT', 'company_name': 'Larsen & Toubro Limited'},
    {'symbol': 'AXISBANK', 'company_name': 'Axis Bank Limited'},
    {'symbol': 'WIPRO', 'company_name': 'Wipro Limited'},
    {'symbol': 'NESTLEIND', 'company_name': 'Nestle India Limited'}
]

RECOMMENDATION_STRENGTHS = ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'OPPORTUNISTIC_BUY']
EFFECTIVENESS_LEVELS = ['Excellent', 'Good', 'Moderate', 'Fair']

def generate_technical_score():
    """Generate a realistic technical score between 0.3 and 0.9"""
    return round(random.uniform(0.3, 0.9), 4)

def generate_fundamental_score():
    """Generate a realistic fundamental score between 0.2 and 0.8"""
    return round(random.uniform(0.2, 0.8), 4)

def generate_sentiment_score():
    """Generate a realistic sentiment score between 0.1 and 0.7"""
    return round(random.uniform(0.1, 0.7), 4)

def generate_combined_score(tech_score, fund_score, sent_score):
    """Generate combined score based on individual scores"""
    # Weighted average with some randomness
    combined = (tech_score * 0.4 + fund_score * 0.4 + sent_score * 0.2)
    return round(combined + random.uniform(-0.1, 0.1), 4)

def generate_trade_plan():
    """Generate realistic trade plan data"""
    buy_price = round(random.uniform(100, 5000), 2)
    sell_price = round(buy_price * random.uniform(1.05, 1.25), 2)  # 5-25% upside
    days_to_target = random.randint(30, 180)
    
    return {
        'buy_price': buy_price,
        'sell_price': sell_price,
        'days_to_target': days_to_target,
        'expected_return_percent': round(((sell_price - buy_price) / buy_price) * 100, 2),
        'est_time_to_target': f"{days_to_target} days"
    }

def generate_backtest_metrics():
    """Generate realistic backtest metrics"""
    cagr = round(random.uniform(5, 25), 2)
    win_rate = round(random.uniform(45, 75), 2)
    max_drawdown = round(random.uniform(8, 25), 2)
    total_trades = random.randint(15, 50)
    winning_trades = int((win_rate / 100) * total_trades)
    losing_trades = total_trades - winning_trades
    
    # Generate sample transactions
    transactions = []
    for i in range(min(10, total_trades)):
        action = random.choice(['BUY', 'SELL'])
        price = round(random.uniform(100, 3000), 2)
        shares = random.randint(10, 100)
        date = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
        
        transactions.append({
            'strategy': random.choice(['RSI_Strategy', 'MACD_Strategy', 'MA_Strategy']),
            'date': date,
            'action': action,
            'price': price,
            'shares': shares,
            'value': round(price * shares, 2)
        })
    
    return {
        'cagr': cagr,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'sharpe_ratio': round(random.uniform(0.8, 2.5), 2),
        'effectiveness': random.choice(EFFECTIVENESS_LEVELS),
        'buy_sell_transactions': transactions,
        'strategy_breakdown': {
            'RSI_Strategy': {
                'cagr': round(random.uniform(3, 20), 2),
                'win_rate': round(random.uniform(40, 70), 2),
                'max_drawdown': round(random.uniform(5, 20), 2),
                'total_trades': random.randint(5, 20),
                'trades': []
            },
            'MACD_Strategy': {
                'cagr': round(random.uniform(4, 22), 2),
                'win_rate': round(random.uniform(42, 72), 2),
                'max_drawdown': round(random.uniform(6, 22), 2),
                'total_trades': random.randint(5, 20),
                'trades': []
            }
        },
        'date_range': {
            'start_date': (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'period_days': 730
        },
        'capital_info': {
            'initial_capital': 100000,
            'final_capital': round(100000 * (1 + cagr/100) ** 2, 2),
            'total_return': round(((100000 * (1 + cagr/100) ** 2) - 100000) / 100000 * 100, 2)
        }
    }

def generate_reason(symbol, recommendation_strength):
    """Generate a realistic reason for the recommendation"""
    reasons = [
        f"{symbol} shows strong technical momentum with bullish indicators across multiple timeframes.",
        f"Fundamental analysis reveals {symbol} is undervalued with strong growth prospects.",
        f"{symbol} demonstrates excellent risk-adjusted returns with consistent outperformance.",
        f"Technical breakout pattern identified in {symbol} with volume confirmation.",
        f"{symbol} exhibits strong sector rotation momentum with institutional buying.",
        f"Value opportunity in {symbol} with improving financial metrics and market position."
    ]
    
    strength_modifiers = {
        'STRONG_BUY': "Exceptional opportunity with high conviction across all metrics.",
        'BUY': "Solid investment opportunity with favorable risk-reward profile.",
        'WEAK_BUY': "Moderate opportunity with some upside potential identified.",
        'OPPORTUNISTIC_BUY': "Tactical opportunity with specific entry conditions met."
    }
    
    base_reason = random.choice(reasons)
    modifier = strength_modifiers.get(recommendation_strength, "")
    
    return f"{base_reason} {modifier}".strip()

def add_dummy_recommendations(count=10):
    """Add dummy stock recommendations to the database"""
    logger.info(f"Adding {count} dummy recommendations...")
    
    app = create_app()
    with app.app_context():
        db = get_mongodb()
        
        added_count = 0
        for i in range(count):
            # Select a random stock
            stock = random.choice(SAMPLE_STOCKS)
            symbol = stock['symbol']
            company_name = stock['company_name']
            
            # Generate scores
            tech_score = generate_technical_score()
            fund_score = generate_fundamental_score()
            sent_score = generate_sentiment_score()
            combined_score = generate_combined_score(tech_score, fund_score, sent_score)
            
            # Generate trade plan
            trade_plan = generate_trade_plan()
            
            # Generate backtest metrics
            backtest_metrics = generate_backtest_metrics()
            
            # Select recommendation strength
            recommendation_strength = random.choice(RECOMMENDATION_STRENGTHS)
            
            # Generate reason
            reason = generate_reason(symbol, recommendation_strength)
            
            # Prepare document for MongoDB (same format as run_analysis.py)
            doc = {
                'symbol': symbol,
                'company_name': company_name,
                'technical_score': tech_score,
                'fundamental_score': fund_score,
                'sentiment_score': sent_score,
                'combined_score': combined_score,
                'is_recommended': True,
                'recommendation_strength': recommendation_strength,
                'reason': reason,
                'buy_price': trade_plan['buy_price'],
                'sell_price': trade_plan['sell_price'],
                'est_time_to_target': trade_plan['est_time_to_target'],
                'backtest_metrics': backtest_metrics,
                'recommendation_date': datetime.utcnow(),
                'expected_return_percent': trade_plan['expected_return_percent']
            }
            
            try:
                # Use upsert to insert or update (same as run_analysis.py)
                result = db.recommended_shares.update_one(
                    {'symbol': symbol},
                    {'$set': doc},
                    upsert=True
                )
                
                if result.upserted_id:
                    logger.info(f"Added new dummy recommendation: {symbol} - buy_price=${trade_plan['buy_price']:.2f}, sell_price=${trade_plan['sell_price']:.2f}, expected_return={trade_plan['expected_return_percent']:.2f}%")
                else:
                    logger.info(f"Updated existing dummy recommendation: {symbol}")
                
                added_count += 1
                
            except Exception as e:
                logger.error(f"Error adding dummy recommendation for {symbol}: {e}")
        
        logger.info(f"Successfully added {added_count} dummy recommendations")

def add_dummy_backtest_results(count=15):
    """Add dummy backtest results to the database"""
    logger.info(f"Adding {count} dummy backtest results...")
    
    app = create_app()
    with app.app_context():
        added_count = 0
        
        for i in range(count):
            # Select a random stock
            stock = random.choice(SAMPLE_STOCKS)
            symbol = stock['symbol']
            
            # Generate backtest metrics
            cagr = round(random.uniform(5, 25), 2)
            win_rate = round(random.uniform(45, 75), 2)
            max_drawdown = round(random.uniform(8, 25), 2)
            total_trades = random.randint(15, 50)
            winning_trades = int((win_rate / 100) * total_trades)
            losing_trades = total_trades - winning_trades
            
            # Generate additional metrics
            avg_trade_duration = random.randint(3, 15)
            avg_profit_per_trade = round(random.uniform(500, 3000), 2)
            avg_loss_per_trade = round(random.uniform(-2000, -300), 2)
            largest_win = round(random.uniform(3000, 10000), 2)
            largest_loss = round(random.uniform(-8000, -2000), 2)
            sharpe_ratio = round(random.uniform(0.8, 2.5), 2)
            volatility = round(random.uniform(15, 35), 2)
            
            # Date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            # Capital info
            initial_capital = 100000
            final_capital = round(initial_capital * (1 + cagr/100) ** 2, 2)
            total_return = round(((final_capital - initial_capital) / initial_capital) * 100, 2)
            
            try:
                # Use the same function as run_analysis.py
                insert_backtest_result(
                    symbol=symbol,
                    period='Overall',
                    cagr=cagr,
                    win_rate=win_rate,
                    max_drawdown=max_drawdown,
                    total_trades=total_trades,
                    winning_trades=winning_trades,
                    losing_trades=losing_trades,
                    avg_trade_duration=avg_trade_duration,
                    avg_profit_per_trade=avg_profit_per_trade,
                    avg_loss_per_trade=avg_loss_per_trade,
                    largest_win=largest_win,
                    largest_loss=largest_loss,
                    sharpe_ratio=sharpe_ratio,
                    sortino_ratio=round(sharpe_ratio * 1.2, 2),
                    calmar_ratio=round(cagr / max_drawdown, 2),
                    volatility=volatility,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    final_capital=final_capital,
                    total_return=total_return
                )
                
                logger.info(f"Added backtest result for {symbol}: CAGR={cagr:.2f}%, Win Rate={win_rate:.2f}%, Max Drawdown={max_drawdown:.2f}%")
                added_count += 1
                
            except Exception as e:
                logger.error(f"Error adding dummy backtest result for {symbol}: {e}")
        
        logger.info(f"Successfully added {added_count} dummy backtest results")

def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Add dummy data to the database')
    parser.add_argument('--recommendations', type=int, default=10, help='Number of dummy recommendations to add (default: 10)')
    parser.add_argument('--backtest-results', type=int, default=15, help='Number of dummy backtest results to add (default: 15)')
    parser.add_argument('--recommendations-only', action='store_true', help='Add only recommendations')
    parser.add_argument('--backtest-only', action='store_true', help='Add only backtest results')
    
    args = parser.parse_args()
    
    try:
        if args.backtest_only:
            add_dummy_backtest_results(args.backtest_results)
        elif args.recommendations_only:
            add_dummy_recommendations(args.recommendations)
        else:
            # Add both by default
            add_dummy_recommendations(args.recommendations)
            add_dummy_backtest_results(args.backtest_results)
        
        logger.info("Dummy data addition completed successfully!")
        
        # Show summary
        app = create_app()
        with app.app_context():
            db = get_mongodb()
            total_recommendations = db.recommended_shares.count_documents({})
            total_backtest_results = db.backtest_results.count_documents({})
            
            logger.info(f"Total recommendations in database: {total_recommendations}")
            logger.info(f"Total backtest results in database: {total_backtest_results}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
