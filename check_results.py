#!/usr/bin/env python3
"""
Check Analysis Results and System Status
"""

# Apply OpenMP fix
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

from database import get_mongodb
from pymongo import MongoClient
from datetime import datetime
import json

def check_system_status():
    """Check system components status."""
    print("🔍 System Status Check")
    print("=" * 50)
    
    # Check MongoDB connection
    try:
        client = MongoClient('localhost', 27017, serverSelectionTimeoutMS=2000)
        client.server_info()
        print("✅ MongoDB: Connected")
        
        # Check database and collections
        db = get_mongodb()
        collections = db.list_collection_names()
        print(f"✅ Database: super_advice (Collections: {collections})")
        
        # Check recommendations
        recommendations = db.recommended_shares
        total_recs = recommendations.count_documents({})
        recent_recs = recommendations.count_documents({
            'recommendation_date': {
                '$gte': datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            }
        })
        print(f"📊 Recommendations: {total_recs} total, {recent_recs} today")
        
        # Check backtest results
        backtest = db.backtest_results
        total_backtests = backtest.count_documents({})
        print(f"📈 Backtest Results: {total_backtests} entries")
        
        return True, db
        
    except Exception as e:
        print(f"❌ MongoDB: Connection failed - {e}")
        return False, None

def show_latest_recommendations(db, limit=5):
    """Show latest stock recommendations."""
    print(f"\n📋 Latest {limit} Recommendations")
    print("=" * 50)
    
    try:
        recommendations = db.recommended_shares.find(
            {'is_recommended': True}
        ).sort('recommendation_date', -1).limit(limit)
        
        count = 0
        for rec in recommendations:
            count += 1
            symbol = rec.get('symbol', 'Unknown')
            company = rec.get('company_name', 'Unknown Company')
            tech_score = rec.get('technical_score', 0)
            fund_score = rec.get('fundamental_score', 0)
            sent_score = rec.get('sentiment_score', 0)
            combined = rec.get('combined_score', 0)
            strength = rec.get('recommendation_strength', 'UNKNOWN')
            
            # Trade plan info
            trade_plan = rec.get('trade_plan', {})
            buy_price = trade_plan.get('buy_price', rec.get('buy_price', 0))
            sell_price = trade_plan.get('sell_price', rec.get('sell_price', 0))
            
            # Backtest info
            backtest = rec.get('backtest_metrics', {})
            cagr = backtest.get('cagr', 0)
            win_rate = backtest.get('win_rate', 0)
            
            print(f"\n{count}. {symbol} - {company[:30]}...")
            print(f"   Strength: {strength}")
            print(f"   Scores: Tech={tech_score:.2f}, Fund={fund_score:.2f}, Sent={sent_score:.2f}")
            print(f"   Combined: {combined:.2f}")
            print(f"   Trade: Buy@₹{buy_price:.2f} → Sell@₹{sell_price:.2f}")
            if cagr:
                print(f"   Backtest: CAGR={cagr:.1f}%, Win Rate={win_rate:.1f}%")
            print(f"   Date: {rec.get('recommendation_date', 'Unknown').strftime('%Y-%m-%d %H:%M') if rec.get('recommendation_date') else 'Unknown'}")
        
        if count == 0:
            print("No recommendations found.")
            
    except Exception as e:
        print(f"Error fetching recommendations: {e}")

def show_config_status():
    """Show current configuration status."""
    print(f"\n⚙️  Configuration Status")
    print("=" * 50)
    
    try:
        from config import MAX_WORKER_THREADS, ANALYSIS_CONFIG, STRATEGY_CONFIG
        
        print(f"🧵 Max Worker Threads: {MAX_WORKER_THREADS}")
        
        print(f"\n📊 Analysis Modules:")
        enabled_modules = [k for k, v in ANALYSIS_CONFIG.items() if v]
        disabled_modules = [k for k, v in ANALYSIS_CONFIG.items() if not v]
        
        print(f"   ✅ Enabled ({len(enabled_modules)}): {', '.join(enabled_modules)}")
        print(f"   ❌ Disabled ({len(disabled_modules)}): {', '.join(disabled_modules)}")
        
        print(f"\n🎯 Trading Strategies:")
        enabled_strategies = [k for k, v in STRATEGY_CONFIG.items() if v]
        print(f"   ✅ Enabled: {len(enabled_strategies)} strategies")
        
    except Exception as e:
        print(f"Error reading config: {e}")

def main():
    """Main function."""
    print("🚀 Super Advice Analysis System")
    print("=" * 50)
    
    # Check system status
    mongodb_ok, db = check_system_status()
    
    if mongodb_ok:
        show_latest_recommendations(db)
    
    show_config_status()
    
    print(f"\n✨ System Check Complete!")
    
    # Show next steps
    print(f"\n🎯 Next Steps:")
    print("1. Run test analysis: python run_analysis.py --test")
    print("2. Run full analysis: python run_analysis.py --max-stocks 50")
    print("3. View web interface: python app.py")
    print("4. Check this status: python check_results.py")

if __name__ == "__main__":
    main()
