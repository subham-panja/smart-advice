#!/usr/bin/env python3
"""
Script to check recommendations from the analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import get_mongodb
from datetime import datetime

def main():
    try:
        db = get_mongodb()
        recommendations = list(db.recommended_shares.find({}).sort('recommendation_date', -1))
        
        print(f'Total recommendations found: {len(recommendations)}')
        print()
        
        if recommendations:
            print('Top 10 Recent Recommendations:')
            print('=' * 80)
            
            for i, rec in enumerate(recommendations[:10]):
                print(f"{i+1}. {rec.get('symbol', 'N/A')} ({rec.get('company_name', 'N/A')})")
                print(f"   Technical Score: {rec.get('technical_score', 0):.2f}")
                print(f"   Fundamental Score: {rec.get('fundamental_score', 0):.2f}")
                print(f"   Sentiment Score: {rec.get('sentiment_score', 0):.2f}")
                print(f"   Buy Price: ${rec.get('buy_price', 0):.2f}")
                print(f"   Sell Price: ${rec.get('sell_price', 0):.2f}")
                print(f"   Est. Time to Target: {rec.get('est_time_to_target', 'N/A')}")
                
                backtest = rec.get('backtest_metrics', {})
                if backtest:
                    print(f"   Backtest CAGR: {backtest.get('cagr', 0):.2f}%")
                    print(f"   Backtest Win Rate: {backtest.get('win_rate', 0):.2f}%")
                    print(f"   Backtest Max Drawdown: {backtest.get('max_drawdown', 0):.2f}%")
                    print(f"   Effectiveness: {backtest.get('effectiveness', 'Unknown')}")
                    print(f"   Total Trades: {backtest.get('total_trades', 0)}")
                
                print(f"   Reason: {rec.get('reason', 'N/A')}")
                
                rec_date = rec.get('recommendation_date')
                if rec_date:
                    print(f"   Date: {rec_date.strftime('%Y-%m-%d %H:%M')}")
                print()
        else:
            print('No recommendations found in the database.')
            
        # Also check backtest results
        print("=" * 80)
        print("BACKTEST RESULTS SUMMARY:")
        print("=" * 80)
        
        backtest_results = list(db.backtest_results.find({}).sort('created_at', -1).limit(20))
        print(f"Total backtest results: {len(backtest_results)}")
        
        if backtest_results:
            print("\nTop 10 Backtest Results:")
            for i, result in enumerate(backtest_results[:10]):
                print(f"{i+1}. {result.get('symbol', 'N/A')} - {result.get('strategy', 'Overall')}")
                print(f"   CAGR: {result.get('cagr', 0):.2f}%")
                print(f"   Win Rate: {result.get('win_rate', 0):.2f}%")
                print(f"   Max Drawdown: {result.get('max_drawdown', 0):.2f}%")
                print()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
