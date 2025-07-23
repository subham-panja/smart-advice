#!/usr/bin/env python3
"""
Quick fix script to regenerate ALEMBICLTD analysis with corrected strategies
"""

from scripts.analyzer import StockAnalyzer
from database import get_mongodb
import config
import json

def fix_alembic_analysis():
    """Re-run analysis for ALEMBICLTD with fixed strategies"""
    
    # Initialize analyzer
    analyzer = StockAnalyzer()
    
    # Analysis configuration
    app_config = {
        'HISTORICAL_DATA_PERIOD': '2y',
        'SKIP_SENTIMENT': True  # Skip sentiment for faster testing
    }
    
    print("Re-analyzing ALEMBICLTD with fixed strategies...")
    
    # Run fresh analysis
    result = analyzer.analyze_stock('ALEMBICLTD', app_config)
    
    print("\nUpdated Analysis Results:")
    print("=" * 60)
    print(f"Symbol: {result['symbol']}")
    print(f"Technical Score: {result['technical_score']:.4f}")
    print(f"Fundamental Score: {result['fundamental_score']:.4f}")
    print(f"Sentiment Score: {result['sentiment_score']:.4f}")
    print(f"Combined Score: {result.get('combined_score', 'N/A')}")
    print(f"Is Recommended: {result['is_recommended']}")
    print(f"Reason: {result['reason']}")
    
    # Check backtest results
    if 'backtest' in result:
        backtest = result['backtest']
        print(f"\nBacktest Status: {backtest.get('status', 'N/A')}")
        if backtest.get('status') == 'completed':
            combined_metrics = backtest.get('combined_metrics', {})
            print(f"CAGR: {combined_metrics.get('avg_cagr', 0)}%")
            print(f"Win Rate: {combined_metrics.get('avg_win_rate', 0)}%")
            print(f"Max Drawdown: {combined_metrics.get('avg_max_drawdown', 0)}%")
    
    # Check trade plan
    if 'trade_plan' in result:
        trade_plan = result['trade_plan']
        print(f"\nTrade Plan:")
        print(f"Buy Price: {trade_plan.get('buy_price', 0)}")
        print(f"Sell Price: {trade_plan.get('sell_price', 0)}")
        print(f"Days to Target: {trade_plan.get('days_to_target', 0)}")
        print(f"Risk/Reward: {trade_plan.get('risk_reward_ratio', 0)}")
    
    # Update database with fresh results
    try:
        db = get_mongodb()
        collection = db[config.MONGODB_COLLECTIONS['recommended_shares']]
        
        # Remove old entry
        collection.delete_one({'symbol': 'ALEMBICLTD'})
        
        # Insert updated entry if recommended
        if result['is_recommended']:
            from models.recommendation import RecommendedShare
            from datetime import datetime
            
            # Create new recommendation object
            recommendation = RecommendedShare(
                symbol=result['symbol'],
                company_name=result['company_name'],
                technical_score=result['technical_score'],
                fundamental_score=result['fundamental_score'],
                sentiment_score=result['sentiment_score'],
                reason=result['reason'],
                buy_price=result.get('trade_plan', {}).get('buy_price', 0.0),
                sell_price=result.get('trade_plan', {}).get('sell_price', 0.0),
                est_time_to_target=f"{result.get('trade_plan', {}).get('days_to_target', 0)} days"
            )
            
            # Add backtest metrics if available
            if result.get('backtest', {}).get('status') == 'completed':
                combined_metrics = result['backtest']['combined_metrics']
                recommendation.backtest_metrics = {
                    'cagr': combined_metrics.get('avg_cagr', 0),
                    'win_rate': combined_metrics.get('avg_win_rate', 0),
                    'max_drawdown': combined_metrics.get('avg_max_drawdown', 0),
                    'total_trades': combined_metrics.get('strategies_tested', 0),
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'sharpe_ratio': combined_metrics.get('avg_sharpe_ratio', 0),
                    'effectiveness': 'Good' if combined_metrics.get('avg_cagr', 0) > 5 else 'Poor',
                    'buy_sell_transactions': [],
                    'strategy_breakdown': {},
                    'date_range': {
                        'start_date': '',
                        'end_date': '',
                        'period_days': 0
                    },
                    'capital_info': {
                        'initial_capital': 100000,
                        'final_capital': 100000 * (1 + combined_metrics.get('avg_cagr', 0)/100),
                        'total_return': combined_metrics.get('avg_cagr', 0)
                    }
                }
            
            # Save to database
            recommendation.save()
            print(f"\n✅ Updated ALEMBICLTD recommendation in database")
        else:
            print(f"\n❌ ALEMBICLTD not recommended after fixes")
    
    except Exception as e:
        print(f"\n⚠️ Error updating database: {e}")
    
    return result


if __name__ == "__main__":
    result = fix_alembic_analysis()
