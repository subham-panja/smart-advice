import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List
from models.recommendation import RecommendedShare
from database import get_mongodb, insert_backtest_result

logger = logging.getLogger(__name__)

class PersistenceHandler:
    """Handles all database persistence for stock analysis results."""
    
    def __init__(self, app):
        self.app = app

    def clear_old_data(self, days_old: int = 7):
        """Clear old data (recommendations and backtest results) older than specified days."""
        try:
            # get_mongodb() creates its own direct MongoClient - no Flask context needed
            db = get_mongodb()

            if days_old == 0:
                logger.info("Purging ALL data from database (days_old=0)")
                r1 = db['recommended_shares'].delete_many({})
                r2 = db['backtest_results'].delete_many({})
                logger.info(f"Deleted {r1.deleted_count} recommendations and {r2.deleted_count} backtest records")
            else:
                logger.info(f"Purging data older than {days_old} days")
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_old)
                r1 = db['recommended_shares'].delete_many({'recommendation_date': {'$lt': cutoff_date}})
                r2 = db['backtest_results'].delete_many({'created_at': {'$lt': cutoff_date}})
                logger.info(f"Deleted {r1.deleted_count} recommendations and {r2.deleted_count} backtest records older than {days_old} days")

        except Exception as e:
            logger.error(f"Error clearing old data: {e}")

    def save_recommendation(self, analysis_result: Dict[str, Any]) -> bool:
        """Save analysis result to the database (only BUY recommendations, not HOLD)."""
        try:
            recommendation_strength = analysis_result.get('recommendation_strength', 'HOLD')
            is_recommended = analysis_result.get('is_recommended', False)
            
            if recommendation_strength == 'HOLD' or not is_recommended:
                logger.info(f"Skipping {recommendation_strength} for {analysis_result.get('symbol', 'UNKNOWN')}")
                return True
            
            valid_buy_recommendations = ['STRONG_BUY', 'BUY']
            if recommendation_strength not in valid_buy_recommendations:
                return True
            
            rec = RecommendedShare(
                symbol=analysis_result['symbol'],
                company_name=analysis_result['company_name'],
                technical_score=analysis_result['technical_score'],
                fundamental_score=analysis_result['fundamental_score'],
                sentiment_score=analysis_result['sentiment_score'],
                reason=analysis_result['reason']
            )
            
            trade_plan = analysis_result.get('trade_plan', {})
            buy_price = trade_plan.get('buy_price', 0.0) if trade_plan else analysis_result.get('buy_price', 0.0)
            sell_price = trade_plan.get('sell_price', 0.0) if trade_plan else analysis_result.get('sell_price', 0.0)
            
            backtest_metrics = self._extract_detailed_backtest_metrics(analysis_result)
            
            db = get_mongodb()
            doc = {
                'symbol': rec.symbol,
                'company_name': rec.company_name,
                'technical_score': rec.technical_score,
                'fundamental_score': rec.fundamental_score,
                'sentiment_score': rec.sentiment_score,
                'combined_score': analysis_result.get('combined_score', 0.0),
                'is_recommended': is_recommended,
                'recommendation_strength': recommendation_strength,
                'reason': rec.reason,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'backtest_metrics': backtest_metrics,
                'recommendation_date': datetime.utcnow(),
                # Include other detailed analysis fields...
                'detailed_analysis': analysis_result.get('detailed_analysis', {}),
                'market_regime': analysis_result.get('market_regime', {})
            }
            
            db.recommended_shares.update_one({'symbol': rec.symbol}, {'$set': doc}, upsert=True)
            return True
        except Exception as e:
            logger.error(f"Error saving recommendation: {e}")
            return False

    def save_backtest_results(self, analysis_result: Dict[str, Any]) -> bool:
        """Save backtest results to the database."""
        try:
            backtest_results = analysis_result.get('backtest_results', analysis_result.get('backtest', {}))
            if not backtest_results or backtest_results.get('status') != 'completed':
                return False
            
            symbol = analysis_result['symbol']
            metrics = backtest_results.get('combined_metrics', backtest_results.get('overall_metrics', {}))
            
            if not metrics:
                return False
            
            if not self._check_existing_backtest_result(symbol, 'Overall'):
                insert_backtest_result(
                    symbol, 'Overall',
                    metrics.get('avg_cagr', 0),
                    metrics.get('avg_win_rate', 0),
                    metrics.get('avg_max_drawdown', 0),
                    total_trades=metrics.get('total_trades'),
                    winning_trades=metrics.get('winning_trades'),
                    losing_trades=metrics.get('losing_trades'),
                    avg_profit_per_trade=metrics.get('avg_profit_per_trade'),
                    avg_loss_per_trade=metrics.get('avg_loss_per_trade'),
                    avg_trade_duration=metrics.get('avg_trade_duration'),
                    largest_win=metrics.get('largest_win'),
                    largest_loss=metrics.get('largest_loss'),
                    sharpe_ratio=metrics.get('avg_sharpe_ratio'),
                    sortino_ratio=metrics.get('avg_sortino_ratio'),
                    calmar_ratio=metrics.get('avg_calmar_ratio'),
                    volatility=metrics.get('avg_volatility'),
                    start_date=metrics.get('start_date'),
                    end_date=metrics.get('end_date'),
                    initial_capital=metrics.get('initial_capital'),
                    final_capital=metrics.get('avg_final_value'),
                    total_return=metrics.get('avg_roi'),
                )
                logger.info(f"Saved detailed backtest results for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
            return False

    def _check_existing_backtest_result(self, symbol: str, period: str) -> bool:
        """Check if backtest result exists for today."""
        try:
            db = get_mongodb()
            today = datetime.now(timezone.utc).date()
            query = {
                'symbol': symbol,
                'period': period,
                '$expr': {'$eq': [{'$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}}, today.strftime('%Y-%m-%d')]}
            }
            return db.backtest_results.count_documents(query) > 0
        except Exception:
            return False

    def _extract_detailed_backtest_metrics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified extraction of backtest metrics."""
        # Implementation similar to run_analysis.py but more compact
        metrics = {'cagr': 0.0, 'win_rate': 0.0, 'max_drawdown': 0.0}
        backtest = analysis_result.get('backtest_results', analysis_result.get('backtest', {}))
        if backtest and backtest.get('status') == 'completed':
            m = backtest.get('combined_metrics', backtest.get('overall_metrics', {}))
            metrics['cagr'] = m.get('avg_cagr', 0) or m.get('average_cagr', 0)
            metrics['win_rate'] = m.get('avg_win_rate', 0) or m.get('average_win_rate', 0)
            metrics['max_drawdown'] = m.get('avg_max_drawdown', 0) or m.get('average_max_drawdown', 0)
        return metrics

    def extract_backtest_cagr(self, analysis_result: Dict[str, Any]) -> str:
        """Extract CAGR for logging."""
        m = self._extract_detailed_backtest_metrics(analysis_result)
        return f"{m['cagr']:.2f}"
