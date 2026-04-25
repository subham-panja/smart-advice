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
                    expectancy=metrics.get('avg_expectancy'),
                    profit_factor=metrics.get('avg_profit_factor'),
                    recovery_factor=metrics.get('avg_recovery_factor'),
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
        metrics = {
            'cagr': 0.0, 'win_rate': 0.0, 'max_drawdown': 0.0,
            'expectancy': 0.0, 'profit_factor': 0.0, 'recovery_factor': 0.0
        }
        backtest = analysis_result.get('backtest_results', analysis_result.get('backtest', {}))
        if backtest and backtest.get('status') == 'completed':
            m = backtest.get('combined_metrics', backtest.get('overall_metrics', {}))
            metrics['cagr'] = m.get('avg_cagr', 0) or m.get('average_cagr', 0)
            metrics['win_rate'] = m.get('avg_win_rate', 0) or m.get('average_win_rate', 0)
            metrics['max_drawdown'] = m.get('avg_max_drawdown', 0) or m.get('average_max_drawdown', 0)
            metrics['expectancy'] = m.get('avg_expectancy', 0.0)
            metrics['profit_factor'] = m.get('avg_profit_factor', 0.0)
            metrics['recovery_factor'] = m.get('avg_recovery_factor', 0.0)
        return metrics

    def extract_backtest_cagr(self, analysis_result: Dict[str, Any]) -> str:
        """Extract CAGR for logging."""
        m = self._extract_detailed_backtest_metrics(analysis_result)
        return f"{m['cagr']:.2f}"

    # ─── NEW MULTI-STAGE PERSISTENCE METHODS ────────────────────────────────

    def save_analysis_snapshot(self, analysis_result: Dict[str, Any], scan_run_id=None) -> bool:
        """Save analysis snapshot for EVERY stock (pass or fail) for debugging."""
        try:
            db = get_mongodb()
            
            # Extract strategy signals detail
            strategy_signals = {}
            detailed = analysis_result.get('detailed_analysis', {})
            tech_detail = detailed.get('technical', {})
            individual_strategies = tech_detail.get('individual_strategies', {})
            
            for name, info in individual_strategies.items():
                strategy_signals[name] = {
                    'signal': info.get('signal', 0),
                    'type': info.get('type', 'UNKNOWN')
                }
            
            decision_debug = analysis_result.get('decision_debug', {})
            
            # Price snapshot from latest data
            price_snapshot = {}
            if 'price_data' in analysis_result:
                pd = analysis_result['price_data']
                price_snapshot = {
                    'close': pd.get('close'),
                    'sma_200': pd.get('sma_200'),
                    'ema_21': pd.get('ema_21'),
                    'rsi_14': pd.get('rsi_14'),
                    'adx_14': pd.get('adx_14'),
                    'atr_14': pd.get('atr_14'),
                    'volume': pd.get('volume'),
                    'volume_avg_20': pd.get('volume_avg_20'),
                }
            
            doc = {
                'symbol': analysis_result.get('symbol', 'UNKNOWN'),
                'company_name': analysis_result.get('company_name', ''),
                'scan_run_id': scan_run_id,
                'analyzed_at': datetime.utcnow(),
                'technical_score': analysis_result.get('technical_score', 0),
                'combined_score': analysis_result.get('combined_score', 0),
                'recommendation': analysis_result.get('recommendation_strength', 'NO_SIGNAL'),
                'is_recommended': analysis_result.get('is_recommended', False),
                'hold_reasons': decision_debug.get('hold_reasons', []),
                'strategy_signals': strategy_signals,
                'positive_signals': sum(1 for s in strategy_signals.values() if s.get('signal', 0) > 0),
                'total_signals': len(strategy_signals),
                'price_snapshot': price_snapshot,
            }
            
            db.analysis_snapshots.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving analysis snapshot: {e}")
            return False

    def save_swing_gate_results(self, symbol: str, gate_results: Dict[str, Any], scan_run_id=None) -> bool:
        """Save swing gate pass/fail results for debugging."""
        try:
            db = get_mongodb()
            
            doc = {
                'symbol': symbol,
                'scan_run_id': scan_run_id,
                'analyzed_at': datetime.utcnow(),
                'all_gates_passed': gate_results.get('all_gates_passed', False),
                'gate_1_trend': gate_results.get('gate_1_trend', {}),
                'gate_2_mtf': gate_results.get('gate_2_mtf', {}),
                'gate_3_volatility': gate_results.get('gate_3_volatility', {}),
                'gate_4_volume': gate_results.get('gate_4_volume', {}),
            }
            
            db.swing_gate_results.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving swing gate results: {e}")
            return False

    def save_trade_signal(self, symbol: str, signal_data: Dict[str, Any], scan_run_id=None) -> bool:
        """Save trade signal with entry/exit levels for stocks that pass all gates."""
        try:
            db = get_mongodb()
            
            doc = {
                'symbol': symbol,
                'scan_run_id': scan_run_id,
                'signal_date': datetime.utcnow(),
                'status': 'ACTIVE',
                'entry_price': signal_data.get('entry_price'),
                'entry_pattern': signal_data.get('entry_pattern', 'unknown'),
                'pattern_strength': signal_data.get('pattern_strength', 0),
                'signal_strength': signal_data.get('signal_strength', 0),
                'patterns': signal_data.get('patterns', {}),
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit_1': signal_data.get('take_profit_1'),
                'take_profit_2': signal_data.get('take_profit_2'),
                'trailing_stop_distance': signal_data.get('trailing_stop_distance'),
                'time_stop_bars': signal_data.get('time_stop_bars', 15),
                'risk_per_share': signal_data.get('risk_per_share'),
                'risk_reward_1': signal_data.get('risk_reward_1'),
                'risk_reward_2': signal_data.get('risk_reward_2'),
                'atr': signal_data.get('atr'),
                'bars_since_entry': 0,
                'current_pnl_pct': 0.0,
                'tp1_hit': False,
                'highest_since_entry': signal_data.get('entry_price'),
            }
            
            db.trade_signals.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Error saving trade signal: {e}")
            return False

    def create_scan_run(self, config_snapshot: Dict[str, Any], macro_regime: Dict[str, Any] = None) -> Any:
        """Create a scan run document and return its ID."""
        try:
            db = get_mongodb()
            
            doc = {
                'started_at': datetime.utcnow(),
                'completed_at': None,
                'duration_seconds': None,
                'config_snapshot': config_snapshot,
                'macro_regime': macro_regime or {},
                'results_summary': {},
            }
            
            result = db.scan_runs.insert_one(doc)
            return result.inserted_id
        except Exception as e:
            logger.error(f"Error creating scan run: {e}")
            return None

    def complete_scan_run(self, scan_run_id, summary: Dict[str, Any]) -> bool:
        """Update scan run with completion data."""
        try:
            db = get_mongodb()
            
            db.scan_runs.update_one(
                {'_id': scan_run_id},
                {'$set': {
                    'completed_at': datetime.utcnow(),
                    'duration_seconds': summary.get('duration_seconds', 0),
                    'results_summary': summary,
                }}
            )
            return True
        except Exception as e:
            logger.error(f"Error completing scan run: {e}")
            return False

