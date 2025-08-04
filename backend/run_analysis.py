#!/usr/bin/env python3
"""
Automated Stock Analysis Script
File: run_analysis.py

This script automatically analyzes all NSE stocks and saves recommendations to the database.
Designed to be run via cron job every hour.
"""

# Fix OpenMP/threading issues on macOS - MUST be set before importing numpy/scipy/sklearn
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import gc
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from scripts.data_fetcher import get_all_nse_symbols, get_filtered_nse_symbols, get_offline_symbols_from_cache
from scripts.analyzer import StockAnalyzer
from models.recommendation import RecommendedShare
from database import query_mongodb, get_mongodb, insert_backtest_result, init_db, close_db
from utils.logger import setup_logging
from utils.cache_manager import get_cache_manager
from config import MAX_WORKER_THREADS, BATCH_SIZE, REQUEST_DELAY

# Initialize logger variable (will be configured based on verbose flag in AutomatedStockAnalysis)
logger = None

class AutomatedStockAnalysis:
    """Main class for automated stock analysis."""
    
    def __init__(self, verbose=False):
        """Initialize the analyzer."""
        self.app = create_app()
        
        # Reconfigure logging based on verbose flag BEFORE creating analyzer
        # This ensures all module loggers respect the verbose setting
        from utils.logger import setup_logging
        global logger
        logger = setup_logging(verbose=verbose)
        
        self.analyzer = StockAnalyzer()
        self.start_time = datetime.now()
        self.verbose = verbose
        self.progress_callback = None
        
    def clear_old_data(self, days_old: int = 7):
        """Clear old data (recommendations and backtest results) older than specified days.
        
        Args:
            days_old: Number of days to keep. If 0, removes all data.
        """
        with self.app.app_context():
            try:
                db = get_mongodb()
                from datetime import datetime, timedelta
                
                # Add timeout to prevent hanging
                db.client.server_info()  # Test connection
                
                if days_old == 0:
                    # Remove all data if days_old is 0
                    logger.info("Purging ALL data from database (days_old=0)")
                    
                    # Delete all recommendations with timeout
                    recommendations_collection = db['recommended_shares']
                    rec_result = recommendations_collection.delete_many({}, maxTimeMS=30000)  # 30 second timeout
                    deleted_recommendations = rec_result.deleted_count
                    
                    # Delete all backtest results with timeout
                    backtest_collection = db['backtest_results']
                    backtest_result = backtest_collection.delete_many({}, maxTimeMS=30000)  # 30 second timeout
                    deleted_backtest_results = backtest_result.deleted_count
                    
                    logger.info(f"Complete data purge: {deleted_recommendations} recommendations and {deleted_backtest_results} backtest results deleted (ALL DATA)")
                    
                else:
                    # Delete data older than specified days
                    logger.info(f"Purging data older than {days_old} days")
                    
                    # Use timezone-aware datetime to fix deprecation warning
                    cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=days_old)
                    
                    # Delete old recommendations with timeout
                    recommendations_collection = db['recommended_shares']
                    rec_result = recommendations_collection.delete_many({
                        'recommendation_date': {'$lt': cutoff_date}
                    }, maxTimeMS=30000)  # 30 second timeout
                    deleted_recommendations = rec_result.deleted_count
                    
                    # Delete old backtest results with timeout
                    backtest_collection = db['backtest_results']
                    backtest_result = backtest_collection.delete_many({
                        'created_at': {'$lt': cutoff_date}
                    }, maxTimeMS=30000)  # 30 second timeout
                    deleted_backtest_results = backtest_result.deleted_count
                    
                    logger.info(f"Data purge completed: {deleted_recommendations} recommendations and {deleted_backtest_results} backtest results deleted (older than {days_old} days)")
                
                logger.debug("Database purge completed successfully")
                
            except Exception as e:
                logger.error(f"Error clearing old data: {e}")
                logger.warning("Continuing without data purge - this may be due to database connectivity issues")
                # Don't raise the exception - allow the script to continue
    
    def save_recommendation(self, analysis_result: Dict[str, Any]) -> bool:
        """Save analysis result to the database (only BUY recommendations, not HOLD)."""
        try:
            # Filter out HOLD recommendations - we only want BUY recommendations
            # Multiple checks to ensure no HOLD recommendations get through
            recommendation_strength = analysis_result.get('recommendation_strength', 'HOLD')
            is_recommended = analysis_result.get('is_recommended', False)
            
            # CRITICAL SAFETY CHECK: Never save BUY recommendations with negative combined scores
            combined_score = analysis_result.get('combined_score', 0.0)
            if combined_score < 0 and recommendation_strength in ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'OPPORTUNISTIC_BUY']:
                logger.warning(f"SAFETY CHECK: Blocking BUY recommendation for {analysis_result.get('symbol', 'UNKNOWN')} with negative combined score ({combined_score:.4f})")
                return True  # Block this recommendation
            
            # Skip if recommendation strength is HOLD
            if recommendation_strength == 'HOLD':
                logger.info(f"Skipping HOLD recommendation for {analysis_result.get('symbol', 'UNKNOWN')}")
                return True  # Return True as this is expected behavior, not an error
            
            # Skip if is_recommended is False (additional safety check)
            if not is_recommended:
                logger.info(f"Skipping non-recommended stock {analysis_result.get('symbol', 'UNKNOWN')} (is_recommended=False)")
                return True
            
            # Only proceed with valid BUY recommendations
            valid_buy_recommendations = ['STRONG_BUY', 'BUY', 'WEAK_BUY', 'OPPORTUNISTIC_BUY']
            if recommendation_strength not in valid_buy_recommendations:
                logger.info(f"Skipping invalid recommendation strength '{recommendation_strength}' for {analysis_result.get('symbol', 'UNKNOWN')}")
                return True
            
            # Create RecommendedShare object
            rec = RecommendedShare(
                symbol=analysis_result['symbol'],
                company_name=analysis_result['company_name'],
                technical_score=analysis_result['technical_score'],
                fundamental_score=analysis_result['fundamental_score'],
                sentiment_score=analysis_result['sentiment_score'],
                reason=analysis_result['reason']
            )
            
            # Extract trade plan data with improved handling
            trade_plan = analysis_result.get('trade_plan', {})
            
            # Get trade-level fields from trade_plan with fallback to legacy fields
            buy_price = None
            sell_price = None
            est_time_to_target = "Unknown"
            
            if trade_plan and not trade_plan.get('error'):
                # Primary source: trade_plan data
                buy_price = trade_plan.get('buy_price')
                sell_price = trade_plan.get('sell_price')
                days_to_target = trade_plan.get('days_to_target', 0)
                
                # Handle None values and convert to appropriate types
                if buy_price is not None:
                    buy_price = float(buy_price)
                else:
                    buy_price = 0.0
                    
                if sell_price is not None:
                    sell_price = float(sell_price)
                else:
                    sell_price = 0.0
                    
                # Format estimated time to target
                if days_to_target and days_to_target > 0:
                    est_time_to_target = f"{int(days_to_target)} days"
                elif days_to_target == 0:
                    est_time_to_target = "Immediate"
                else:
                    est_time_to_target = "Unknown"
            else:
                # Fallback to legacy columns or defaults for backward compatibility
                buy_price = analysis_result.get('buy_price', 0.0)
                sell_price = analysis_result.get('sell_price', 0.0)
                est_time_to_target = analysis_result.get('est_time_to_target', "Unknown")
                
                # Log when falling back to legacy fields
                if buy_price != 0.0 or sell_price != 0.0:
                    logger.info(f"Using legacy trade fields for {rec.symbol}: buy_price={buy_price}, sell_price={sell_price}")
            
            # Extract backtest metrics including detailed transaction data
            backtest_metrics = self._extract_detailed_backtest_metrics(analysis_result)
            
            # Calculate expected return percentage if both buy and sell prices are available
            expected_return_percent = 0.0
            if buy_price and sell_price and buy_price > 0:
                expected_return_percent = ((sell_price - buy_price) / buy_price) * 100
            
            # Log the trade-level values being stored
            logger.info(f"Storing trade-level data for {rec.symbol}: buy_price={buy_price}, sell_price={sell_price}, expected_return_percent={expected_return_percent:.2f}%, est_time_to_target={est_time_to_target}")
            
            # Log backtest metrics
            if backtest_metrics.get('cagr'):
                logger.info(f"Backtest metrics for {rec.symbol}: CAGR={backtest_metrics['cagr']:.2f}%, "
                           f"Win Rate={backtest_metrics['win_rate']:.2f}%, "
                           f"Max Drawdown={backtest_metrics['max_drawdown']:.2f}%, "
                           f"Total Trades={backtest_metrics.get('total_trades', 0)}")
            
            # Use MongoDB upsert to insert or update
            from database import get_mongodb
            db = get_mongodb()
            try:
                from datetime import datetime
                
                # Prepare document for MongoDB
                doc = {
                    'symbol': rec.symbol,
                    'company_name': rec.company_name,
                    'technical_score': rec.technical_score,
                    'fundamental_score': rec.fundamental_score,
                    'sentiment_score': rec.sentiment_score,
                    'combined_score': analysis_result.get('combined_score', 0.0),
                    'is_recommended': analysis_result.get('is_recommended', False),
                    'recommendation_strength': analysis_result.get('recommendation_strength', 'HOLD'),
                    'reason': rec.reason,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'est_time_to_target': est_time_to_target,
                    'backtest_metrics': backtest_metrics,
                    'recommendation_date': datetime.utcnow(),
                    'expected_return_percent': expected_return_percent,
                    'detailed_analysis': analysis_result.get('detailed_analysis', {}),
                    'sector_analysis': analysis_result.get('sector_analysis', {}),
                    'market_regime': analysis_result.get('market_regime', {}),
                    'market_microstructure': analysis_result.get('market_microstructure', {}),
                    'alternative_data': analysis_result.get('alternative_data', {}),
                    'prediction': analysis_result.get('prediction', {}),
                    'rl_action': analysis_result.get('rl_action', {}),
                    'tca_analysis': analysis_result.get('tca_analysis', {})
            }
                
                # Use upsert to insert or update
                result = db.recommended_shares.update_one(
                    {'symbol': rec.symbol},
                    {'$set': doc},
                    upsert=True
                )
                
                # Get backtest CAGR for logging
                backtest_cagr = self._extract_backtest_cagr(analysis_result)
                
                if result.upserted_id:
                    logger.info(f"Added new recommendation: {rec.symbol} - buy_price=${buy_price:.2f}, sell_price=${sell_price:.2f}, ETA={est_time_to_target}, backtest_CAGR={backtest_cagr}%")
                else:
                    logger.info(f"Updated existing recommendation: {rec.symbol} - buy_price=${buy_price:.2f}, sell_price=${sell_price:.2f}, ETA={est_time_to_target}, backtest_CAGR={backtest_cagr}%")
                
                logger.debug(f"Database write successful for {rec.symbol}")
                
                # Save backtest results if available
                self.save_backtest_results(analysis_result)
                
                return True
                
            except Exception as e:
                logger.error(f"Database error saving recommendation for {rec.symbol}: {e}")
                return False
                
        except Exception as e:
            logger.exception(f"Unexpected error saving recommendation for {analysis_result.get('symbol', 'UNKNOWN')}: {e}")
            return False
    
    def _extract_backtest_cagr(self, analysis_result: Dict[str, Any]) -> str:
        """Extract backtest CAGR from analysis results for logging."""
        try:
            # Try to get from backtest results
            backtest_results = analysis_result.get('backtest_results', {})
            if backtest_results and 'error' not in backtest_results:
                overall_metrics = backtest_results.get('overall_metrics', {})
                if overall_metrics:
                    avg_cagr = overall_metrics.get('average_cagr', 0)
                    return f"{avg_cagr:.2f}"
            
            # Try to get from backtest field (alternative structure)
            backtest = analysis_result.get('backtest', {})
            if backtest and backtest.get('status') == 'completed':
                combined_metrics = backtest.get('combined_metrics', {})
                if combined_metrics:
                    avg_cagr = combined_metrics.get('avg_cagr', 0)
                    return f"{avg_cagr:.2f}"
            
            return "N/A"
        except Exception as e:
            logger.error(f"Error extracting backtest CAGR: {e}")
            return "N/A"

    def _extract_detailed_backtest_metrics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed backtest metrics including buy/sell transactions."""
        try:
            # Initialize detailed metrics structure
            detailed_metrics = {
                'cagr': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'sharpe_ratio': 0.0,
                'effectiveness': 'Low',
                'buy_sell_transactions': [],
                'strategy_breakdown': {},
                'date_range': {},
                'capital_info': {}
            }
            
            # Try to get from backtest results
            backtest_results = analysis_result.get('backtest_results', {})
            if not backtest_results:
                backtest_results = analysis_result.get('backtest', {})
            
            if backtest_results and 'error' not in backtest_results and backtest_results.get('status') == 'completed':
                # Get combined metrics
                combined_metrics = backtest_results.get('combined_metrics', {})
                overall_metrics = backtest_results.get('overall_metrics', {})
                source_metrics = combined_metrics if combined_metrics else overall_metrics
                
                # Extract individual strategy results first to calculate aggregated metrics
                strategy_results = backtest_results.get('strategy_results', {})
                total_trades_sum = 0
                winning_trades_sum = 0
                losing_trades_sum = 0
                all_transactions = []
                valid_strategies = 0
                
                # Process strategy results
                for strategy_name, strategy_result in strategy_results.items():
                    if strategy_result.get('status') == 'completed':
                        valid_strategies += 1
                        strategy_trades = strategy_result.get('total_trades', 0)
                        strategy_cagr = strategy_result.get('cagr', 0)
                        strategy_win_rate = strategy_result.get('win_rate', 0)
                        
                        # Store strategy breakdown
                        detailed_metrics['strategy_breakdown'][strategy_name] = {
                            'cagr': strategy_cagr,
                            'win_rate': strategy_win_rate,
                            'max_drawdown': strategy_result.get('max_drawdown', 0),
                            'total_trades': strategy_trades,
                            'trades': strategy_result.get('trades', [])
                        }
                        
                        # Accumulate trade counts
                        total_trades_sum += strategy_trades
                        
                        # Calculate winning/losing trades from win rate and total trades
                        if strategy_trades > 0 and strategy_win_rate > 0:
                            strategy_winning_trades = int((strategy_win_rate / 100) * strategy_trades)
                            strategy_losing_trades = strategy_trades - strategy_winning_trades
                            winning_trades_sum += strategy_winning_trades
                            losing_trades_sum += strategy_losing_trades
                        
                        # Extract buy/sell transactions from strategy trades
                        trades = strategy_result.get('trades', [])
                        for trade in trades[-10:]:  # Keep last 10 trades per strategy
                            if isinstance(trade, dict):
                                transaction = {
                                    'strategy': strategy_name,
                                    'date': str(trade.get('date', '')),
                                    'action': trade.get('action', 'UNKNOWN'),
                                    'price': trade.get('price', 0),
                                    'shares': trade.get('shares', 0),
                                    'value': trade.get('value', 0)
                                }
                                all_transactions.append(transaction)
                
                # Calculate aggregated metrics
                if valid_strategies > 0:
                    # Use average trades per strategy for overall metrics
                    detailed_metrics['total_trades'] = int(total_trades_sum / valid_strategies)
                    detailed_metrics['winning_trades'] = int(winning_trades_sum / valid_strategies)
                    detailed_metrics['losing_trades'] = int(losing_trades_sum / valid_strategies)
                
                # Extract basic metrics from source_metrics
                if source_metrics:
                    detailed_metrics['cagr'] = source_metrics.get('avg_cagr', 0) or source_metrics.get('average_cagr', 0)
                    detailed_metrics['win_rate'] = source_metrics.get('avg_win_rate', 0) or source_metrics.get('average_win_rate', 0)
                    detailed_metrics['max_drawdown'] = source_metrics.get('avg_max_drawdown', 0) or source_metrics.get('average_max_drawdown', 0)
                    detailed_metrics['sharpe_ratio'] = source_metrics.get('avg_sharpe_ratio', 0) or source_metrics.get('average_sharpe_ratio', 0)
                    
                    # Use source_metrics for total_trades if it exists and is greater than calculated
                    source_total_trades = source_metrics.get('total_trades', 0)
                    if source_total_trades > detailed_metrics['total_trades']:
                        detailed_metrics['total_trades'] = source_total_trades
                    
                    # Extract date range information with fallbacks
                    start_date = source_metrics.get('start_date', '')
                    end_date = source_metrics.get('end_date', '')
                    period_days = source_metrics.get('period_days', 0)
                    
                    # If date range is empty, try to estimate from analysis context
                    if not start_date or not end_date or period_days == 0:
                        # Try to get from analysis_result context
                        symbol = analysis_result.get('symbol', '')
                        if symbol:
                            # Estimate typical backtesting period (e.g., 2 years)
                            from datetime import datetime, timedelta
                            end_date = datetime.now().strftime('%Y-%m-%d')
                            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                            period_days = 730
                    
                    detailed_metrics['date_range'] = {
                        'start_date': start_date,
                        'end_date': end_date,
                        'period_days': period_days
                    }
                    
                    # Extract capital information with proper calculations
                    initial_capital = source_metrics.get('initial_capital', 100000)
                    final_capital = source_metrics.get('final_capital', 0)
                    total_return = source_metrics.get('total_return', 0)
                    
                    # Calculate final_capital if not provided
                    if final_capital == 0 and detailed_metrics['cagr'] != 0:
                        # Calculate based on CAGR and period
                        years = period_days / 365.25 if period_days > 0 else 2.0
                        final_capital = initial_capital * ((1 + detailed_metrics['cagr'] / 100) ** years)
                    
                    # Calculate total_return if not provided
                    if total_return == 0 and final_capital > 0:
                        total_return = ((final_capital - initial_capital) / initial_capital) * 100
                    
                    detailed_metrics['capital_info'] = {
                        'initial_capital': initial_capital,
                        'final_capital': round(final_capital, 2),
                        'total_return': round(total_return, 2)
                    }
                
                # Determine effectiveness based on CAGR and win rate
                cagr = detailed_metrics['cagr']
                win_rate = detailed_metrics['win_rate']
                
                if cagr >= 15 and win_rate >= 60:
                    detailed_metrics['effectiveness'] = 'Excellent'
                elif cagr >= 10 and win_rate >= 50:
                    detailed_metrics['effectiveness'] = 'Good'
                elif cagr >= 5 and win_rate >= 45:
                    detailed_metrics['effectiveness'] = 'Moderate'
                elif cagr >= 0 and win_rate >= 40:
                    detailed_metrics['effectiveness'] = 'Fair'
                else:
                    detailed_metrics['effectiveness'] = 'Poor'
                
                # Sort and limit transactions
                detailed_metrics['buy_sell_transactions'] = sorted(
                    all_transactions,
                    key=lambda x: x.get('date', ''),
                    reverse=True
                )[:50]  # Limit to prevent database bloat
            
            return detailed_metrics
            
        except Exception as e:
            logger.error(f"Error extracting detailed backtest metrics: {e}")
            return {
                'cagr': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'effectiveness': 'Unknown',
                'error': str(e)
            }
    
    def _extract_backtest_metrics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all backtest metrics from analysis results."""
        try:
            # Initialize defaults
            metrics = {
                'backtest_cagr': None,
                'backtest_win_rate': None,
                'backtest_max_drawdown': None,
                'backtest_sharpe_ratio': None,
                'backtest_total_trades': None,
                'backtest_avg_trade_return': None
            }
            
            # Try to get from backtest results
            backtest_results = analysis_result.get('backtest_results', {})
            if not backtest_results:
                backtest_results = analysis_result.get('backtest', {})
            
            if backtest_results and 'error' not in backtest_results:
                # Get combined metrics (new structure)
                combined_metrics = backtest_results.get('combined_metrics', {})
                
                # Also check for overall_metrics (old structure) for backward compatibility
                overall_metrics = backtest_results.get('overall_metrics', {})
                
                # Use combined_metrics first, fallback to overall_metrics
                source_metrics = combined_metrics if combined_metrics else overall_metrics
                
                if source_metrics:
                    metrics['backtest_cagr'] = source_metrics.get('avg_cagr') or source_metrics.get('average_cagr')
                    metrics['backtest_win_rate'] = source_metrics.get('avg_win_rate') or source_metrics.get('average_win_rate')
                    metrics['backtest_max_drawdown'] = source_metrics.get('avg_max_drawdown') or source_metrics.get('average_max_drawdown')
                    metrics['backtest_sharpe_ratio'] = source_metrics.get('avg_sharpe_ratio') or source_metrics.get('average_sharpe_ratio')
                    
                    # For total trades, try to get from strategies results or use estimated value
                    strategy_results = backtest_results.get('strategy_results', {})
                    if strategy_results:
                        # Sum up total trades from all strategies
                        total_trades = 0
                        total_avg_return = 0
                        valid_strategies = 0
                        
                        for strategy_name, strategy_result in strategy_results.items():
                            if strategy_result.get('status') == 'completed':
                                strategy_trades = strategy_result.get('total_trades', 0)
                                strategy_avg_return = strategy_result.get('avg_trade_return', 0)
                                
                                if strategy_trades and strategy_trades > 0:
                                    total_trades += strategy_trades
                                    total_avg_return += strategy_avg_return
                                    valid_strategies += 1
                        
                        if valid_strategies > 0:
                            # Use average across strategies
                            metrics['backtest_total_trades'] = int(total_trades / valid_strategies)
                            metrics['backtest_avg_trade_return'] = total_avg_return / valid_strategies
                    
                    # If still no total trades, try to get from source_metrics
                    if not metrics['backtest_total_trades']:
                        metrics['backtest_total_trades'] = source_metrics.get('total_trades') or source_metrics.get('strategies_tested')
                    
                    if not metrics['backtest_avg_trade_return']:
                        # Calculate average trade return from CAGR and total trades if available
                        if metrics['backtest_cagr'] and metrics['backtest_total_trades']:
                            metrics['backtest_avg_trade_return'] = metrics['backtest_cagr'] / max(1, metrics['backtest_total_trades'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error extracting backtest metrics: {e}")
            return {
                'backtest_cagr': None,
                'backtest_win_rate': None,
                'backtest_max_drawdown': None,
                'backtest_sharpe_ratio': None,
                'backtest_total_trades': None,
                'backtest_avg_trade_return': None
            }
    
    def _check_existing_backtest_result(self, symbol: str, period: str) -> bool:
        """Check if backtest result for (symbol, period) already exists today."""
        try:
            from datetime import datetime, timezone
            db = get_mongodb()
            
            # Get today's date in UTC
            today = datetime.now(timezone.utc).date()
            
            # Query for existing backtest result created today
            query = {
                'symbol': symbol,
                'period': period,
                '$expr': {
                    '$eq': [
                        {'$dateToString': {'format': '%Y-%m-%d', 'date': '$created_at'}},
                        today.strftime('%Y-%m-%d')
                    ]
                }
            }
            
            count = db.backtest_results.count_documents(query)
            return count > 0
        except Exception as e:
            logger.error(f"Error checking existing backtest result for {symbol}-{period}: {e}")
            return False
    
    def save_backtest_results(self, analysis_result: Dict[str, Any]) -> bool:
        """Save backtest results to the database."""
        try:
            # Try both 'backtest_results' and 'backtest' keys for compatibility
            backtest_results = analysis_result.get('backtest_results', {})
            if not backtest_results:
                backtest_results = analysis_result.get('backtest', {})
            
            if not backtest_results or 'error' in backtest_results:
                logger.debug(f"No valid backtest results found for {analysis_result.get('symbol', 'UNKNOWN')}")
                return False
            
            symbol = analysis_result['symbol']
            
            # Check if backtest completed successfully
            if backtest_results.get('status') != 'completed':
                logger.debug(f"Backtest not completed for {symbol}: {backtest_results.get('status', 'unknown')}")
                return False
            
            # Get combined metrics (new structure)
            combined_metrics = backtest_results.get('combined_metrics', {})
            
            # Also check for overall_metrics (old structure) for backward compatibility
            overall_metrics = backtest_results.get('overall_metrics', {})
            
            if not combined_metrics and not overall_metrics:
                logger.debug(f"No metrics found in backtest results for {symbol}")
                return False
            
            # Save overall backtest metrics
            try:
                if not self._check_existing_backtest_result(symbol, 'Overall'):
                    # Use combined_metrics first, fallback to overall_metrics
                    metrics = combined_metrics if combined_metrics else overall_metrics
                    
                    cagr = metrics.get('avg_cagr', 0) or metrics.get('average_cagr', 0)
                    win_rate = metrics.get('avg_win_rate', 0) or metrics.get('average_win_rate', 0)
                    max_drawdown = metrics.get('avg_max_drawdown', 0) or metrics.get('average_max_drawdown', 0)
                    
                insert_backtest_result(
                    symbol, 'Overall', 
                    cagr,
                    win_rate,
                    max_drawdown,
                    total_trades=metrics.get('total_trades'),
                    winning_trades=metrics.get('winning_trades'),
                    losing_trades=metrics.get('losing_trades'),
                    avg_trade_duration=metrics.get('avg_trade_duration'),
                    avg_profit_per_trade=metrics.get('avg_profit_per_trade'),
                    avg_loss_per_trade=metrics.get('avg_loss_per_trade'),
                    largest_win=metrics.get('largest_win'),
                    largest_loss=metrics.get('largest_loss'),
                    sharpe_ratio=metrics.get('sharpe_ratio'),
                    sortino_ratio=metrics.get('sortino_ratio'),
                    calmar_ratio=metrics.get('calmar_ratio'),
                    volatility=metrics.get('volatility'),
                    start_date=metrics.get('start_date'),
                    end_date=metrics.get('end_date'),
                    initial_capital=metrics.get('initial_capital'),
                    final_capital=metrics.get('final_capital'),
                    total_return=metrics.get('total_return')
                )
                logger.info(f"Saved overall backtest results for {symbol}: CAGR={cagr:.2f}%, Win Rate={win_rate:.2f}%, Max Drawdown={max_drawdown:.2f}%")

                # Save individual period results if available
                period_results = backtest_results.get('period_results', {})
                for period, result in period_results.items():
                    if 'error' not in result and not self._check_existing_backtest_result(symbol, period):
                        insert_backtest_result(
                            symbol, period, 
                            result.get('cagr', 0),
                            result.get('win_rate', 0),
                            result.get('max_drawdown', 0)
                        )
                        logger.debug(f"Saved {period} backtest results for {symbol}")
                
                logger.info(f"Saved backtest results for {symbol}")
                return True

            except Exception as e:
                logger.error(f"Error saving backtest results for {symbol}: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error saving backtest results for {analysis_result.get('symbol', 'UNKNOWN')}: {e}")
            return False
    
    def analyze_single_stock(self, symbol: str, total_stocks: int, current_index: int) -> Dict[str, Any]:
        """
        Analyze a single stock (thread-safe).
        
        Args:
            symbol: Stock symbol to analyze
            total_stocks: Total number of stocks being processed
            current_index: Current stock index for progress tracking
            
        Returns:
            Dictionary containing analysis result and metadata
        """
        try:
            logger.info(f"Analyzing {symbol} ({current_index}/{total_stocks})")
            
            # Perform analysis
            try:
                analysis_result = self.analyzer.analyze_stock(symbol, self.app.config)
                
                logger.debug(f"Analysis result for {symbol}: {analysis_result}")
            except Exception as e:
                logger.exception(f"Error in analyzing stock {symbol}: {e}")
                raise
            
            # Minimal delay to avoid overwhelming APIs (threads handle this naturally)
            if total_stocks > 100:
                time.sleep(REQUEST_DELAY / 5)  # Reduce delay for large batches
            else:
                time.sleep(REQUEST_DELAY)
            
            # Force garbage collection after each stock analysis to prevent memory buildup
            gc.collect()
            
            return {
                'success': True,
                'symbol': symbol,
                'result': analysis_result,
                'recommended': analysis_result.get('is_recommended', False)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e),
                'recommended': False
            }
    
    def analyze_all_stocks(self, max_stocks: int = None, batch_size: int = None, use_all_symbols: bool = False, single_threaded: bool = False, offline_mode: bool = False):
        """
        Analyze all NSE stocks using multithreading and save recommendations.
        
        Args:
            max_stocks: Maximum number of stocks to analyze (for testing)
            batch_size: Number of stocks to process in each batch (from config if None)
            use_all_symbols: If True, use all NSE symbols instead of filtered ones
            single_threaded: If True, process stocks one by one without threading (for debugging)
        """
        mode_str = "single-threaded" if single_threaded else "multithreading"
        if self.verbose:
            logger.info(f"Starting automated stock analysis with {mode_str} (max_stocks={max_stocks}, use_all_symbols={use_all_symbols})")
        
        logger.info("DEBUG: About to fetch stock symbols...")
        
        if offline_mode:
            # Use offline mode - get symbols from cached data only
            if self.verbose:
                logger.info(f"Using OFFLINE mode: Getting symbols from cached data (max_stocks={max_stocks})...")
            logger.info("DEBUG: About to call get_offline_symbols_from_cache...")
            filtered_symbols = get_offline_symbols_from_cache(max_stocks)
            logger.info("DEBUG: Finished calling get_offline_symbols_from_cache")
            
            if not filtered_symbols:
                logger.error("No cached symbols found in offline mode. Try running without --offline first to build cache.")
                return
        elif use_all_symbols:
            # Get all NSE symbols without filtering
            if self.verbose:
                logger.info(f"Fetching all NSE symbols (max_stocks={max_stocks})...")
            logger.info("DEBUG: About to call get_all_nse_symbols...")
            all_symbols = get_all_nse_symbols()
            logger.info("DEBUG: Finished calling get_all_nse_symbols")
            
            if not all_symbols:
                logger.error("No NSE symbols found. Exiting analysis.")
                return
            
            # Convert to dictionary format for consistency with filtered_symbols
            if isinstance(all_symbols, list):
                filtered_symbols = {symbol: {'company_name': symbol} for symbol in all_symbols}
            else:
                filtered_symbols = all_symbols
            
            # Apply max_stocks limit if specified
            if max_stocks and len(filtered_symbols) > max_stocks:
                symbols_list = list(filtered_symbols.keys())[:max_stocks]
                filtered_symbols = {k: filtered_symbols[k] for k in symbols_list}
                if self.verbose:
                    logger.info(f"Limited to first {max_stocks} symbols from all NSE stocks")
        else:
            # Get filtered NSE symbols (actively traded with historical data)
            if self.verbose:
                logger.info(f"Fetching actively traded NSE symbols with historical data (max_stocks={max_stocks})...")
            logger.info("DEBUG: About to call get_filtered_nse_symbols...")
            filtered_symbols = get_filtered_nse_symbols(max_stocks)
            logger.info("DEBUG: Finished calling get_filtered_nse_symbols")
        
        logger.info("DEBUG: Symbol fetching completed, creating symbols list...")
        symbols_list = list(filtered_symbols.keys())
        logger.info(f"DEBUG: Created symbols list with {len(symbols_list)} symbols")
        symbol_type = "all NSE" if use_all_symbols else "actively traded"
        total_stocks = len(symbols_list)
        
        # Always show stock count regardless of verbose mode
        if self.verbose:
            logger.info(f"Found {total_stocks} {symbol_type} stocks to analyze")
        else:
            print(f"\rAnalyzing {total_stocks} {symbol_type} stocks...", flush=True)
            
        processed_count = 0
        recommended_count = 0
        not_recommended_count = 0
        failed_count = 0
        
        # Use batch size from config if not specified
        if batch_size is None:
            batch_size = BATCH_SIZE
        
        # Use full thread pool for better performance
        effective_threads = MAX_WORKER_THREADS
        
        if self.verbose:
            logger.info(f"Using {effective_threads} threads for processing {total_stocks} stocks")
            logger.info(f"Processing {total_stocks} stocks in batches of {batch_size} using {effective_threads} threads")
        
        # Process stocks in batches
        for i in range(0, total_stocks, batch_size):
            batch = symbols_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}: stocks {i+1}-{min(i+batch_size, total_stocks)}")
            
            if single_threaded:
                # Single-threaded mode for debugging
                logger.info("Using SINGLE-THREADED mode for debugging")
                for j, symbol in enumerate(batch):
                    try:
                        logger.info(f"Processing {symbol} in single-threaded mode...")
                        result = self.analyze_single_stock(symbol, total_stocks, i + j + 1)
                        logger.debug(f"Received result for {symbol}: {result}")
                        processed_count += 1
                        
                        if result['success']:
                            if self.save_recommendation(result['result']):
                                if result['recommended']:
                                    recommended_count += 1
                                else:
                                    not_recommended_count += 1
                            else:
                                failed_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.exception(f"Error in single-threaded processing for {symbol}: {e}")
                        failed_count += 1
                        processed_count += 1
            else:
                # Multi-threaded mode with timeout handling
                with ThreadPoolExecutor(max_workers=effective_threads) as executor:
                    # Submit all tasks for this batch
                    future_to_symbol = {
                        executor.submit(self.analyze_single_stock, symbol, total_stocks, i + j + 1): symbol
                        for j, symbol in enumerate(batch)
                    }
                    
                    # Process completed tasks with timeout
                    for future in as_completed(future_to_symbol, timeout=300):  # 5 minute timeout per stock
                        symbol = future_to_symbol[future]
                        try:
                            result = future.result(timeout=60)  # 1 minute timeout to get result
                            logger.debug(f"Received result for {symbol}: {result}")
                            processed_count += 1
                            
                            if result['success']:
                                if self.save_recommendation(result['result']):
                                    if result['recommended']:
                                        recommended_count += 1
                                    else:
                                        not_recommended_count += 1
                                else:
                                    failed_count += 1
                            else:
                                failed_count += 1
                                
                        except TimeoutError:
                            logger.error(f"Timeout processing {symbol} - skipping")
                            failed_count += 1
                            processed_count += 1
                        except Exception as e:
                            logger.exception(f"Error in ThreadPoolExecutor for {symbol}: {e}")
                            failed_count += 1
                            processed_count += 1
            
            # Log progress and trigger garbage collection after each batch
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            avg_time_per_stock = elapsed_time / processed_count if processed_count > 0 else 0
            estimated_remaining = (total_stocks - processed_count) * avg_time_per_stock

            # Manually trigger garbage collection
            gc.collect()

            if self.verbose:
                logger.info(f"Progress: {processed_count}/{total_stocks} stocks processed, "
                           f"{recommended_count} recommendations, {not_recommended_count} not recommended, {failed_count} failed, "
                           f"~{estimated_remaining/60:.1f} minutes remaining")
            elif self.progress_callback:
                # Call progress callback for non-verbose mode
                current_stock_symbol = batch[-1] if batch else ''
                self.progress_callback(processed_count, total_stocks, recommended_count, current_stock_symbol)
            else:
                # Fallback progress display if no callback is set
                progress_percent = (processed_count / total_stocks) * 100
                print(f"\rProgress: {progress_percent:.1f}% ({processed_count}/{total_stocks}) - {recommended_count} recommendations", end='', flush=True)
        
        # Final summary
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"Analysis complete!")
        logger.info(f"Total stocks processed: {processed_count}")
        logger.info(f"Recommendations generated: {recommended_count}")
        logger.info(f"Stocks not recommended: {not_recommended_count}")
        logger.info(f"Analysis failures: {failed_count}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Average time per stock: {total_time/processed_count:.1f} seconds")
        
        # Log current recommendations count
        try:
            from database import get_mongodb
            db = get_mongodb()
            total_recommendations = db.recommended_shares.count_documents({})
            logger.info(f"Total recommendations in MongoDB: {total_recommendations}")
        except Exception as e:
            logger.error(f"Error getting total recommendations count: {e}")
    
    def run_analysis(self, max_stocks: int = None, use_all_symbols: bool = False, offline_mode: bool = False):
        """
        Run the complete analysis process.
        
        Args:
            max_stocks: Maximum number of stocks to analyze (for testing)
            use_all_symbols: If True, use all NSE symbols instead of filtered ones
        """
        with self.app.app_context():
            try:
                logger.info("Starting run_analysis method")
                
                # Clean corrupted cache files first
                logger.info("Cleaning corrupted cache files...")
                cache_manager = get_cache_manager()
                cleaned_files = cache_manager.clean_corrupted_cache_files()
                if cleaned_files > 0:
                    logger.info(f"Cleaned {cleaned_files} corrupted cache files")
                logger.info("Cache cleaning completed")
                
                # Get configurable threshold for data purge
                days_old = self.app.config.get('DATA_PURGE_DAYS', 7)
                logger.info(f"Data purge threshold: {days_old} days")
                
                # Clear old data (recommendations and backtest results) at the start
                logger.info("SKIPPING database purge operation temporarily for debugging...")
                # self.clear_old_data(days_old=days_old)
                logger.info("Database purge operation skipped")
                
                # Analyze all stocks
                logger.info("Starting stock analysis...")
                self.analyze_all_stocks(max_stocks=max_stocks, use_all_symbols=use_all_symbols, offline_mode=offline_mode, single_threaded=getattr(self, 'single_threaded', False))
                logger.info("Stock analysis completed")
                
                logger.info("Automated analysis completed successfully")
                
            except Exception as e:
                logger.error(f"Error in automated analysis: {e}")
                raise


def main():
    """Main entry point for the script."""
    import argparse
    import logging
    
    parser = argparse.ArgumentParser(description='Automated NSE Stock Analysis')
    parser.add_argument('--max-stocks', type=int, help='Maximum number of stocks to analyze (for testing)')
    parser.add_argument('--test', action='store_true', help='Run in test mode with limited stocks')
    parser.add_argument('--all', action='store_true', help='Analyze all NSE stocks (not just filtered/actively traded ones)')
    parser.add_argument('--offline', action='store_true', help='Use offline mode (cached data only, no API calls)')
    parser.add_argument('--purge-days', type=int, help='Number of days to keep old data (overrides config). Use 0 to remove ALL data.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging with detailed output')
    parser.add_argument('--single-threaded', action='store_true', help='Use single-threaded mode for debugging (slower but more stable)')
    parser.add_argument('--disable-volume-filter', action='store_true', help='Disable volume-based filtering for analysis')
    
    args = parser.parse_args()
    
    # Configure logging IMMEDIATELY based on verbose flag
    from utils.logger import setup_logging
    global logger
    logger = setup_logging(verbose=args.verbose)
    
    # Set test mode parameters
    if args.test:
        max_stocks = 2
        if args.verbose:
            logger.info("Running in TEST mode with limited stocks")
    else:
        max_stocks = args.max_stocks
        if args.verbose:
            logger.info("Running in PRODUCTION mode with all stocks")
    
    # Log symbol selection mode
    if args.all:
        if args.verbose:
            logger.info("Using ALL NSE symbols (including inactive/low-volume stocks)")
    else:
        if args.verbose:
            logger.info("Using FILTERED NSE symbols (only actively traded stocks)")
    
    try:
        # Create analyzer with correct verbose setting from the start
        analyzer = AutomatedStockAnalysis(verbose=args.verbose)
        
        # Set single_threaded flag
        analyzer.single_threaded = args.single_threaded
        if args.single_threaded and args.verbose:
            logger.info("Single-threaded mode enabled for debugging")
        
        # Override config if CLI argument provided
        if args.purge_days is not None:
            analyzer.app.config['DATA_PURGE_DAYS'] = args.purge_days
            if args.verbose:
                logger.info(f"Data purge days set to {args.purge_days} (from CLI argument)")
        
        if args.verbose:
            # Verbose mode - logging already configured in constructor
            analyzer.run_analysis(max_stocks=max_stocks, use_all_symbols=args.all, offline_mode=args.offline)
            logger.info("Script completed successfully")
        else:
            # Non-verbose mode - logging already configured in constructor
            
            # Setup progress callback for non-verbose mode
            last_progress_update = [0]  # Use list to allow modification in nested function
            
            def progress_callback(processed, total, recommendations):
                progress_percent = (processed / total) * 100
                # Only update every 2% or when complete to show more frequent updates
                if progress_percent - last_progress_update[0] >= 2 or processed == total:
                    bar_length = 30
                    filled_length = int(bar_length * processed // total)
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    print(f"\rProgress: |{bar}| {progress_percent:.1f}% ({processed}/{total}) - {recommendations} recommendations", end='', flush=True)
                    last_progress_update[0] = progress_percent
            
            analyzer.progress_callback = progress_callback
            
            # Show initial message (we'll update this after getting the actual stock count)
            print(f"Initializing analysis...")
            analyzer.run_analysis(max_stocks=max_stocks, use_all_symbols=args.all, offline_mode=args.offline)
            print("\n")
            
            # Show final summary in non-verbose mode
            try:
                from database import get_mongodb
                db = get_mongodb()
                total_recommendations = db.recommended_shares.count_documents({})
                print(f"Analysis completed. Total recommendations in database: {total_recommendations}")
            except Exception:
                print("Analysis completed.")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
