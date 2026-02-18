"""
Main Stock Analyzer
File: scripts/analyzer.py

This module combines technical, fundamental, and sentiment analysis to generate
comprehensive stock recommendations.
"""

# Fix OpenMP/threading issues on macOS - MUST be set before importing numpy/scipy/sklearn
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any
from scripts.data_fetcher import get_all_nse_symbols, get_historical_data
from scripts.strategy_evaluator import StrategyEvaluator
from scripts.fundamental_analysis import FundamentalAnalysis
from scripts.sentiment_analysis import SentimentAnalysis
from models.recommendation import RecommendedShare
from utils.logger import setup_logging
from config import HISTORICAL_DATA_PERIOD, ANALYSIS_WEIGHTS, RECOMMENDATION_THRESHOLDS
from scripts.risk_management import RiskManager
from scripts.sector_analysis import SectorAnalyzer
from scripts.backtesting_runner import BacktestingRunner
from scripts.market_regime_detection import MarketRegimeDetection
from scripts.market_microstructure import MarketMicrostructureAnalyzer
from scripts.alternative_data_analyzer import AlternativeDataAnalyzer
from scripts.predictor import PricePredictor
from scripts.rl_trading_agent import RLTradingAgent
from scripts.tca_analysis import TransactionCostAnalyzer

logger = setup_logging()

class StockAnalyzer:
    """
    Main stock analyzer that combines all analysis types.
    """
    
    def __init__(self):
        """
        Initialize the stock analyzer.
        """
        logger.info("Initializing StockAnalyzer...")
        
        logger.info("Initializing StrategyEvaluator...")
        self.strategy_evaluator = StrategyEvaluator()
        logger.info("StrategyEvaluator initialized")
        
        logger.info("Initializing FundamentalAnalysis...")
        self.fundamental_analyzer = FundamentalAnalysis()
        logger.info("FundamentalAnalysis initialized")
        
        # Note: SentimentAnalysis initialization deferred to avoid model loading hang
        logger.info("Deferring SentimentAnalysis initialization to avoid hang...")
        self.sentiment_analyzer = None
        logger.info("SentimentAnalysis will be initialized on first use")
        
        logger.info("Initializing RiskManager...")
        self.risk_manager = RiskManager()
        logger.info("RiskManager initialized")
        
        # Skip heavy initializations that might hang
        logger.info("Skipping heavy analyzer initializations to prevent hangs...")
        
        # These analyzers will be initialized on first use if needed
        self.sector_analyzer = None
        self.market_regime_detector = None
        self.market_microstructure_analyzer = None
        self.alternative_data_analyzer = None
        self.predictor = None
        self.rl_trading_agent = None
        self.tca_analyzer = None
        
        logger.info("Heavy analyzers will be initialized on demand")
        
        logger.info("StockAnalyzer initialization complete")
        
    def analyze_stock(self, symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on a stock.
        
        Args:
            symbol: Stock symbol
            app_config: Application configuration
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Get company name
            all_symbols = get_all_nse_symbols()
            company_name = all_symbols.get(symbol, symbol)
            
            logger.info(f"Starting analysis for {symbol} ({company_name})")
            
            # Initialize result structure
            result = {
                'symbol': symbol,
                'company_name': company_name,
                'technical_score': 0.0,
                'fundamental_score': 0.0,
                'sentiment_score': 0.0,
                'is_recommended': False,
                'reason': [],
                'detailed_analysis': {}
            }
            
            # 1. Technical Analysis
            logger.info(f"Performing technical analysis for {symbol}")
            logger.debug(f"Starting technical analysis for {symbol}")
            
            try:
                historical_data = get_historical_data(symbol, app_config.get('HISTORICAL_DATA_PERIOD', HISTORICAL_DATA_PERIOD))
                logger.debug(f"Retrieved {len(historical_data)} days of historical data for {symbol}")
                
                if historical_data.empty:
                    logger.warning(f"No historical data for {symbol}")
                    result['reason'].append("No historical data available for technical analysis")
                    result['technical_score'] = -1.0
                    technical_analysis = {'error': 'No historical data'}
                else:
                    logger.debug(f"Running strategy evaluation for {symbol}")
                    technical_analysis = self.strategy_evaluator.evaluate_strategies(symbol, historical_data)
                    # Convert technical score to -1 to 1 scale, but less punitive for neutral signals
                    # Original: 0.0 -> -1.0, 0.5 -> 0.0, 1.0 -> 1.0
                    # New: 0.0 -> -0.3, 0.5 -> 0.4, 1.0 -> 1.0 (more forgiving for neutral technical signals)
                    raw_score = technical_analysis['technical_score']
                    if raw_score <= 0.5:
                        result['technical_score'] = (raw_score * 1.4) - 0.3  # 0.0->-0.3, 0.5->0.4
                    else:
                        result['technical_score'] = (raw_score - 0.5) * 1.2 + 0.4  # 0.5->0.4, 1.0->1.0
                    logger.debug(f"Technical analysis complete for {symbol}: raw={raw_score:.2f}, scaled={result['technical_score']:.2f}")
                    
            except Exception as e:
                logger.exception(f"Error in technical analysis for {symbol}: {e}")
                result['reason'].append(f"Technical analysis failed: {str(e)}")
                result['technical_score'] = -1.0
                technical_analysis = {'error': str(e)}
                
            result['detailed_analysis']['technical'] = technical_analysis
            
            # 2. Fundamental Analysis - Skip if disabled for speed
            if app_config.get('SKIP_FUNDAMENTAL', False):
                logger.info(f"Skipping fundamental analysis for {symbol} (SKIP_FUNDAMENTAL=True)")
                result['fundamental_score'] = 0.1  # Default to neutral positive score
                result['reason'].append("Fundamental analysis skipped for performance")
            else:
                logger.info(f"Performing fundamental analysis for {symbol}")
                logger.debug(f"Starting fundamental analysis for {symbol}")
                
                try:
                    fundamental_score = self.fundamental_analyzer.perform_fundamental_analysis(symbol)
                    result['fundamental_score'] = fundamental_score
                    logger.debug(f"Fundamental analysis complete for {symbol}: score={fundamental_score:.2f}")
                    
                    if fundamental_score > 0:
                        result['reason'].append("Fundamental analysis shows positive indicators")
                    elif fundamental_score < -0.5:
                        result['reason'].append("Fundamental analysis shows negative indicators")
                    else:
                        result['reason'].append("Fundamental analysis shows neutral indicators")
                        
                except Exception as e:
                    logger.exception(f"Error in fundamental analysis for {symbol}: {e}")
                    result['fundamental_score'] = 0.1  # Default to neutral positive score
                    result['reason'].append("Fundamental analysis unavailable")
            
            # 3. Sentiment Analysis
            if app_config.get('SKIP_SENTIMENT', False):
                logger.info(f"Skipping sentiment analysis for {symbol} (SKIP_SENTIMENT=True)")
                result['sentiment_score'] = 0.1  # Default to neutral positive score
                result['reason'].append("Sentiment analysis skipped for performance")
            else:
                logger.info(f"Performing sentiment analysis for {symbol}")
                logger.debug(f"Starting sentiment analysis for {symbol} ({company_name})")
                
                try:
                    # Initialize sentiment analyzer on first use
                    if self.sentiment_analyzer is None:
                        logger.info("Initializing SentimentAnalysis on first use...")
                        self.sentiment_analyzer = SentimentAnalysis()
                        logger.info("SentimentAnalysis initialized")
                    
                    sentiment_score = self.sentiment_analyzer.perform_sentiment_analysis(company_name)
                    result['sentiment_score'] = sentiment_score
                    logger.debug(f"Sentiment analysis complete for {symbol}: score={sentiment_score:.2f}")
                    
                    if sentiment_score > 0.1:
                        result['reason'].append("News sentiment is positive")
                    elif sentiment_score < -0.1:
                        result['reason'].append("News sentiment is negative")
                    else:
                        result['reason'].append("News sentiment is neutral")
                        
                except Exception as e:
                    logger.exception(f"Error in sentiment analysis for {symbol}: {e}")
                    result['sentiment_score'] = 0.1  # Default to neutral positive score
                    result['reason'].append("Sentiment analysis unavailable")
            
            # 4. Sector Analysis (only if enabled)
            analysis_config = app_config.get('ANALYSIS_CONFIG', {})
            if analysis_config.get('sector_analysis'):
                logger.info(f"Performing sector analysis for {symbol}")
                logger.debug(f"Starting sector analysis for {symbol}")
                try:
                    # Lazy init sector analyzer
                    if self.sector_analyzer is None:
                        from scripts.sector_analysis import SectorAnalyzer
                        self.sector_analyzer = SectorAnalyzer()
                    sector_recommendation = self.sector_analyzer.get_sector_recommendation(symbol)
                    result['sector_analysis'] = sector_recommendation
                    logger.debug(f"Sector analysis complete for {symbol}: {sector_recommendation['recommendation']}")
                    
                    # Adjust recommendation based on sector momentum
                    if sector_recommendation['recommendation'] == 'Strong Sector Momentum - Favorable':
                        result['reason'].append("Strong sector momentum supports the trade")
                    elif sector_recommendation['recommendation'] == 'Weak Sector Momentum - Caution':
                        result['reason'].append("Weak sector momentum - trade with caution")
                    else:
                        result['reason'].append("Neutral sector momentum")
                except Exception as e:
                    logger.exception(f"Error in sector analysis for {symbol}: {e}")
                    result['sector_analysis'] = {'error': str(e)}

            # 5. Advanced Analysis Modules
            analysis_config = app_config.get('ANALYSIS_CONFIG', {})

            if analysis_config.get('market_regime_detection'):
                logger.info(f"Performing market regime detection for {symbol}")
                try:
                    # Re-instantiate for specific symbol
                    self.market_regime_detector.symbol = symbol
                    self.market_regime_detector.fit_all_models()
                    regime = self.market_regime_detector.get_comprehensive_regime_analysis()
                    result['market_regime'] = regime
                    result['reason'].append(f"Market Regime: {regime.get('consensus_regime')}")
                except Exception as e:
                    logger.exception(f"Error in market regime detection for {symbol}: {e}")
                    result['market_regime'] = {'error': str(e)}

            if analysis_config.get('market_microstructure'):
                logger.info(f"Performing market microstructure analysis for {symbol}")
                try:
                    microstructure = self.market_microstructure_analyzer.analyze(symbol)
                    result['market_microstructure'] = microstructure
                except Exception as e:
                    logger.exception(f"Error in market microstructure analysis for {symbol}: {e}")
                    result['market_microstructure'] = {'error': str(e)}

            if analysis_config.get('alternative_data'):
                logger.info(f"Performing alternative data analysis for {symbol}")
                try:
                    # Pass historical data to the enhanced alternative data analyzer
                    alt_data = self.alternative_data_analyzer.analyze(symbol, historical_data)
                    result['alternative_data'] = alt_data
                    
                    # Add to reasoning based on alternative data score
                    alt_score = alt_data.get('final_alternative_score', 0)
                    if alt_score > 0.1:
                        result['reason'].append(f"Alternative data signals positive (score: {alt_score:.3f})")
                    elif alt_score < -0.1:
                        result['reason'].append(f"Alternative data signals negative (score: {alt_score:.3f})")
                    else:
                        result['reason'].append("Alternative data signals neutral")
                        
                except Exception as e:
                    logger.exception(f"Error in alternative data analysis for {symbol}: {e}")
                    result['alternative_data'] = {'error': str(e)}

            if analysis_config.get('predictive_analysis'):
                logger.info(f"Performing predictive analysis for {symbol}")
                try:
                    # Re-instantiate for this symbol
                    predictor = PricePredictor(symbol)
                    train_data, test_data = predictor.prepare_data()
                    if train_data and test_data is not None:
                        predictor.train(train_data, epochs=20)
                        predicted_price = predictor.predict_next_day_price(test_data)
                        result['prediction'] = {'predicted_price': predicted_price}
                    else:
                        result['prediction'] = {'error': 'Not enough data'}
                except Exception as e:
                    logger.exception(f"Error in predictive analysis for {symbol}: {e}")
                    result['prediction'] = {'error': str(e)}

            if analysis_config.get('rl_trading_agent'):
                logger.info(f"Performing RL trading agent analysis for {symbol}")
                try:
                    agent = RLTradingAgent(symbol)
                    rl_action = agent.run_analysis(historical_data)
                    result['rl_action'] = rl_action
                except Exception as e:
                    logger.exception(f"Error in RL trading agent analysis for {symbol}: {e}")
                    result['rl_action'] = {'error': str(e)}

            if analysis_config.get('tca_analysis'):
                logger.info(f"Performing TCA analysis for {symbol}")
                try:
                    # Example cost estimation for 100 shares at latest close price
                    if not historical_data.empty:
                        trade_value = historical_data['Close'].iloc[-1] * 100
                        tca = self.tca_analyzer.estimate_trade_costs(trade_value)
                        result['tca_analysis'] = tca
                    else:
                        result['tca_analysis'] = {'error': 'No historical data'}
                except Exception as e:
                    logger.exception(f"Error in TCA analysis for {symbol}: {e}")
                    result['tca_analysis'] = {'error': str(e)}
            
            # 6. Initial Combined Analysis (without backtest consideration)
            logger.info(f"Combining analysis results for {symbol}")
            logger.debug(f"Starting combined analysis for {symbol}")
            
            try:
                result = self._combine_analysis_results(result, consider_backtest=False, keep_reason_as_list=True)
                logger.debug(f"Combined analysis complete for {symbol}: combined_score={result.get('combined_score', 0):.2f}")
            except Exception as e:
                logger.exception(f"Error in combined analysis for {symbol}: {e}")
                result['is_recommended'] = False
                result['reason'].append(f"Combined analysis failed: {str(e)}")
            
            # 7. Trade-level Analysis - Get detailed trade recommendations
            if not historical_data.empty:
                logger.info(f"Performing trade-level analysis for {symbol}")
                logger.debug(f"Starting trade-level analysis for {symbol}")
                
                try:
                    trade_analysis = self.analyze(symbol)
                    logger.debug(f"Trade-level analysis complete for {symbol}: {trade_analysis.get('recommendation', 'UNKNOWN')}")
                    
                    # Merge trade analysis into main result
                    if 'error' not in trade_analysis:
                        result['trade_plan'] = {
                            'buy_price': trade_analysis.get('buy_price'),
                            'sell_price': trade_analysis.get('sell_price'),
                            'stop_loss': trade_analysis.get('stop_loss'),
                            'days_to_target': trade_analysis.get('days_to_target'),
                            'entry_timing': trade_analysis.get('entry_timing'),
                            'risk_reward_ratio': trade_analysis.get('risk_reward_ratio'),
                            'confidence': trade_analysis.get('confidence')
                        }
                        
                        # Convert None values to safe defaults for DB insertion
                        for key, value in result['trade_plan'].items():
                            if value is None:
                                if key in ['buy_price', 'sell_price', 'stop_loss']:
                                    result['trade_plan'][key] = 0.0
                                elif key == 'days_to_target':
                                    result['trade_plan'][key] = 0
                                elif key in ['risk_reward_ratio', 'confidence']:
                                    result['trade_plan'][key] = 0.0
                                else:
                                    result['trade_plan'][key] = ''
                    else:
                        result['trade_plan'] = {
                            'buy_price': 0.0,
                            'sell_price': 0.0,
                            'stop_loss': 0.0,
                            'days_to_target': 0,
                            'entry_timing': 'WAIT',
                            'risk_reward_ratio': 0.0,
                            'confidence': 0.0
                        }
                except Exception as e:
                    logger.exception(f"Error in trade-level analysis for {symbol}: {e}")
                    result['trade_plan'] = {
                        'buy_price': 0.0,
                        'sell_price': 0.0,
                        'stop_loss': 0.0,
                        'days_to_target': 0,
                        'entry_timing': 'WAIT',
                        'risk_reward_ratio': 0.0,
                        'confidence': 0.0,
                        'error': str(e)
                    }
            
            # 8. Backtesting Analysis
            if not historical_data.empty:
                logger.info(f"Performing backtesting analysis for {symbol}")
                logger.debug(f"Starting backtesting for {symbol}")
                
                try:
                    backtest_results = self._perform_backtesting(symbol, historical_data)
                    result['backtest'] = backtest_results
                    
                    # Log backtesting results
                    if backtest_results.get('status') == 'completed':
                        combined_metrics = backtest_results.get('combined_metrics', {})
                        logger.debug(f"Backtesting complete for {symbol}: CAGR={combined_metrics.get('avg_cagr', 0)}%")
                        result['reason'].append(f"Backtesting: CAGR={combined_metrics.get('avg_cagr', 0)}%, Win Rate={combined_metrics.get('avg_win_rate', 0)}%")
                    elif backtest_results.get('status') == 'insufficient_data':
                        logger.info(f"Skipping backtesting for {symbol} due to insufficient data")
                        result['reason'].append(f"Backtesting skipped: {backtest_results.get('message', 'insufficient data')}")
                    else:
                        logger.warning(f"Backtesting failed for {symbol}: {backtest_results.get('error', 'unknown error')}")
                        result['reason'].append("Backtesting failed")
                except Exception as e:
                    logger.exception(f"Error in backtesting for {symbol}: {e}")
                    result['backtest'] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    result['reason'].append("Backtesting failed")
            else:
                result['backtest'] = {
                    'status': 'no_data',
                    'message': 'No historical data available for backtesting'
                }
            
            # 9. Final Recommendation Check (considering backtest results)
            if result.get('is_recommended', False):
                logger.info(f"Final recommendation check for {symbol} with backtest results")
                try:
                    # Re-evaluate recommendation with backtest consideration
                    result = self._combine_analysis_results(result, consider_backtest=True, keep_reason_as_list=True)
                    logger.info(f"Final recommendation for {symbol}: {result.get('is_recommended', False)} (after backtest check)")
                except Exception as e:
                    logger.exception(f"Error in final recommendation check for {symbol}: {e}")
            
            # 10. Risk Management Analysis (only for recommended stocks after final check)
            if not historical_data.empty and result.get('is_recommended', False):
                logger.info(f"Performing risk management analysis for {symbol}")
                current_price = historical_data['Close'].iloc[-1]
                
                # Calculate stop loss
                stop_loss_info = self.risk_manager.calculate_stop_loss(historical_data, current_price)
                
                # Calculate position size
                position_info = self.risk_manager.calculate_position_size(current_price, stop_loss_info['stop_loss'])
                
                # Calculate profit targets
                profit_targets = self.risk_manager.calculate_profit_targets(current_price, stop_loss_info['stop_loss'])
                
                # Calculate pivot points
                pivot_points = self.risk_manager.calculate_pivot_points(historical_data)
                
                result['risk_management'] = {
                    'current_price': current_price,
                    'stop_loss': stop_loss_info,
                    'position_sizing': position_info,
                    'profit_targets': profit_targets,
                    'pivot_points': pivot_points
                }
            
            # Convert reason from list to string at the very end
            if isinstance(result['reason'], list):
                result['reason'] = " ".join(result['reason'])
            
            logger.info(f"Analysis complete for {symbol}: Technical={result['technical_score']:.2f}, "
                       f"Fundamental={result['fundamental_score']:.2f}, Sentiment={result['sentiment_score']:.2f}, "
                       f"Recommended={result['is_recommended']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing stock {symbol}: {e}")
            return {
                'symbol': symbol,
                'company_name': company_name if 'company_name' in locals() else symbol,
                'technical_score': -1.0,
                'fundamental_score': -1.0,
                'sentiment_score': 0.0,
                'is_recommended': False,
                'reason': [f"Analysis error: {str(e)}"],
                'detailed_analysis': {'error': str(e)}
            }
    
    def _combine_analysis_results(self, result: Dict[str, Any], consider_backtest: bool = True, keep_reason_as_list: bool = False) -> Dict[str, Any]:
        """
        Combine technical, fundamental, and sentiment analysis results.
        
        Args:
            result: Analysis results dictionary
            consider_backtest: Whether to consider backtest results in recommendation
            keep_reason_as_list: Whether to keep reason as list instead of converting to string
            
        Returns:
            Updated results dictionary with combined recommendation
        """
        technical_score = result['technical_score']
        fundamental_score = result['fundamental_score']
        sentiment_score = result['sentiment_score']
        
        # Get configurable weights and thresholds
        technical_weight = ANALYSIS_WEIGHTS.get('technical', 0.5)
        fundamental_weight = ANALYSIS_WEIGHTS.get('fundamental', 0.3)
        sentiment_weight = ANALYSIS_WEIGHTS.get('sentiment', 0.2)
        
        # Normalize weights to ensure they sum to 1
        total_weight = technical_weight + fundamental_weight + sentiment_weight
        if total_weight != 1.0:
            technical_weight /= total_weight
            fundamental_weight /= total_weight
            sentiment_weight /= total_weight
        
        combined_score = (
            technical_score * technical_weight +
            fundamental_score * fundamental_weight +
            sentiment_score * sentiment_weight
        )
        
        result['combined_score'] = combined_score
        result['analysis_weights'] = {
            'technical': technical_weight,
            'fundamental': fundamental_weight,
            'sentiment': sentiment_weight
        }
        
        # Get configurable thresholds
        strong_buy_threshold = RECOMMENDATION_THRESHOLDS.get('strong_buy_combined', 0.3)
        buy_threshold = RECOMMENDATION_THRESHOLDS.get('buy_combined', 0.2)
        technical_strong_threshold = RECOMMENDATION_THRESHOLDS.get('technical_strong_buy', 0.5)
        sell_threshold = RECOMMENDATION_THRESHOLDS.get('sell_combined', -0.3)
        sentiment_positive_threshold = RECOMMENDATION_THRESHOLDS.get('sentiment_positive', 0.1)
        
        # Recommendation logic with flexible backtest requirements
        if consider_backtest:
            # Get backtest CAGR and trade plan ETA
            backtest_cagr = result.get('backtest', {}).get('combined_metrics', {}).get('avg_cagr', 0)
            trade_plan = result.get('trade_plan', {})
            days_to_target = trade_plan.get('days_to_target', 0) if trade_plan else 0
            
            # More flexible backtest requirements - allow strong analysis to override
            strong_analysis_override = (
                (technical_score > 0.3 and fundamental_score > 0.3) or  # Strong both
                (combined_score > 0.3) or  # Very strong combined
                (technical_score > 0.4) or  # Very strong technical
                (fundamental_score > 0.5)   # Very strong fundamental
            )
            
            if strong_analysis_override:
                # Strong analysis signals can override backtest requirements
                backtest_condition = True
                threshold_reason = f"Strong analysis signals override backtest requirement (CAGR: {backtest_cagr:.2f}%)"
            elif days_to_target > 30:
                # For trades longer than 30 days, use relaxed CAGR requirement
                min_backtest_return = 1.0  # Reduced from 2.0% to 1.0%
                backtest_condition = backtest_cagr >= min_backtest_return
                threshold_reason = f"ETA {days_to_target} days requires CAGR >= 1.0%"
            else:
                # For trades 30 days or less, use standard threshold
                min_backtest_return = RECOMMENDATION_THRESHOLDS.get('min_backtest_return', 0.0)
                backtest_condition = backtest_cagr >= min_backtest_return
                threshold_reason = f"ETA {days_to_target} days requires CAGR >= {min_backtest_return}%"
                
            logger.info(f"Backtest check for {result.get('symbol', 'UNKNOWN')}: CAGR={backtest_cagr:.2f}%, {threshold_reason}, Condition={backtest_condition}")
        else:
            # Don't consider backtest results in initial analysis
            backtest_condition = True
            backtest_cagr = 0
            min_backtest_return = 0
            threshold_reason = "Initial analysis - backtest not considered"
        
        # Enhanced recommendation logic - more flexible and nuanced
        technical_minimum = RECOMMENDATION_THRESHOLDS.get('technical_minimum', -0.1)
        fundamental_minimum = RECOMMENDATION_THRESHOLDS.get('fundamental_minimum', -0.2)
        
        # Strong buy: Multiple positive indicators with good backtest
        if ((technical_score > 0.1 and fundamental_score > 0.1 and sentiment_score > 0) or 
            (combined_score > strong_buy_threshold) or
            (technical_score > 0.3 and fundamental_score > -0.1) or
            (fundamental_score > 0.4 and technical_score > -0.1)) and backtest_condition:
            result['is_recommended'] = True
            result['recommendation_strength'] = 'STRONG_BUY'
            if technical_score > 0.1 and fundamental_score > 0.1 and sentiment_score > 0:
                result['reason'].append("All analysis types show positive signals")
            else:
                result['reason'].append("High combined score indicates strong buy opportunity")
        
        # Regular buy: More flexible criteria with negative combined scores allowed
        elif (combined_score >= buy_threshold and  # Allow negative scores based on threshold
              ((technical_score > technical_minimum and fundamental_score > fundamental_minimum) or 
               (fundamental_score > 0 and technical_score > technical_minimum) or 
               (technical_score > 0.1 and sentiment_score > sentiment_positive_threshold) or 
               (fundamental_score > 0.1 and sentiment_score > sentiment_positive_threshold) or 
               (combined_score > buy_threshold))):
            result['is_recommended'] = True
            result['recommendation_strength'] = 'BUY'
            result['reason'].append("Analysis meets buy criteria with acceptable combined score")
        
        # Technical-focused buy: Technical analysis is primary for swing trading
        elif (combined_score >= -0.30 and  # More lenient negative tolerance
              (technical_score > technical_strong_threshold or 
               (technical_score > -0.20 and combined_score > -0.25)) and backtest_condition):
            result['is_recommended'] = True
            result['recommendation_strength'] = 'WEAK_BUY'
            result['reason'].append("Technical analysis signals with acceptable risk profile")
        
        # Opportunistic buy: Reasonable combined score even with mixed individual scores
        elif (combined_score > (buy_threshold * 0.5) and combined_score >= -0.25 and  # Allow some negative
              backtest_condition and not consider_backtest):
            result['is_recommended'] = True
            result['recommendation_strength'] = 'OPPORTUNISTIC_BUY'
            result['reason'].append("Moderate combined score suggests potential opportunity")
        
        else:
            result['is_recommended'] = False
            # Changed from showing SELL to always showing HOLD - we only provide BUY recommendations
            result['recommendation_strength'] = 'HOLD'
            if consider_backtest and backtest_cagr < min_backtest_return:
                result['reason'].append(f"Backtest CAGR ({backtest_cagr:.2f}%) below minimum threshold ({min_backtest_return:.2f}%)")
            else:
                result['reason'].append("Analysis does not support buying at this time")
        
        # Format reason as a single string only if requested
        if not keep_reason_as_list:
            result['reason'] = " ".join(result['reason'])
        
        return result
    
    def analyze_multiple_stocks(self, symbols: list, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze multiple stocks and return aggregated results.
        
        Args:
            symbols: List of stock symbols
            app_config: Application configuration
            
        Returns:
            Dictionary containing results for all stocks
        """
        results = {}
        recommended_stocks = []
        
        for symbol in symbols:
            try:
                result = self.analyze_stock(symbol, app_config)
                results[symbol] = result
                
                if result['is_recommended']:
                    recommended_stocks.append(result)
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                results[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'is_recommended': False
                }
        
        # Sort recommended stocks by combined score
        recommended_stocks.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return {
            'total_analyzed': len(symbols),
            'total_recommended': len(recommended_stocks),
            'recommended_stocks': recommended_stocks,
            'all_results': results
        }
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Compute buy/sell recommendations and timing for a stock.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary containing:
            - buy_price: Recommended buy price
            - sell_price: Recommended sell price
            - days_to_target: Estimated days to reach target
            - recommendation: BUY/SELL/HOLD
            - entry_timing: IMMEDIATE/WAIT_FOR_DIP/WAIT_FOR_BREAKOUT
            - risk_reward_ratio: Risk to reward ratio
            - stop_loss: Stop loss price
            - confidence: Confidence level (0-1)
        """
        try:
            # Get historical data
            historical_data = get_historical_data(symbol, HISTORICAL_DATA_PERIOD)
            
            if historical_data.empty:
                return {
                    'symbol': symbol,
                    'error': 'No historical data available',
                    'buy_price': None,
                    'sell_price': None,
                    'days_to_target': None,
                    'recommendation': 'HOLD',
                    'entry_timing': 'WAIT',
                    'risk_reward_ratio': 0,
                    'stop_loss': None,
                    'confidence': 0
                }
            
            current_price = historical_data['Close'].iloc[-1]
            
            # Calculate technical indicators
            sma_20 = ta.SMA(historical_data['Close'].values, timeperiod=20)
            sma_50 = ta.SMA(historical_data['Close'].values, timeperiod=50)
            ema_12 = ta.EMA(historical_data['Close'].values, timeperiod=12)
            ema_26 = ta.EMA(historical_data['Close'].values, timeperiod=26)
            rsi = ta.RSI(historical_data['Close'].values, timeperiod=14)
            atr = ta.ATR(historical_data['High'].values, historical_data['Low'].values, 
                        historical_data['Close'].values, timeperiod=14)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(historical_data['Close'].values, 
                                                     timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            # Get latest values
            current_sma_20 = sma_20[-1] if not pd.isna(sma_20[-1]) else current_price
            current_sma_50 = sma_50[-1] if not pd.isna(sma_50[-1]) else current_price
            current_ema_12 = ema_12[-1] if not pd.isna(ema_12[-1]) else current_price
            current_ema_26 = ema_26[-1] if not pd.isna(ema_26[-1]) else current_price
            current_rsi = rsi[-1] if not pd.isna(rsi[-1]) else 50
            current_atr = atr[-1] if not pd.isna(atr[-1]) else current_price * 0.02
            current_bb_upper = bb_upper[-1] if not pd.isna(bb_upper[-1]) else current_price * 1.05
            current_bb_lower = bb_lower[-1] if not pd.isna(bb_lower[-1]) else current_price * 0.95
            
            # Find support and resistance levels
            support_level = self._find_support_resistance(historical_data, 'support')
            resistance_level = self._find_support_resistance(historical_data, 'resistance')
            
            # Generate buy/sell recommendations with risk-reward logic
            recommendation_data = self._generate_buy_sell_recommendations(
                current_price, current_sma_20, current_sma_50, current_ema_12, current_ema_26,
                current_rsi, current_atr, current_bb_upper, current_bb_lower,
                support_level, resistance_level
            )
            
            # Calculate days to target using volatility
            days_to_target = self._estimate_days_to_target(
                current_price, recommendation_data['sell_price'], current_atr
            )
            
            # Calculate confidence based on signal strength
            confidence = self._calculate_confidence(
                current_price, current_sma_20, current_sma_50, current_rsi,
                recommendation_data['recommendation']
            )
            
            result = {
                'symbol': symbol,
                'current_price': current_price,
                'buy_price': recommendation_data['buy_price'],
                'sell_price': recommendation_data['sell_price'],
                'stop_loss': recommendation_data['stop_loss'],
                'days_to_target': days_to_target,
                'recommendation': recommendation_data['recommendation'],
                'entry_timing': recommendation_data['entry_timing'],
                'risk_reward_ratio': recommendation_data['risk_reward_ratio'],
                'confidence': confidence,
                'technical_indicators': {
                    'sma_20': current_sma_20,
                    'sma_50': current_sma_50,
                    'ema_12': current_ema_12,
                    'ema_26': current_ema_26,
                    'rsi': current_rsi,
                    'atr': current_atr,
                    'bb_upper': current_bb_upper,
                    'bb_lower': current_bb_lower,
                    'support_level': support_level,
                    'resistance_level': resistance_level
                }
            }
            
            logger.info(f"Analysis complete for {symbol}: {recommendation_data['recommendation']} "
                       f"at {current_price:.2f}, target: {recommendation_data['sell_price']:.2f}, "
                       f"days to target: {days_to_target}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze method for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'buy_price': None,
                'sell_price': None,
                'days_to_target': None,
                'recommendation': 'HOLD',
                'entry_timing': 'WAIT',
                'risk_reward_ratio': 0,
                'stop_loss': None,
                'confidence': 0
            }
    
    def _find_support_resistance(self, data: pd.DataFrame, level_type: str) -> float:
        """
        Find support or resistance levels using pivot points.
        
        Args:
            data: Historical price data
            level_type: 'support' or 'resistance'
            
        Returns:
            Support or resistance level
        """
        try:
            if len(data) < 10:
                if level_type == 'support':
                    return data['Low'].min()
                else:
                    return data['High'].max()
            
            # Look for pivot points in the last 20 days
            lookback = min(20, len(data))
            recent_data = data.tail(lookback)
            
            if level_type == 'support':
                # Find local minima (support levels)
                levels = []
                for i in range(2, len(recent_data) - 2):
                    if (recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-1] and 
                        recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+1] and
                        recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-2] and 
                        recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+2]):
                        levels.append(recent_data['Low'].iloc[i])
                
                return max(levels) if levels else recent_data['Low'].min()
            
            else:  # resistance
                # Find local maxima (resistance levels)
                levels = []
                for i in range(2, len(recent_data) - 2):
                    if (recent_data['High'].iloc[i] > recent_data['High'].iloc[i-1] and 
                        recent_data['High'].iloc[i] > recent_data['High'].iloc[i+1] and
                        recent_data['High'].iloc[i] > recent_data['High'].iloc[i-2] and 
                        recent_data['High'].iloc[i] > recent_data['High'].iloc[i+2]):
                        levels.append(recent_data['High'].iloc[i])
                
                return min(levels) if levels else recent_data['High'].max()
                
        except Exception as e:
            logger.error(f"Error finding {level_type} level: {e}")
            if level_type == 'support':
                return data['Low'].min()
            else:
                return data['High'].max()
    
    def _generate_buy_sell_recommendations(self, current_price: float, sma_20: float, sma_50: float,
                                         ema_12: float, ema_26: float, rsi: float, atr: float,
                                         bb_upper: float, bb_lower: float, support: float,
                                         resistance: float) -> Dict[str, Any]:
        """
        Generate buy/sell recommendations based on technical indicators.
        
        Args:
            current_price: Current stock price
            sma_20: 20-day Simple Moving Average
            sma_50: 50-day Simple Moving Average
            ema_12: 12-day Exponential Moving Average
            ema_26: 26-day Exponential Moving Average
            rsi: Relative Strength Index
            atr: Average True Range
            bb_upper: Bollinger Band Upper
            bb_lower: Bollinger Band Lower
            support: Support level
            resistance: Resistance level
            
        Returns:
            Dictionary with buy/sell recommendations
        """
        try:
            # Initialize variables
            recommendation = 'HOLD'
            entry_timing = 'WAIT'
            buy_price = current_price
            sell_price = current_price
            stop_loss = current_price * 0.95
            risk_reward_ratio = 0
            
            # Ensure ATR is reasonable - minimum 0.5% of price
            if atr < current_price * 0.005:
                atr = current_price * 0.02  # Default to 2% volatility
            
            # Ensure support and resistance are reasonable
            if abs(support - current_price) < current_price * 0.02:
                support = current_price * 0.97  # Set support 3% below current price
            if abs(resistance - current_price) < current_price * 0.02:
                resistance = current_price * 1.05  # Set resistance 5% above current price
            
            # Bullish signals
            ma_cross_bullish = ema_12 > ema_26 and sma_20 > sma_50
            price_above_ma = current_price > sma_20 and current_price > sma_50
            rsi_oversold_recovery = 30 < rsi < 70
            near_support = abs(current_price - support) / current_price < 0.05
            breakout_potential = current_price > (resistance * 0.98)
            
            # Bearish signals
            ma_cross_bearish = ema_12 < ema_26 and sma_20 < sma_50
            price_below_ma = current_price < sma_20 and current_price < sma_50
            rsi_overbought = rsi > 70
            near_resistance = abs(current_price - resistance) / current_price < 0.05
            
            # Generate recommendations with realistic targets - ONLY BUY OR HOLD, NEVER SELL
            if ma_cross_bullish and price_above_ma and rsi_oversold_recovery:
                recommendation = 'BUY'
                if breakout_potential:
                    entry_timing = 'IMMEDIATE'
                    buy_price = current_price
                    # Target 3-5% above resistance for breakout plays
                    sell_price = resistance * 1.04  
                elif near_support:
                    entry_timing = 'IMMEDIATE'
                    buy_price = current_price
                    # Target 5-8% above current price for support bounces
                    sell_price = current_price * 1.06
                else:
                    entry_timing = 'WAIT_FOR_DIP'
                    buy_price = max(support * 1.01, current_price * 0.98)
                    # Target 6-10% above buy price
                    sell_price = buy_price * 1.08
                
                # Calculate realistic stop loss (3-5% below buy price)
                stop_loss = buy_price * 0.96
                
            elif ma_cross_bearish or price_below_ma or rsi_overbought or near_resistance:
                # Changed from SELL to HOLD - wait for better conditions
                recommendation = 'HOLD'
                entry_timing = 'WAIT'
                # Set conservative buy price for potential future entry
                buy_price = current_price * 0.95  # Wait for 5% dip
                sell_price = current_price * 1.15  # Target 15% gain when conditions improve
                stop_loss = current_price * 0.90   # 10% stop loss
                
            elif rsi < 30 and near_support:
                recommendation = 'BUY'
                entry_timing = 'WAIT_FOR_BREAKOUT'
                buy_price = support * 1.005  # Buy slightly above support
                sell_price = buy_price * 1.08  # Target 8% gain
                stop_loss = support * 0.96  # 4% below support
                
            elif current_price > bb_upper:
                # Changed from SELL to HOLD - wait for better entry conditions
                recommendation = 'HOLD'
                entry_timing = 'WAIT'
                # Wait for price to come down from overbought levels
                buy_price = current_price * 0.92  # Wait for 8% correction
                sell_price = current_price * 1.10  # Target 10% gain when conditions improve
                stop_loss = current_price * 0.85   # 15% stop loss
                
            elif current_price < bb_lower:
                recommendation = 'BUY'
                entry_timing = 'IMMEDIATE'
                buy_price = current_price
                sell_price = current_price * 1.08  # Target 8% above lower band
                stop_loss = bb_lower * 0.95
            
            # For HOLD recommendations, set dynamic targets based on volatility and other factors
            if recommendation == 'HOLD':
                # Calculate dynamic targets based on stock characteristics
                dynamic_targets = self._calculate_dynamic_hold_targets(
                    current_price, atr, rsi, support, resistance, bb_upper, bb_lower
                )
                buy_price = dynamic_targets['buy_price']
                sell_price = dynamic_targets['sell_price']
                stop_loss = dynamic_targets['stop_loss']
            
            # Calculate risk-reward ratio
            if recommendation == 'BUY' and buy_price and sell_price and stop_loss:
                risk = abs(buy_price - stop_loss)
                reward = abs(sell_price - buy_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0
            elif recommendation == 'SELL' and sell_price and stop_loss:
                risk = abs(stop_loss - sell_price)
                reward = abs(current_price - sell_price)
                risk_reward_ratio = reward / risk if risk > 0 else 0
            
            # Ensure minimum risk-reward ratio dynamically based on volatility and stock characteristics
            if risk_reward_ratio < 2.0 and recommendation == 'BUY' and buy_price and stop_loss:
                # Calculate dynamic risk-reward ratio based on stock volatility
                volatility_pct = (atr / current_price) * 100
                min_ratio = self._calculate_dynamic_risk_reward_ratio(volatility_pct, rsi)
                
                # Adjust sell price to achieve dynamic minimum ratio
                risk = abs(buy_price - stop_loss)
                sell_price = buy_price + (risk * min_ratio)
                risk_reward_ratio = min_ratio
            
            # Ensure prices are realistic (no negative or zero prices)
            if buy_price and buy_price <= 0:
                buy_price = current_price
            if sell_price and sell_price <= 0:
                sell_price = current_price * 1.15
            if stop_loss and stop_loss <= 0:
                stop_loss = current_price * 0.92
            
            return {
                'recommendation': recommendation,
                'entry_timing': entry_timing,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'stop_loss': stop_loss,
                'risk_reward_ratio': round(risk_reward_ratio, 2)
            }
            
        except Exception as e:
            logger.error(f"Error generating buy/sell recommendations: {e}")
            return {
                'recommendation': 'HOLD',
                'entry_timing': 'WAIT',
                'buy_price': current_price,
                'sell_price': current_price * 1.15,  # Default 15% target
                'stop_loss': current_price * 0.92,   # Default 8% stop loss
                'risk_reward_ratio': 1.87  # Approximately 15%/8% = 1.87
            }
    
    def _estimate_days_to_target(self, current_price: float, target_price: float, atr: float) -> int:
        """
        Estimate days to reach target price using volatility (ATR).
        
        Args:
            current_price: Current stock price
            target_price: Target price
            atr: Average True Range (volatility measure)
            
        Returns:
            Estimated days to reach target
        """
        try:
            if not target_price or not current_price or current_price <= 0:
                logger.debug(f"ETA Debug: Invalid prices - target_price={target_price}, current_price={current_price} - returning 30 days")
                return 30
            
            # If prices are very close (within 1%), return short timeframe
            price_diff_pct = abs(target_price - current_price) / current_price
            if price_diff_pct < 0.01:  # Less than 1% difference
                logger.debug(f"ETA Debug: Very small price difference ({price_diff_pct:.2%}) - returning 7 days")
                return 7
            
            # Calculate the price move required
            price_move = abs(target_price - current_price)
            
            # Estimate daily volatility as a percentage
            daily_volatility = (atr / current_price) if current_price > 0 else 0.02
            
            # More realistic estimation approach
            # Calculate percentage move required
            percentage_move = price_move / current_price
            
            logger.debug(f"ETA Debug: current_price={current_price:.2f}, target_price={target_price:.2f}, atr={atr:.2f}")
            logger.debug(f"ETA Debug: price_move={price_move:.2f}, daily_volatility={daily_volatility:.4f}, percentage_move={percentage_move:.4f}")
            
            # Use a more conservative approach - stocks don't move linearly
            # For small moves (< 5%), estimate faster achievement
            # For larger moves (> 10%), estimate much slower achievement
            if percentage_move <= 0.05:  # <= 5% move
                base_days = 15 + (percentage_move * 200)  # 15-25 days for small moves
                move_category = "small (<= 5%)"
            elif percentage_move <= 0.10:  # 5-10% move
                base_days = 25 + ((percentage_move - 0.05) * 600)  # 25-55 days for medium moves
                move_category = "medium (5-10%)"
            else:  # > 10% move
                base_days = 55 + ((percentage_move - 0.10) * 300)  # 55+ days for large moves
                move_category = f"large (> 10%, actual: {percentage_move:.2%})"
            
            logger.debug(f"ETA Debug: move_category={move_category}, base_days={base_days:.1f}")
            
            # Adjust based on volatility
            volatility_adjustment = 1.0
            if daily_volatility > 0.03:  # High volatility (> 3% daily)
                days_estimate = int(base_days * 0.7)  # Faster in volatile markets
                volatility_adjustment = 0.7
                volatility_category = "high (> 3%)"
            elif daily_volatility < 0.015:  # Low volatility (< 1.5% daily)
                days_estimate = int(base_days * 1.3)  # Slower in stable markets
                volatility_adjustment = 1.3
                volatility_category = "low (< 1.5%)"
            else:
                days_estimate = int(base_days)
                volatility_category = "medium (1.5-3%)"
            
            logger.debug(f"ETA Debug: volatility_category={volatility_category}, volatility_adjustment={volatility_adjustment}, days_before_bounds={days_estimate}")
            
            # Ensure reasonable bounds - minimum 7 days, maximum 120 days
            final_days = max(7, min(days_estimate, 120))
            
            logger.debug(f"ETA Debug: final_days={final_days} (after bounds 7-120)")
            logger.info(f"ETA Calculation: {percentage_move:.2%} move, {daily_volatility:.2%} volatility -> {final_days} days")
            
            return final_days
            
        except Exception as e:
            logger.error(f"Error estimating days to target: {e}")
            return 30  # Default to 30 days
    
    def _calculate_confidence(self, current_price: float, sma_20: float, sma_50: float, 
                            rsi: float, recommendation: str) -> float:
        """
        Calculate confidence level for the recommendation.
        
        Args:
            current_price: Current stock price
            sma_20: 20-day Simple Moving Average
            sma_50: 50-day Simple Moving Average
            rsi: Relative Strength Index
            recommendation: Buy/Sell/Hold recommendation
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Price relative to moving averages
            if recommendation == 'BUY':
                if current_price > sma_20 > sma_50:
                    confidence += 0.2
                elif current_price > sma_20:
                    confidence += 0.1
                    
                # RSI signals
                if 40 < rsi < 60:
                    confidence += 0.2
                elif 30 < rsi < 70:
                    confidence += 0.1
                    
            elif recommendation == 'SELL':
                if current_price < sma_20 < sma_50:
                    confidence += 0.2
                elif current_price < sma_20:
                    confidence += 0.1
                    
                # RSI signals
                if rsi > 70:
                    confidence += 0.2
                elif rsi < 30:
                    confidence -= 0.1
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _perform_backtesting(self, symbol: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform backtesting on the strategies using historical data.
        
        Args:
            symbol: Stock symbol
            historical_data: Historical price data
            
        Returns:
            Dictionary containing backtesting results
        """
        try:
            # Initialize backtesting runner
            backtest_runner = BacktestingRunner()
            
            # Run comprehensive backtesting
            results = backtest_runner.run(symbol, historical_data)
            
            logger.info(f"Backtesting completed for {symbol}: {results.get('status', 'unknown')}")
            
            # Log key metrics if available
            if results.get('status') == 'completed':
                combined_metrics = results.get('combined_metrics', {})
                logger.info(f"Backtesting metrics for {symbol}: CAGR={combined_metrics.get('avg_cagr', 0)}%, "
                           f"Win Rate={combined_metrics.get('avg_win_rate', 0)}%, "
                           f"Max Drawdown={combined_metrics.get('avg_max_drawdown', 0)}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error performing backtesting for {symbol}: {e}")
            return {
                'error': str(e),
                'symbol': symbol,
                'status': 'error'
            }
    
    def _simulate_trading_strategy(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Simulate trading strategy on historical data.
        
        Args:
            data: Historical price data
            symbol: Stock symbol
            
        Returns:
            Dictionary containing trading simulation results
        """
        try:
            # Initialize trading simulation
            initial_capital = 100000  # 1 lakh
            cash = initial_capital
            position = 0
            trades = []
            portfolio_values = []
            
            # Calculate indicators for the entire period
            closes = data['Close'].values
            sma_20 = ta.SMA(closes, timeperiod=20)
            sma_50 = ta.SMA(closes, timeperiod=50)
            rsi = ta.RSI(closes, timeperiod=14)
            
            # Skip initial NaN values
            start_idx = 50  # Ensure we have enough data for all indicators
            
            for i in range(start_idx, len(data)):
                current_price = data['Close'].iloc[i]
                current_date = data.index[i]
                
                # Calculate current portfolio value
                portfolio_value = cash + (position * current_price)
                portfolio_values.append(portfolio_value)
                
                # Generate signals based on our strategy
                signal = self._generate_trading_signal(
                    current_price, sma_20[i], sma_50[i], rsi[i]
                )
                
                # Execute trades based on signals
                if signal == 'BUY' and position == 0 and cash > current_price:
                    # Buy signal - enter position
                    shares_to_buy = int(cash * 0.95 / current_price)  # Use 95% of available cash
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        cash -= cost
                        position = shares_to_buy
                        trades.append({
                            'date': current_date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': cost
                        })
                
                elif signal == 'SELL' and position > 0:
                    # Sell signal - exit position
                    proceeds = position * current_price
                    cash += proceeds
                    trades.append({
                        'date': current_date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': position,
                        'value': proceeds
                    })
                    position = 0
            
            # Calculate final portfolio value
            final_price = data['Close'].iloc[-1]
            final_portfolio_value = cash + (position * final_price)
            
            # Calculate performance metrics
            total_return = (final_portfolio_value - initial_capital) / initial_capital * 100
            
            # Calculate win rate
            winning_trades = 0
            total_trades = len([t for t in trades if t['action'] == 'SELL'])
            
            buy_price = None
            for trade in trades:
                if trade['action'] == 'BUY':
                    buy_price = trade['price']
                elif trade['action'] == 'SELL' and buy_price:
                    if trade['price'] > buy_price:
                        winning_trades += 1
                    buy_price = None
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate maximum drawdown
            max_drawdown = 0
            peak_value = initial_capital
            for value in portfolio_values:
                if value > peak_value:
                    peak_value = value
                drawdown = (peak_value - value) / peak_value * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate CAGR (Compound Annual Growth Rate)
            days_in_period = (data.index[-1] - data.index[start_idx]).days
            years = days_in_period / 365.25
            cagr = ((final_portfolio_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
            
            return {
                'initial_capital': initial_capital,
                'final_portfolio_value': final_portfolio_value,
                'total_return': round(total_return, 2),
                'cagr': round(cagr, 2),
                'win_rate': round(win_rate, 2),
                'max_drawdown': round(max_drawdown, 2),
                'total_trades': len(trades),
                'winning_trades': winning_trades,
                'period_days': days_in_period,
                'trades': trades[-10:] if len(trades) > 10 else trades  # Last 10 trades
            }
            
        except Exception as e:
            logger.error(f"Error simulating trading strategy: {e}")
            return {
                'error': str(e),
                'initial_capital': 100000,
                'final_portfolio_value': 100000,
                'total_return': 0,
                'cagr': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'total_trades': 0
            }
    
    def _generate_trading_signal(self, price: float, sma_20: float, sma_50: float, rsi: float) -> str:
        """
        Generate enhanced trading signal based on multiple technical indicators.
        
        Args:
            price: Current price
            sma_20: 20-day Simple Moving Average
            sma_50: 50-day Simple Moving Average
            rsi: Relative Strength Index
            
        Returns:
            Trading signal: 'BUY', 'SELL', or 'HOLD'
        """
        try:
            # Handle NaN values
            if pd.isna(sma_20) or pd.isna(sma_50) or pd.isna(rsi):
                return 'HOLD'
            
            # Enhanced buy signals with multiple confirmations
            buy_signals = 0
            sell_signals = 0
            
            # Trend signals
            if price > sma_20 > sma_50:  # Uptrend
                buy_signals += 2
            elif price > sma_20:  # Price above short MA
                buy_signals += 1
            elif price < sma_20 < sma_50:  # Downtrend
                sell_signals += 2
            elif price < sma_20:  # Price below short MA
                sell_signals += 1
            
            # RSI signals - more nuanced approach
            if 30 < rsi < 50:  # RSI recovering from oversold
                buy_signals += 1
            elif 40 < rsi < 60:  # RSI in neutral zone - good for continuation
                buy_signals += 0.5
            elif rsi > 70:  # Overbought - sell signal
                sell_signals += 1.5
            elif rsi < 30:  # Oversold - potential reversal but wait for confirmation
                buy_signals += 0.5
                
            # Momentum signals
            sma_slope = (sma_20 - sma_50) / sma_50 * 100  # Slope of MA difference
            if sma_slope > 2:  # Strong upward momentum
                buy_signals += 1
            elif sma_slope < -2:  # Strong downward momentum
                sell_signals += 1
                
            # Price position relative to moving averages
            price_above_ma20_pct = (price - sma_20) / sma_20 * 100
            if 1 < price_above_ma20_pct < 5:  # Price moderately above MA20
                buy_signals += 0.5
            elif price_above_ma20_pct > 8:  # Price too far above MA20
                sell_signals += 0.5
            elif -2 < price_above_ma20_pct < 1:  # Price near or slightly below MA20
                buy_signals += 0.3
                
            # Decision logic with threshold
            if buy_signals >= 2.5 and sell_signals < 1.5:
                return 'BUY'
            elif sell_signals >= 2.0 and buy_signals < 1.0:
                return 'SELL'
            else:
                return 'HOLD'
            
        except Exception as e:
            logger.error(f"Error generating trading signal: {e}")
            return 'HOLD'
    
    def _calculate_overall_backtest_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall performance metrics from all backtest periods.
        
        Args:
            backtest_results: Results from different time periods
            
        Returns:
            Dictionary containing overall metrics
        """
        try:
            if not backtest_results:
                return {}
            
            # Calculate averages across all periods
            avg_cagr = sum(r.get('cagr', 0) for r in backtest_results.values()) / len(backtest_results)
            avg_win_rate = sum(r.get('win_rate', 0) for r in backtest_results.values()) / len(backtest_results)
            avg_max_drawdown = sum(r.get('max_drawdown', 0) for r in backtest_results.values()) / len(backtest_results)
            total_trades = sum(r.get('total_trades', 0) for r in backtest_results.values())
            
            # Find best and worst performing periods
            best_period = max(backtest_results.keys(), key=lambda k: backtest_results[k].get('cagr', 0))
            worst_period = min(backtest_results.keys(), key=lambda k: backtest_results[k].get('cagr', 0))
            
            return {
                'average_cagr': round(avg_cagr, 2),
                'average_win_rate': round(avg_win_rate, 2),
                'average_max_drawdown': round(avg_max_drawdown, 2),
                'total_trades_all_periods': total_trades,
                'best_performing_period': best_period,
                'worst_performing_period': worst_period,
                'best_period_cagr': round(backtest_results[best_period].get('cagr', 0), 2),
                'worst_period_cagr': round(backtest_results[worst_period].get('cagr', 0), 2),
                'consistency_score': round(100 - (max(r.get('max_drawdown', 0) for r in backtest_results.values())), 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall backtest metrics: {e}")
            return {}
    
    def _calculate_dynamic_hold_targets(self, current_price: float, atr: float, rsi: float,
                                       support: float, resistance: float, bb_upper: float, 
                                       bb_lower: float) -> Dict[str, float]:
        """
        Calculate dynamic buy/sell targets for HOLD recommendations based on multiple factors.
        
        Args:
            current_price: Current stock price
            atr: Average True Range (volatility)
            rsi: Relative Strength Index
            support: Support level
            resistance: Resistance level
            bb_upper: Bollinger Band upper
            bb_lower: Bollinger Band lower
            
        Returns:
            Dictionary with dynamic buy_price, sell_price, and stop_loss
        """
        try:
            # Calculate volatility percentage
            volatility_pct = (atr / current_price) * 100
            
            # Base adjustments based on volatility
            if volatility_pct > 4.0:  # High volatility stock
                buy_discount = np.random.uniform(0.08, 0.12)  # 8-12% discount
                sell_premium = np.random.uniform(0.18, 0.25)  # 18-25% target
                stop_discount = np.random.uniform(0.12, 0.18)  # 12-18% stop loss
                volatility_factor = "high"
            elif volatility_pct > 2.5:  # Medium volatility stock
                buy_discount = np.random.uniform(0.05, 0.08)  # 5-8% discount
                sell_premium = np.random.uniform(0.12, 0.18)  # 12-18% target
                stop_discount = np.random.uniform(0.08, 0.12)  # 8-12% stop loss
                volatility_factor = "medium"
            else:  # Low volatility stock
                buy_discount = np.random.uniform(0.03, 0.06)  # 3-6% discount
                sell_premium = np.random.uniform(0.08, 0.15)  # 8-15% target
                stop_discount = np.random.uniform(0.05, 0.08)  # 5-8% stop loss
                volatility_factor = "low"
            
            # Adjust based on RSI conditions
            if rsi > 65:  # Overbought conditions
                buy_discount += 0.03  # Wait for bigger dip
                sell_premium *= 0.85  # Lower target
                rsi_adjustment = "overbought"
            elif rsi < 35:  # Oversold conditions
                buy_discount *= 0.7  # Don't wait for as big a dip
                sell_premium += 0.03  # Higher target potential
                rsi_adjustment = "oversold"
            else:
                rsi_adjustment = "neutral"
            
            # Incorporate support/resistance levels
            support_distance = abs(current_price - support) / current_price
            resistance_distance = abs(resistance - current_price) / current_price
            
            # If close to support, adjust buy price
            if support_distance < 0.05:  # Within 5% of support
                buy_price = support * 1.005  # Buy slightly above support
                support_factor = "near_support"
            else:
                buy_price = current_price * (1 - buy_discount)
                support_factor = "normal"
            
            # If close to resistance, adjust sell price
            if resistance_distance < 0.08:  # Within 8% of resistance
                sell_price = min(resistance * 1.02, current_price * (1 + sell_premium))
                resistance_factor = "near_resistance"
            else:
                sell_price = current_price * (1 + sell_premium)
                resistance_factor = "normal"
            
            # Calculate stop loss
            stop_loss = current_price * (1 - stop_discount)
            
            # Add some randomization to avoid identical results
            randomization_factor = np.random.uniform(0.98, 1.02)  # ±2% randomization
            sell_price *= randomization_factor
            
            # Ensure minimum and maximum bounds
            buy_price = max(buy_price, current_price * 0.85)  # Max 15% discount
            sell_price = min(sell_price, current_price * 1.35)  # Max 35% target
            stop_loss = max(stop_loss, current_price * 0.75)  # Max 25% stop loss
            
            # Log the dynamic calculation for debugging
            logger.debug(f"Dynamic HOLD targets: price={current_price:.2f}, volatility={volatility_factor} ({volatility_pct:.2f}%), "
                        f"RSI={rsi_adjustment} ({rsi:.1f}), support={support_factor}, resistance={resistance_factor}")
            logger.debug(f"Targets: buy={buy_price:.2f} ({((current_price-buy_price)/current_price*100):.1f}% discount), "
                        f"sell={sell_price:.2f} ({((sell_price-current_price)/current_price*100):.1f}% target), "
                        f"stop={stop_loss:.2f} ({((current_price-stop_loss)/current_price*100):.1f}% stop)")
            
            return {
                'buy_price': buy_price,
                'sell_price': sell_price,
                'stop_loss': stop_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic HOLD targets: {e}")
            # Fallback to basic calculation with some randomization
            base_discount = np.random.uniform(0.04, 0.07)  # 4-7% discount
            base_premium = np.random.uniform(0.10, 0.16)   # 10-16% target
            base_stop = np.random.uniform(0.08, 0.12)      # 8-12% stop
            
            return {
                'buy_price': current_price * (1 - base_discount),
                'sell_price': current_price * (1 + base_premium),
                'stop_loss': current_price * (1 - base_stop)
            }
    
    def _calculate_dynamic_risk_reward_ratio(self, volatility_pct: float, rsi: float) -> float:
        """
        Calculate dynamic minimum risk-reward ratio based on stock characteristics.
        
        Args:
            volatility_pct: Stock volatility as percentage
            rsi: Relative Strength Index
            
        Returns:
            Dynamic minimum risk-reward ratio
        """
        try:
            # Base ratio starts at 2.0
            base_ratio = 2.0
            
            # Adjust based on volatility
            if volatility_pct > 4.0:  # High volatility - can accept lower ratio
                volatility_adjustment = np.random.uniform(1.8, 2.2)
            elif volatility_pct > 2.5:  # Medium volatility
                volatility_adjustment = np.random.uniform(2.2, 2.8)
            else:  # Low volatility - need higher ratio
                volatility_adjustment = np.random.uniform(2.5, 3.2)
            
            # Adjust based on RSI (momentum factor)
            if rsi > 65:  # Overbought - need higher ratio for safety
                rsi_adjustment = 1.15
            elif rsi < 35:  # Oversold - can accept lower ratio for quick gains
                rsi_adjustment = 0.85
            else:  # Neutral
                rsi_adjustment = 1.0
                
            # Calculate final ratio with some randomization
            final_ratio = volatility_adjustment * rsi_adjustment
            final_ratio *= np.random.uniform(0.95, 1.05)  # ±5% randomization
            
            # Ensure reasonable bounds
            final_ratio = max(1.5, min(final_ratio, 4.0))
            
            logger.debug(f"Dynamic risk-reward ratio: volatility={volatility_pct:.2f}%, RSI={rsi:.1f} -> ratio={final_ratio:.2f}")
            
            return round(final_ratio, 2)
            
        except Exception as e:
            logger.error(f"Error calculating dynamic risk-reward ratio: {e}")
            return 2.5  # Default fallback
    
    def get_analyzer_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the analyzer's capabilities.
        
        Returns:
            Dictionary containing analyzer summary
        """
        strategy_summary = self.strategy_evaluator.get_strategy_summary()
        
        return {
            'technical_analysis': {
                'total_strategies': strategy_summary['total_loaded'],
                'strategies': strategy_summary['loaded_strategies']
            },
            'fundamental_analysis': {
                'enabled': True,
                'metrics': ['P/E Ratio', 'P/B Ratio', 'Debt-to-Equity', 'EPS Growth', 'Revenue Growth', 'Dividend Yield']
            },
            'sentiment_analysis': {
                'enabled': True,
                'model': self.sentiment_analyzer.model_name,
                'news_sources': ['Google News']
            },
            'buy_sell_recommendations': {
                'enabled': True,
                'features': ['Buy/Sell prices', 'Stop loss calculation', 'Risk-reward ratios', 'Entry timing', 'Days to target estimation']
            }
        }


# Convenience function for single stock analysis
def analyze_stock(symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a single stock using all analysis types.
    
    Args:
        symbol: Stock symbol
        app_config: Application configuration
        
    Returns:
        Analysis results dictionary
    """
    analyzer = StockAnalyzer()
    return analyzer.analyze_stock(symbol, app_config)


# Convenience function for buy/sell recommendations
def analyze(symbol: str) -> Dict[str, Any]:
    """
    Analyze a single stock for buy/sell recommendations and timing.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary with buy/sell recommendations and timing
    """
    analyzer = StockAnalyzer()
    return analyzer.analyze(symbol)
