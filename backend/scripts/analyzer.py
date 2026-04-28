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
from scripts.swing_trading_signals import SwingTradingSignalAnalyzer
from utils.logger import setup_logging
from config import HISTORICAL_DATA_PERIOD, ANALYSIS_WEIGHTS, RECOMMENDATION_THRESHOLDS, ANALYSIS_CONFIG
from scripts.risk_management import RiskManager
from scripts.sector_analysis import SectorAnalyzer
from scripts.backtesting_runner import BacktestingRunner
from scripts.market_regime_detection import MarketRegimeDetection
from scripts.market_microstructure import MarketMicrostructureAnalyzer
from scripts.alternative_data_analyzer import AlternativeDataAnalyzer
from scripts.predictor import PricePredictor
from scripts.rl_trading_agent import RLTradingAgent
from scripts.trade_logic import TradeLogic
from scripts.backtest_utils import BacktestUtils

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
        
        self.strategy_evaluator = StrategyEvaluator()
        self.fundamental_analyzer = FundamentalAnalysis()
        self.sentiment_analyzer = None
        self.sector_analyzer = None
        self.risk_manager = RiskManager()
        self.trade_logic = TradeLogic()
        self.swing_analyzer = SwingTradingSignalAnalyzer()
        self.backtest_utils = BacktestUtils()
        
        logger.info("StockAnalyzer initialization complete")
        
    def analyze_stock(self, symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive analysis on a stock."""
        try:
            # Basic metadata
            all_symbols = get_all_nse_symbols()
            company_name = all_symbols.get(symbol, symbol)
            logger.info(f"Analyzing {symbol} ({company_name})")
            
            result = {
                'symbol': symbol, 'company_name': company_name,
                'technical_score': 0.0, 'fundamental_score': 0.0, 'sentiment_score': 0.0,
                'is_recommended': False, 'reason': [], 'detailed_analysis': {}
            }
            
            # 1. Technical Analysis
            try:
                fresh = app_config.get('FRESH_DATA', False)
                period = app_config.get('HISTORICAL_DATA_PERIOD', HISTORICAL_DATA_PERIOD)
                historical_data = get_historical_data(symbol, period, fresh=fresh)
                
                if historical_data.empty:
                    result['reason'].append("No historical data available")
                    result['technical_score'] = -1.0
                    tech_analysis = {'error': 'No data'}
                else:
                    tech_analysis = self.strategy_evaluator.evaluate_strategies(symbol, historical_data)
                    # Simplified linear mapping: Map 0-1 range to roughly -1 to 1 range
                    # 0 signal -> -0.5, 0.5 signal -> 0.0, 1.0 signal -> 1.0
                    raw = tech_analysis['technical_score']
                    result['technical_score'] = (raw * 1.5) - 0.5
            except Exception as e:
                logger.error(f"Tech error {symbol}: {e}")
                result['technical_score'] = -1.0
                tech_analysis = {'error': str(e)}
            # 1.5 Swing Trading Gates (The "ignored" config reconnection)
            swing_gate_res = self.swing_analyzer.analyze_swing_opportunity(symbol, historical_data)
            result['swing_analysis'] = swing_gate_res
            
            # Incorporate swing results into technical score if enabled in RECOMMENDATION_THRESHOLDS
            if RECOMMENDATION_THRESHOLDS.get('require_all_gates', True) and not swing_gate_res.get('all_gates_passed', False):
                # Only penalize if technical score was positive
                if result['technical_score'] > 0:
                    result['technical_score'] = min(result['technical_score'], 0.1)  # Penalize if gates fail
                
                failed_gates = [k for k, v in swing_gate_res.get('gates_passed', {}).items() if not v]
                if failed_gates:
                    result['reason'].append(f"Gate failure: {', '.join(failed_gates)}")

            # Smart Money: Delivery Volume Check
            try:
                from scripts.smart_money_tracker import SmartMoneyTracker
                tracker = SmartMoneyTracker()
                delivery_pct = tracker.get_delivery_volume(symbol)
                result['delivery_volume_pct'] = delivery_pct
                
                # Bonus for high delivery accumulation (Institutional Buying)
                if delivery_pct > 60:
                    result['technical_score'] += 0.15
                    result['reason'].append(f"MASSIVE Institutional Accumulation (Delivery: {delivery_pct:.1f}%)")
                elif delivery_pct > 40:
                    result['technical_score'] += 0.05
                    result['reason'].append(f"Strong Delivery Volume ({delivery_pct:.1f}%)")
                
            except Exception as e:
                logger.warning(f"Failed to fetch delivery volume for {symbol}: {e}")
                result['delivery_volume_pct'] = 0.0

            result['detailed_analysis']['technical'] = tech_analysis
            result['detailed_analysis']['swing_gates'] = swing_gate_res
            result['detailed_analysis']['smart_money'] = {'delivery_pct': result.get('delivery_volume_pct', 0)}
            
            # 2. Fundamental & Sentiment (Optional)
            # Use top-level config or ANALYSIS_CONFIG nested values
            skip_fundamental = (app_config.get('SKIP_FUNDAMENTAL') or 
                              not app_config.get('ANALYSIS_CONFIG', {}).get('fundamental_analysis', True))
            
            if not skip_fundamental:
                result['fundamental_score'] = self.fundamental_analyzer.perform_fundamental_analysis(symbol)
                result['reason'].append("Fundamental signals included")
            else:
                result['fundamental_score'] = 0.0  # Neutral when skipped
            
            skip_sentiment = (app_config.get('SKIP_SENTIMENT') or 
                            not app_config.get('ANALYSIS_CONFIG', {}).get('sentiment_analysis', True))
            
            if not skip_sentiment:
                if not self.sentiment_analyzer: self.sentiment_analyzer = SentimentAnalysis()
                result['sentiment_score'] = self.sentiment_analyzer.perform_sentiment_analysis(company_name)
                result['reason'].append("Sentiment signals included")
            else:
                result['sentiment_score'] = 0.0  # Neutral when skipped
            
            # 3. Sector & Market Modules
            analysis_config = app_config.get('ANALYSIS_CONFIG', {})
            if analysis_config.get('sector_analysis'):
                if not self.sector_analyzer: self.sector_analyzer = SectorAnalyzer()
                # Get comprehensive analysis
                sector_res = self.sector_analyzer.get_comprehensive_sector_analysis(symbol)
                result['sector_analysis'] = sector_res
                # Use the sector score (-1 to 1) 
                result['sector_score'] = sector_res.get('sector_score', 0.0)
                
                if RECOMMENDATION_THRESHOLDS.get('sector_filter_enabled', False):
                    min_sector_score = RECOMMENDATION_THRESHOLDS.get('min_sector_score', -0.2)
                    if result['sector_score'] < min_sector_score:
                        result['technical_score'] -= 0.2  # Penalize for bad sector
                        result['reason'].append(f"Weak Sector Performance: {sector_res.get('sector', 'Unknown')}")
                    elif result['sector_score'] > 0.4:
                        result['technical_score'] += 0.1  # Bonus for leading sector
                        result['reason'].append(f"Leading Sector Tailwinds: {sector_res.get('sector', 'Unknown')}")
            else:
                result['sector_score'] = 0.0
            # 4. Preliminary Combination
            result = self._combine_analysis_results(result, consider_backtest=False, keep_reason_as_list=True)
            
            # 5. Trade & Backtest Delegation
            if not historical_data.empty:
                # Delegate trade plan
                trade_res = self.trade_logic.analyze(symbol, historical_data)
                result['trade_plan'] = {
                    'buy_price': trade_res.get('buy_price', 0.0),
                    'sell_price': trade_res.get('sell_price', 0.0),
                    'stop_loss': trade_res.get('stop_loss', 0.0),
                    'days_to_target': trade_res.get('days_to_target', 0),
                    'entry_timing': trade_res.get('entry_timing', 'WAIT'),
                    'risk_reward_ratio': trade_res.get('risk_reward_ratio', 0.0),
                    'confidence': trade_res.get('confidence', 0.0)
                }
                
                # Delegate Backtesting
                bt_res = self.backtest_utils.perform_backtesting(symbol, historical_data)
                result['backtest'] = bt_res
                if bt_res.get('status') == 'completed':
                    metrics = bt_res.get('combined_metrics', {})
                    result['reason'].append(f"Backtest: CAGR={metrics.get('avg_cagr', 0)}%, Win={metrics.get('avg_win_rate', 0)}%")
                
                # 6. Final recommendation considering backtest
                if result.get('is_recommended'):
                    result = self._combine_analysis_results(result, consider_backtest=True, keep_reason_as_list=True)
                
                # 7. Risk Management
                if result.get('is_recommended'):
                    cp = historical_data['Close'].iloc[-1]
                    sl = self.risk_manager.calculate_stop_loss(historical_data, cp)
                    result['risk_management'] = {
                        'current_price': cp, 'stop_loss': sl,
                        'position_sizing': self.risk_manager.calculate_position_size(cp, sl['stop_loss']),
                        'profit_targets': self.risk_manager.calculate_profit_targets(cp, sl['stop_loss']),
                        'pivot_points': self.risk_manager.calculate_pivot_points(historical_data)
                    }
            
            if isinstance(result['reason'], list): result['reason'] = " ".join(result['reason'])
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {'symbol': symbol, 'is_recommended': False, 'error': str(e)}
    
    def analyze_stock_with_data(self, symbol: str, company_name: str, 
                                 historical_data: pd.DataFrame, app_config: Dict[str, Any],
                                 index_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze a stock using pre-fetched historical data.
        Used by the multiprocessing pipeline where data is fetched in Phase 1.
        
        Args:
            symbol: Stock symbol
            company_name: Human-readable company name
            historical_data: Pre-fetched DataFrame with OHLCV data
            app_config: Application config dictionary
        """
        try:
            logger.info(f"Analyzing {symbol} ({company_name}) [with pre-fetched data]")
            
            # Ensure strategy evaluator is using the latest config from app_config
            if 'STRATEGY_CONFIG' in app_config:
                self.strategy_evaluator.strategy_config = app_config['STRATEGY_CONFIG']
                self.strategy_evaluator.load_strategies()
            
            result = {
                'symbol': symbol, 'company_name': company_name,
                'technical_score': 0.0, 'fundamental_score': 0.0, 'sentiment_score': 0.0,
                'is_recommended': False, 'reason': [], 'detailed_analysis': {}
            }
            
            # 1. Technical Analysis
            try:
                if historical_data.empty:
                    result['reason'].append("No historical data available")
                    result['technical_score'] = -1.0
                    tech_analysis = {'error': 'No data'}
                else:
                    tech_analysis = self.strategy_evaluator.evaluate_strategies(symbol, historical_data, index_data=index_data)
                    result['technical_score'] = tech_analysis['technical_score'] # Linear 0.0 to 1.0
            except Exception as e:
                logger.error(f"Tech error {symbol}: {e}")
                result['technical_score'] = -1.0
                tech_analysis = {'error': str(e)}
            result['detailed_analysis']['technical'] = tech_analysis
            
            # 2. Fundamental & Sentiment (Optional)
            skip_fundamental = (app_config.get('SKIP_FUNDAMENTAL') or 
                              not app_config.get('ANALYSIS_CONFIG', {}).get('fundamental_analysis', True))
            if not skip_fundamental:
                result['fundamental_score'] = self.fundamental_analyzer.perform_fundamental_analysis(symbol)
                result['reason'].append("Fundamental signals included")
            else:
                result['fundamental_score'] = 0.0
            
            skip_sentiment = (app_config.get('SKIP_SENTIMENT') or 
                            not app_config.get('ANALYSIS_CONFIG', {}).get('sentiment_analysis', True))
            if not skip_sentiment:
                if not self.sentiment_analyzer: self.sentiment_analyzer = SentimentAnalysis()
                result['sentiment_score'] = self.sentiment_analyzer.perform_sentiment_analysis(company_name)
                result['reason'].append("Sentiment signals included")
            else:
                result['sentiment_score'] = 0.0
            
            # 3. Preliminary Combination
            result = self._combine_analysis_results(result, consider_backtest=False, keep_reason_as_list=True)
            
            # 4. Trade & Backtest
            if not historical_data.empty:
                trade_res = self.trade_logic.analyze(symbol, historical_data)
                result['trade_plan'] = {
                    'buy_price': trade_res.get('buy_price', 0.0),
                    'sell_price': trade_res.get('sell_price', 0.0),
                    'stop_loss': trade_res.get('stop_loss', 0.0),
                    'days_to_target': trade_res.get('days_to_target', 0),
                    'entry_timing': trade_res.get('entry_timing', 'WAIT'),
                    'risk_reward_ratio': trade_res.get('risk_reward_ratio', 0.0),
                    'confidence': trade_res.get('confidence', 0.0)
                }
                
                bt_res = self.backtest_utils.perform_backtesting(symbol, historical_data)
                result['backtest'] = bt_res
                if bt_res.get('status') == 'completed':
                    metrics = bt_res.get('combined_metrics', {})
                    result['reason'].append(f"Backtest: CAGR={metrics.get('avg_cagr', 0)}%, Win={metrics.get('avg_win_rate', 0)}%")
                
                if result.get('is_recommended'):
                    result = self._combine_analysis_results(result, consider_backtest=True, keep_reason_as_list=True)
                
                if result.get('is_recommended'):
                    cp = historical_data['Close'].iloc[-1]
                    sl = self.risk_manager.calculate_stop_loss(historical_data, cp)
                    result['risk_management'] = {
                        'current_price': cp, 'stop_loss': sl,
                        'position_sizing': self.risk_manager.calculate_position_size(cp, sl['stop_loss']),
                        'profit_targets': self.risk_manager.calculate_profit_targets(cp, sl['stop_loss']),
                        'pivot_points': self.risk_manager.calculate_pivot_points(historical_data)
                    }
            
            if isinstance(result['reason'], list): result['reason'] = " ".join(result['reason'])
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} with pre-fetched data: {e}")
            return {'symbol': symbol, 'is_recommended': False, 'error': str(e)}
    
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
        
        # Get configurable weights — only use non-zero (active) pillars
        technical_weight = ANALYSIS_WEIGHTS.get('technical', 0.5)
        fundamental_weight = ANALYSIS_WEIGHTS.get('fundamental', 0.3)
        sentiment_weight = ANALYSIS_WEIGHTS.get('sentiment', 0.0)
        
        # Normalize only active weights (non-zero) to sum to 1
        active_weights = {}
        if technical_weight > 0: active_weights['technical'] = technical_weight
        if fundamental_weight > 0: active_weights['fundamental'] = fundamental_weight
        if sentiment_weight > 0: active_weights['sentiment'] = sentiment_weight
        
        total_weight = sum(active_weights.values())
        if total_weight > 0 and total_weight != 1.0:
            for k in active_weights:
                active_weights[k] /= total_weight
        
        technical_weight = active_weights.get('technical', 0.0)
        fundamental_weight = active_weights.get('fundamental', 0.0)
        sentiment_weight = active_weights.get('sentiment', 0.0)
        
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
        sentiment_negative_threshold = RECOMMENDATION_THRESHOLDS.get('sentiment_negative', -0.2)
        sentiment_cap_positive = RECOMMENDATION_THRESHOLDS.get('sentiment_cap_positive', 0.5)
        sentiment_cap_negative = RECOMMENDATION_THRESHOLDS.get('sentiment_cap_negative', -0.5)
        market_trend_weight = RECOMMENDATION_THRESHOLDS.get('market_trend_weight', 0.0)
        
        # Penalize if sentiment is significantly negative
        if sentiment_score < sentiment_negative_threshold:
            combined_score -= abs(sentiment_score) * 0.5
            result['reason'].append(f"Penalized due to negative sentiment ({sentiment_score:.2f})")
            
        # Cap sentiment influence
        sentiment_score = max(min(sentiment_score, sentiment_cap_positive), sentiment_cap_negative)

        # Adjust score based on market trend only if explicitly calculated
        # Do not use a default value (1.0) as it artificially inflates the score when the module is disabled
        if 'market_trend' in result and market_trend_weight > 0:
            market_trend_score = result['market_trend']
            combined_score += (market_trend_score - 0.5) * market_trend_weight
            
        result['combined_score'] = combined_score
        
        # Recommendation logic with flexible backtest requirements
        if consider_backtest:
            # Get backtest CAGR and trade plan ETA
            backtest_cagr = result.get('backtest', {}).get('combined_metrics', {}).get('avg_cagr', 0)
            trade_plan = result.get('trade_plan', {})
            days_to_target = trade_plan.get('days_to_target', 0) if trade_plan else 0
            
            # Strict override — both pillars must be strong
            strong_analysis_override = (
                (technical_score > 0.3 and fundamental_score > 0.3)  # Both pillars strong
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
        
        # Get gate toggle
        require_all_gates = RECOMMENDATION_THRESHOLDS.get('require_all_gates', True)
        
        # Enhanced recommendation logic - more flexible and nuanced
        technical_minimum = RECOMMENDATION_THRESHOLDS.get('technical_minimum', 0.1)
        fundamental_minimum = RECOMMENDATION_THRESHOLDS.get('fundamental_minimum', 0.0)
        
        # Check which analysis modules are actually enabled
        fundamental_enabled = ANALYSIS_CONFIG.get('fundamental_analysis', False)
        
        # Core gates check
        passes_gates = True
        hold_reasons = []
        if require_all_gates:
            if technical_score < technical_minimum:
                passes_gates = False
                hold_reasons.append(
                    f"technical_score {technical_score:.3f} < minimum {technical_minimum:.3f}"
                )
            # Only enforce fundamental gate when fundamental_analysis is enabled.
            # When disabled, score is 0.0 by design — don't block on a skipped module.
            if fundamental_enabled and fundamental_score < fundamental_minimum:
                passes_gates = False
                hold_reasons.append(
                    f"fundamental_score {fundamental_score:.3f} < minimum {fundamental_minimum:.3f}"
                )
            
            # 1. Sector filter integration
            if RECOMMENDATION_THRESHOLDS.get('sector_filter_enabled', False):
                sector_data = result.get('sector_analysis', {})
                sector_score = sector_data.get('score', 0)
                min_sector_score = RECOMMENDATION_THRESHOLDS.get('min_sector_score', -0.5)
                if sector_score < min_sector_score:
                    passes_gates = False
                    hold_reasons.append(f"sector_score {sector_score:.2f} < minimum {min_sector_score:.2f}")
            
            # 2. Enhanced Volume confirmation check
            if RECOMMENDATION_THRESHOLDS.get('volume_confirmation_required', True):
                swing_gates = result.get('detailed_analysis', {}).get('swing_gates', {})
                vol_gate_passed = swing_gates.get('gates_passed', {}).get('volume_confirmation', True)
                
                # Check specific confidence threshold if available from technical analysis
                tech_vol_score = result.get('detailed_analysis', {}).get('technical', {}).get('volume_score', 1.0)
                vol_threshold = RECOMMENDATION_THRESHOLDS.get('volume_confidence_threshold', 0.45)
                
                if not vol_gate_passed or tech_vol_score < vol_threshold:
                    passes_gates = False
                    reason_msg = "Volume gate failed" if not vol_gate_passed else f"Volume confidence {tech_vol_score:.2f} < {vol_threshold}"
                    hold_reasons.append(reason_msg)
        
        # 3. Risk-Reward Ratio Gate (using trade_plan if available)
        if result.get('trade_plan'):
            rr_ratio = result['trade_plan'].get('risk_reward_ratio', 0)
            min_rr = RECOMMENDATION_THRESHOLDS.get('min_risk_reward_ratio', 1.8)
            if rr_ratio < min_rr:
                # We don't necessarily block the trade, but we downgrade it
                result['reason'].append(f"Low risk-reward ratio ({rr_ratio:.2f} < {min_rr})")
        
        # Strong buy: Multiple positive indicators with good backtest
        if passes_gates and ((technical_score > 0.3 and fundamental_score > 0.3 and sentiment_score > 0) or 
            (combined_score > strong_buy_threshold) or
            (technical_score > 0.4 and fundamental_score > 0.1) or
            (fundamental_score > 0.5 and technical_score > 0.1)) and backtest_condition:
            result['is_recommended'] = True
            result['recommendation_strength'] = 'STRONG_BUY'
            if technical_score > 0.3 and fundamental_score > 0.3 and sentiment_score > 0:
                result['reason'].append("All analysis types show strong positive signals")
            else:
                result['reason'].append("Superior combined score indicates strong buy opportunity")
        
        # Regular buy: Must pass gates
        elif passes_gates and combined_score >= buy_threshold:
            result['is_recommended'] = True
            result['recommendation_strength'] = 'BUY'
            result['reason'].append("Analysis meets buy criteria with acceptable combined score")
        
        
        else:
            result['is_recommended'] = False
            # Changed from showing SELL to always showing HOLD - we only provide BUY recommendations
            result['recommendation_strength'] = 'HOLD'
            if consider_backtest and backtest_cagr < min_backtest_return:
                result['reason'].append(f"Backtest CAGR ({backtest_cagr:.2f}%) below minimum threshold ({min_backtest_return:.2f}%)")
                hold_reasons.append(
                    f"backtest_cagr {backtest_cagr:.2f} < minimum {min_backtest_return:.2f}"
                )
            else:
                result['reason'].append("Analysis does not support buying at this time")
            if combined_score < buy_threshold:
                hold_reasons.append(
                    f"combined_score {combined_score:.3f} < buy threshold {buy_threshold:.3f}"
                )
            if not passes_gates and not hold_reasons:
                hold_reasons.append("core gates not satisfied")

        decision_debug = {
            'technical_score': round(technical_score, 4),
            'fundamental_score': round(fundamental_score, 4),
            'sentiment_score': round(sentiment_score, 4),
            'combined_score': round(combined_score, 4),
            'buy_threshold': round(buy_threshold, 4),
            'strong_buy_threshold': round(strong_buy_threshold, 4),
            'technical_minimum': round(technical_minimum, 4),
            'fundamental_minimum': round(fundamental_minimum, 4),
            'passes_gates': passes_gates,
            'backtest_condition': backtest_condition,
            'backtest_cagr': round(backtest_cagr, 4),
            'min_backtest_return': round(min_backtest_return, 4),
            'min_risk_reward': RECOMMENDATION_THRESHOLDS.get('min_risk_reward_ratio', 1.8),
            'sector_filter_enabled': RECOMMENDATION_THRESHOLDS.get('sector_filter_enabled', False),
            'hold_reasons': hold_reasons,
        }
        result['decision_debug'] = decision_debug

        logger.info(
            "Decision for %s: strength=%s combined=%.3f tech=%.3f fund=%.3f passes_gates=%s backtest_ok=%s reasons=%s",
            result.get('symbol', 'UNKNOWN'),
            result.get('recommendation_strength', 'UNKNOWN'),
            combined_score,
            technical_score,
            fundamental_score,
            passes_gates,
            backtest_condition,
            "; ".join(hold_reasons) if hold_reasons else "buy/strong_buy",
        )

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
    
    def get_analyzer_summary(self) -> Dict[str, Any]:
        """Get a summary of the analyzer's capabilities."""
        summary = self.strategy_evaluator.get_strategy_summary()
        return {
            'technical_analysis': {'total_strategies': summary['total_loaded']},
            'fundamental_analysis': {'enabled': True},
            'sentiment_analysis': {'enabled': True}
        }

def analyze_stock(symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single stock using all analysis types."""
    return StockAnalyzer().analyze_stock(symbol, app_config)

def analyze(symbol: str) -> Dict[str, Any]:
    """Backward compatible analysis function."""
    res = StockAnalyzer().analyze_stock(symbol, {})
    return {**res.get('trade_plan', {}), 'symbol': symbol, 'is_recommended': res.get('is_recommended')}
