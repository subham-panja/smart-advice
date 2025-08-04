"""
Strategy Evaluator for Technical Analysis
File: scripts/strategy_evaluator.py

This module evaluates multiple trading strategies and combines their signals
to generate overall technical analysis scores.
"""

import pandas as pd
import importlib
from typing import Dict, List, Any
from utils.logger import setup_logging
from config import STRATEGY_CONFIG, MIN_RECOMMENDATION_SCORE

logger = setup_logging()

class StrategyEvaluator:
    """
    Evaluates multiple trading strategies and combines their signals.
    """
    
    def __init__(self, strategy_config: Dict[str, bool] = None):
        """
        Initialize the strategy evaluator.
        
        Args:
            strategy_config: Dictionary of strategy names and their enabled status
        """
        self.strategy_config = strategy_config or STRATEGY_CONFIG
        self.strategy_instances = {}
        self.load_strategies()
        
    def load_strategies(self):
        """
        Load and initialize all enabled strategies.
        """
        strategy_mapping = {
            # Core strategies
            'MA_Crossover_50_200': 'scripts.strategies.ma_crossover_50_200',
            'RSI_Overbought_Oversold': 'scripts.strategies.rsi_overbought_oversold',
            'MACD_Signal_Crossover': 'scripts.strategies.macd_signal_crossover',
            'Bollinger_Band_Breakout': 'scripts.strategies.bollinger_band_breakout',
            'EMA_Crossover_12_26': 'scripts.strategies.ema_crossover_12_26',
            'Stochastic_Overbought_Oversold': 'scripts.strategies.stochastic_overbought_oversold',
            'ADX_Trend_Strength': 'scripts.strategies.adx_trend_strength',
            
            # High-priority swing trading strategies
            'Volume_Breakout': 'scripts.strategies.volume_breakout',
            'Support_Resistance_Breakout': 'scripts.strategies.support_resistance_breakout',
            'Fibonacci_Retracement': 'scripts.strategies.fibonacci_retracement',
            'Multi_Timeframe_RSI': 'scripts.strategies.multi_timeframe_rsi',
            
            # Advanced strategies (newly enabled)
            'DEMA_Crossover': 'scripts.strategies.dema_crossover',
            'Gap_Trading': 'scripts.strategies.gap_trading',
            'Channel_Trading': 'scripts.strategies.channel_trading',
            
            # PHASE 1 ADVANCED PATTERN RECOGNITION STRATEGIES
            'Chart_Patterns': 'scripts.strategies.chart_patterns',
            'Volume_Profile': 'scripts.strategies.volume_profile',
            
            # Newly implemented strategies
            'SMA_Crossover_20_50': 'scripts.strategies.sma_crossover_20_50',
            'Williams_Percent_R_Overbought_Oversold': 'scripts.strategies.williams_percent_r_strategy',
            'Volume_Price_Trend': 'scripts.strategies.volume_price_trend',
            'On_Balance_Volume': 'scripts.strategies.on_balance_volume',
            'Momentum_Oscillator': 'scripts.strategies.momentum_oscillator',
            'ROC_Rate_of_Change': 'scripts.strategies.roc_rate_of_change',
            'ATR_Volatility': 'scripts.strategies.atr_volatility',
            'Keltner_Channels_Breakout': 'scripts.strategies.keltner_channels_breakout',
            'TEMA_Crossover': 'scripts.strategies.tema_crossover',
            'RSI_Bullish_Divergence': 'scripts.strategies.rsi_bullish_divergence',
            'MACD_Zero_Line_Crossover': 'scripts.strategies.macd_zero_line_crossover',
            'Bollinger_Band_Squeeze': 'scripts.strategies.bollinger_band_squeeze',
            'Stochastic_K_D_Crossover': 'scripts.strategies.stochastic_k_d_crossover',
            'CCI_Crossover': 'scripts.strategies.cci_crossover',
            'Aroon_Oscillator': 'scripts.strategies.aroon_oscillator',
            'Ultimate_Oscillator_Buy': 'scripts.strategies.ultimate_oscillator_buy',
            'Money_Flow_Index_Oversold': 'scripts.strategies.money_flow_index_oversold',
            'Parabolic_SAR_Reversal': 'scripts.strategies.parabolic_sar_reversal',
            'Chaikin_Oscillator': 'scripts.strategies.chaikin_oscillator',
            'Accumulation_Distribution_Line': 'scripts.strategies.accumulation_distribution_line',
            'Triple_Moving_Average': 'scripts.strategies.triple_moving_average',
            'Vortex_Indicator': 'scripts.strategies.vortex_indicator',
            
            # Missing strategies - now implemented
            'Candlestick_Hammer': 'scripts.strategies.candlestick_hammer',
            'Candlestick_Bullish_Engulfing': 'scripts.strategies.candlestick_bullish_engulfing',
            'Candlestick_Doji': 'scripts.strategies.candlestick_doji',
            'Commodity_Channel_Index': 'scripts.strategies.commodity_channel_index',
            'DI_Crossover': 'scripts.strategies.di_crossover',
            'Elder_Ray_Index': 'scripts.strategies.elder_ray_index',
            'Ichimoku_Cloud_Breakout': 'scripts.strategies.ichimoku_cloud_breakout',
            'Ichimoku_Kijun_Tenkan_Crossover': 'scripts.strategies.ichimoku_kijun_tenkan_crossover',
            'Keltner_Channel_Squeeze': 'scripts.strategies.keltner_channel_squeeze',
            'Linear_Regression_Channel': 'scripts.strategies.linear_regression_channel',
            'OBV_Bullish_Divergence': 'scripts.strategies.obv_bullish_divergence',
            'Pivot_Points_Bounce': 'scripts.strategies.pivot_points_bounce',
            'Price_Volume_Trend': 'scripts.strategies.price_volume_trend',
        }
        
        for strategy_name, enabled in self.strategy_config.items():
            if enabled and strategy_name in strategy_mapping:
                try:
                    # Import the strategy module
                    module_path = strategy_mapping[strategy_name]
                    module = importlib.import_module(module_path)
                    
                    # Get the strategy class - handle different class naming conventions
                    if strategy_name == 'Volume_Breakout':
                        strategy_class = getattr(module, 'VolumeBreakoutStrategy')
                    elif strategy_name == 'Support_Resistance_Breakout':
                        strategy_class = getattr(module, 'SupportResistanceBreakoutStrategy')
                    elif strategy_name == 'Fibonacci_Retracement':
                        strategy_class = getattr(module, 'FibonacciRetracementStrategy')
                    elif strategy_name == 'Chart_Patterns':
                        strategy_class = getattr(module, 'ChartPatterns')
                    elif strategy_name == 'Volume_Profile':
                        strategy_class = getattr(module, 'VolumeProfile')
                    elif strategy_name == 'Multi_Timeframe_RSI':
                        strategy_class = getattr(module, 'MultiTimeframeRSI')
                    else:
                        strategy_class = getattr(module, strategy_name)
                    
                    # Initialize the strategy
                    self.strategy_instances[strategy_name] = strategy_class()
                    
                    logger.info(f"Loaded strategy: {strategy_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load strategy {strategy_name}: {e}")
                    import traceback
                    logger.error(f"Full traceback for {strategy_name}: {traceback.format_exc()}")
                    
    def evaluate_strategies(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate all loaded strategies against the given data.
        
        Args:
            symbol: Stock symbol
            data: Historical stock data
            
        Returns:
            Dictionary containing evaluation results
        """
        if data.empty:
            logger.warning(f"No data provided for {symbol}")
            return {
                'symbol': symbol,
                'technical_score': 0.0,
                'positive_signals': 0,
                'total_strategies': 0,
                'strategy_results': {},
                'recommendation': 'HOLD'
            }
        
        strategy_results = {}
        positive_signals = 0
        total_strategies = 0
        
        for strategy_name, strategy_instance in self.strategy_instances.items():
            try:
                # Run the strategy
                signal = strategy_instance.run_strategy(data.copy())
                
                strategy_results[strategy_name] = {
                    'signal': signal,
                    'signal_type': 'BUY' if signal == 1 else 'SELL/HOLD'
                }
                
                total_strategies += 1
                if signal == 1:
                    positive_signals += 1
                    
            except Exception as e:
                logger.error(f"Error running strategy {strategy_name} for {symbol}: {e}")
                strategy_results[strategy_name] = {
                    'signal': -1,
                    'signal_type': 'ERROR',
                    'error': str(e)
                }
                total_strategies += 1
        
        # Calculate technical score
        technical_score = positive_signals / total_strategies if total_strategies > 0 else 0.0
        
        # Determine recommendation
        if technical_score >= MIN_RECOMMENDATION_SCORE:
            recommendation = 'BUY'
        elif technical_score >= 0.5:
            recommendation = 'HOLD'
        else:
            recommendation = 'SELL'
        
        logger.info(f"Technical analysis for {symbol}: {positive_signals}/{total_strategies} positive signals, score: {technical_score:.2f}")
        
        return {
            'symbol': symbol,
            'technical_score': technical_score,
            'positive_signals': positive_signals,
            'total_strategies': total_strategies,
            'strategy_results': strategy_results,
            'recommendation': recommendation
        }
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Get a summary of loaded strategies.
        
        Returns:
            Dictionary with strategy summary information
        """
        return {
            'total_configured': len(self.strategy_config),
            'total_enabled': sum(1 for enabled in self.strategy_config.values() if enabled),
            'total_loaded': len(self.strategy_instances),
            'loaded_strategies': list(self.strategy_instances.keys()),
            'failed_strategies': [
                name for name, enabled in self.strategy_config.items() 
                if enabled and name not in self.strategy_instances
            ]
        }


def evaluate_single_strategy(strategy_name: str, data: pd.DataFrame, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Evaluate a single strategy against the given data.
    
    Args:
        strategy_name: Name of the strategy to evaluate
        data: Historical stock data
        params: Strategy parameters
        
    Returns:
        Dictionary containing strategy evaluation results
    """
    strategy_mapping = {
        'MA_Crossover_50_200': 'scripts.strategies.ma_crossover_50_200',
        'RSI_Overbought_Oversold': 'scripts.strategies.rsi_overbought_oversold',
        'MACD_Signal_Crossover': 'scripts.strategies.macd_signal_crossover',
        'Bollinger_Band_Breakout': 'scripts.strategies.bollinger_band_breakout',
    }
    
    if strategy_name not in strategy_mapping:
        return {
            'strategy_name': strategy_name,
            'signal': -1,
            'error': f"Strategy {strategy_name} not found"
        }
    
    try:
        # Import and initialize the strategy
        module_path = strategy_mapping[strategy_name]
        module = importlib.import_module(module_path)
        strategy_class = getattr(module, strategy_name)
        strategy_instance = strategy_class(params)
        
        # Run the strategy
        signal = strategy_instance.run_strategy(data)
        
        return {
            'strategy_name': strategy_name,
            'signal': signal,
            'signal_type': 'BUY' if signal == 1 else 'SELL/HOLD'
        }
        
    except Exception as e:
        logger.error(f"Error evaluating strategy {strategy_name}: {e}")
        return {
            'strategy_name': strategy_name,
            'signal': -1,
            'error': str(e)
        }
