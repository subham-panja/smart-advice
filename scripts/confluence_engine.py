"""
Multi-Timeframe Confluence Engine
File: scripts/confluence_engine.py

This module validates signals across multiple timeframes to generate higher-probability
trade recommendations by combining signals from different temporal views of the market.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.logger import setup_logging
from scripts.data_fetcher import get_historical_data
from scripts.analyzer import StockAnalyzer
import talib as ta

logger = setup_logging()

class ConfluenceEngine:
    """
    Multi-Timeframe Confluence Engine that validates signals across different timeframes
    to generate higher-probability trading recommendations.
    """
    
    def __init__(self, timeframes: List[str] = ['1d', '4h', '1h']):
        """
        Initialize the Confluence Engine.
        
        Args:
            timeframes: List of timeframes to analyze ['1d', '4h', '1h']
        """
        self.timeframes = timeframes
        self.analyzer = StockAnalyzer()
        
        # Confluence rules configuration
        self.confluence_rules = {
            'strong_multi_timeframe_buy': {
                'daily': ['MA_Crossover_50_200_bullish', 'RSI_oversold_bounce'],
                '4h': ['RSI_oversold_bounce', 'MACD_bullish'],
                '1h': ['bullish_engulfing', 'volume_breakout'],
                'logic': 'AND'  # All conditions must be met
            },
            'multi_timeframe_buy': {
                'daily': ['trend_bullish', 'not_overbought'],
                '4h': ['RSI_oversold_bounce', 'support_bounce'],
                '1h': ['bullish_pattern'],
                'logic': 'OR'  # Any two timeframes showing bullish
            },
            'confluence_sell': {
                'daily': ['trend_bearish', 'resistance_rejection'],
                '4h': ['RSI_overbought', 'MACD_bearish'],
                '1h': ['bearish_engulfing', 'volume_breakdown'],
                'logic': 'AND'
            }
        }
    
    def analyze_multi_timeframe(self, symbol: str, period: str = '6mo') -> Dict[str, Any]:
        """
        Analyze a stock across multiple timeframes to generate confluence signals.
        
        Args:
            symbol: Stock symbol to analyze
            period: Period of historical data to fetch
            
        Returns:
            Dictionary containing multi-timeframe analysis results
        """
        logger.info(f"Starting multi-timeframe confluence analysis for {symbol}")
        
        try:
            # Initialize results structure
            results = {
                'symbol': symbol,
                'timeframe_analysis': {},
                'confluence_signals': {},
                'final_recommendation': 'HOLD',
                'confidence_score': 0.0,
                'supporting_timeframes': []
            }
            
            # Analyze each timeframe
            for timeframe in self.timeframes:
                logger.info(f"Analyzing {symbol} on {timeframe} timeframe")
                
                try:
                    # Fetch data for this timeframe
                    data = get_historical_data(symbol, period, timeframe)
                    
                    if data.empty:
                        logger.warning(f"No data available for {symbol} on {timeframe}")
                        continue
                    
                    # Perform timeframe-specific analysis
                    timeframe_result = self._analyze_single_timeframe(data, timeframe)
                    results['timeframe_analysis'][timeframe] = timeframe_result
                    
                    logger.debug(f"Completed {timeframe} analysis for {symbol}: {timeframe_result['signals']}")
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} on {timeframe}: {e}")
                    continue
            
            # Generate confluence signals
            confluence_signals = self._generate_confluence_signals(results['timeframe_analysis'])
            results['confluence_signals'] = confluence_signals
            
            # Determine final recommendation
            final_rec = self._determine_final_recommendation(confluence_signals)
            results['final_recommendation'] = final_rec['recommendation']
            results['confidence_score'] = final_rec['confidence']
            results['supporting_timeframes'] = final_rec['supporting_timeframes']
            
            logger.info(f"Confluence analysis complete for {symbol}: {results['final_recommendation']} "
                       f"(confidence: {results['confidence_score']:.2f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in multi-timeframe analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'final_recommendation': 'HOLD',
                'confidence_score': 0.0
            }
    
    def _analyze_single_timeframe(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Analyze a single timeframe to extract relevant signals.
        
        Args:
            data: OHLCV data for the timeframe
            timeframe: The timeframe being analyzed
            
        Returns:
            Dictionary containing timeframe analysis results
        """
        try:
            signals = {}
            
            # Calculate technical indicators
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            volume = data['Volume'].values
            
            # Moving Averages
            ma_50 = ta.SMA(close, timeperiod=50)
            ma_200 = ta.SMA(close, timeperiod=200)
            
            # RSI
            rsi = ta.RSI(close, timeperiod=14)
            
            # MACD
            macd_line, macd_signal, macd_hist = ta.MACD(close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close)
            
            # Volume analysis
            volume_ma = ta.SMA(volume.astype(float), timeperiod=20)
            
            # Generate signals based on current conditions
            current_idx = -1  # Latest data point
            
            # Trend Analysis
            if len(ma_50) > 0 and len(ma_200) > 0:
                if not pd.isna(ma_50[current_idx]) and not pd.isna(ma_200[current_idx]):
                    if ma_50[current_idx] > ma_200[current_idx]:
                        signals['trend_bullish'] = True
                        signals['trend_bearish'] = False
                    else:
                        signals['trend_bullish'] = False
                        signals['trend_bearish'] = True
                        
                    # Golden/Death Cross detection
                    if (len(ma_50) > 1 and len(ma_200) > 1 and
                        not pd.isna(ma_50[-2]) and not pd.isna(ma_200[-2])):
                        if ma_50[-2] <= ma_200[-2] and ma_50[current_idx] > ma_200[current_idx]:
                            signals['MA_Crossover_50_200_bullish'] = True
                        elif ma_50[-2] >= ma_200[-2] and ma_50[current_idx] < ma_200[current_idx]:
                            signals['MA_Crossover_50_200_bearish'] = True
            
            # RSI Analysis
            if len(rsi) > 0 and not pd.isna(rsi[current_idx]):
                current_rsi = rsi[current_idx]
                signals['RSI_value'] = current_rsi
                
                if current_rsi < 30:
                    signals['RSI_oversold'] = True
                elif current_rsi > 70:
                    signals['RSI_overbought'] = True
                else:
                    signals['not_overbought'] = True
                    signals['not_oversold'] = True
                
                # RSI bounce detection
                if len(rsi) > 1 and not pd.isna(rsi[-2]):
                    if rsi[-2] < 30 and current_rsi > 35:
                        signals['RSI_oversold_bounce'] = True
                    elif rsi[-2] > 70 and current_rsi < 65:
                        signals['RSI_overbought_decline'] = True
            
            # MACD Analysis
            if (len(macd_line) > 0 and len(macd_signal) > 0 and 
                not pd.isna(macd_line[current_idx]) and not pd.isna(macd_signal[current_idx])):
                
                if macd_line[current_idx] > macd_signal[current_idx]:
                    signals['MACD_bullish'] = True
                else:
                    signals['MACD_bearish'] = True
                
                # MACD crossover detection
                if (len(macd_line) > 1 and len(macd_signal) > 1 and
                    not pd.isna(macd_line[-2]) and not pd.isna(macd_signal[-2])):
                    if (macd_line[-2] <= macd_signal[-2] and 
                        macd_line[current_idx] > macd_signal[current_idx]):
                        signals['MACD_bullish_crossover'] = True
                    elif (macd_line[-2] >= macd_signal[-2] and 
                          macd_line[current_idx] < macd_signal[current_idx]):
                        signals['MACD_bearish_crossover'] = True
            
            # Support/Resistance Analysis
            if len(close) >= 20:
                recent_high = np.max(high[-20:])
                recent_low = np.min(low[-20:])
                current_price = close[current_idx]
                
                # Support bounce
                if current_price <= recent_low * 1.02:  # Within 2% of recent low
                    signals['support_bounce'] = True
                
                # Resistance rejection
                if current_price >= recent_high * 0.98:  # Within 2% of recent high
                    signals['resistance_rejection'] = True
            
            # Volume Analysis
            if len(volume_ma) > 0 and not pd.isna(volume_ma[current_idx]):
                current_volume = volume[current_idx]
                avg_volume = volume_ma[current_idx]
                
                if current_volume > avg_volume * 1.5:  # 50% above average
                    signals['volume_breakout'] = True
                elif current_volume < avg_volume * 0.5:  # 50% below average
                    signals['volume_breakdown'] = True
            
            # Candlestick Pattern Analysis (simplified)
            if len(data) >= 2:
                prev_candle = data.iloc[-2]
                curr_candle = data.iloc[-1]
                
                # Bullish Engulfing
                if (prev_candle['Close'] < prev_candle['Open'] and  # Previous red candle
                    curr_candle['Close'] > curr_candle['Open'] and  # Current green candle
                    curr_candle['Open'] < prev_candle['Close'] and  # Gap down opening
                    curr_candle['Close'] > prev_candle['Open']):    # Engulfs previous candle
                    signals['bullish_engulfing'] = True
                
                # Bearish Engulfing
                if (prev_candle['Close'] > prev_candle['Open'] and  # Previous green candle
                    curr_candle['Close'] < curr_candle['Open'] and  # Current red candle
                    curr_candle['Open'] > prev_candle['Close'] and  # Gap up opening
                    curr_candle['Close'] < prev_candle['Open']):    # Engulfs previous candle
                    signals['bearish_engulfing'] = True
                
                # General bullish/bearish pattern
                if curr_candle['Close'] > curr_candle['Open']:
                    signals['bullish_pattern'] = True
                else:
                    signals['bearish_pattern'] = True
            
            return {
                'timeframe': timeframe,
                'signals': signals,
                'data_points': len(data),
                'latest_price': close[current_idx] if len(close) > 0 else 0,
                'analysis_timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            logger.error(f"Error in single timeframe analysis ({timeframe}): {e}")
            return {
                'timeframe': timeframe,
                'error': str(e),
                'signals': {}
            }
    
    def _generate_confluence_signals(self, timeframe_analysis: Dict) -> Dict[str, Any]:
        """
        Generate confluence signals by combining analysis from different timeframes.
        
        Args:
            timeframe_analysis: Dictionary containing analysis for each timeframe
            
        Returns:
            Dictionary containing confluence signals
        """
        confluence_signals = {}
        
        try:
            # Extract signals from each timeframe
            daily_signals = timeframe_analysis.get('1d', {}).get('signals', {})
            four_hour_signals = timeframe_analysis.get('4h', {}).get('signals', {})
            hourly_signals = timeframe_analysis.get('1h', {}).get('signals', {})
            
            # Strong Multi-Timeframe Buy Signal
            strong_buy_conditions = []
            
            # Daily: Bullish trend and not overbought
            if daily_signals.get('MA_Crossover_50_200_bullish') or daily_signals.get('trend_bullish'):
                strong_buy_conditions.append('daily_bullish')
            
            # 4-hour: RSI oversold bounce or MACD bullish
            if (four_hour_signals.get('RSI_oversold_bounce') or 
                four_hour_signals.get('MACD_bullish_crossover')):
                strong_buy_conditions.append('4h_momentum')
            
            # 1-hour: Bullish pattern with volume
            if (hourly_signals.get('bullish_engulfing') or 
                (hourly_signals.get('bullish_pattern') and hourly_signals.get('volume_breakout'))):
                strong_buy_conditions.append('1h_entry')
            
            confluence_signals['strong_multi_timeframe_buy'] = len(strong_buy_conditions) >= 3
            confluence_signals['strong_buy_components'] = strong_buy_conditions
            
            # Multi-Timeframe Buy Signal (less strict)
            buy_conditions = []
            
            # Daily: Any bullish indication
            if (daily_signals.get('trend_bullish') or 
                daily_signals.get('not_overbought') or
                daily_signals.get('MA_Crossover_50_200_bullish')):
                buy_conditions.append('daily_supportive')
            
            # 4-hour: Mean reversion or momentum
            if (four_hour_signals.get('RSI_oversold_bounce') or 
                four_hour_signals.get('support_bounce') or
                four_hour_signals.get('MACD_bullish')):
                buy_conditions.append('4h_supportive')
            
            # 1-hour: Any bullish pattern
            if (hourly_signals.get('bullish_pattern') or 
                hourly_signals.get('bullish_engulfing') or
                hourly_signals.get('volume_breakout')):
                buy_conditions.append('1h_supportive')
            
            confluence_signals['multi_timeframe_buy'] = len(buy_conditions) >= 2
            confluence_signals['buy_components'] = buy_conditions
            
            # Confluence Sell Signal
            sell_conditions = []
            
            # Daily: Bearish trend or resistance
            if (daily_signals.get('trend_bearish') or 
                daily_signals.get('resistance_rejection')):
                sell_conditions.append('daily_bearish')
            
            # 4-hour: Overbought or bearish momentum
            if (four_hour_signals.get('RSI_overbought') or 
                four_hour_signals.get('MACD_bearish_crossover')):
                sell_conditions.append('4h_bearish')
            
            # 1-hour: Bearish pattern with volume
            if (hourly_signals.get('bearish_engulfing') or 
                (hourly_signals.get('bearish_pattern') and hourly_signals.get('volume_breakdown'))):
                sell_conditions.append('1h_bearish')
            
            confluence_signals['confluence_sell'] = len(sell_conditions) >= 2
            confluence_signals['sell_components'] = sell_conditions
            
            # Calculate overall confluence strength
            total_bullish_signals = len(strong_buy_conditions) + len(buy_conditions)
            total_bearish_signals = len(sell_conditions)
            
            confluence_signals['confluence_strength'] = {
                'bullish_count': total_bullish_signals,
                'bearish_count': total_bearish_signals,
                'net_signal': total_bullish_signals - total_bearish_signals
            }
            
            return confluence_signals
            
        except Exception as e:
            logger.error(f"Error generating confluence signals: {e}")
            return {'error': str(e)}
    
    def _determine_final_recommendation(self, confluence_signals: Dict) -> Dict[str, Any]:
        """
        Determine the final recommendation based on confluence signals.
        
        Args:
            confluence_signals: Dictionary containing confluence signals
            
        Returns:
            Dictionary containing final recommendation and confidence
        """
        try:
            if 'error' in confluence_signals:
                return {
                    'recommendation': 'HOLD',
                    'confidence': 0.0,
                    'supporting_timeframes': [],
                    'reason': 'Error in confluence analysis'
                }
            
            # Strong Multi-Timeframe Buy (highest priority)
            if confluence_signals.get('strong_multi_timeframe_buy', False):
                return {
                    'recommendation': 'STRONG_BUY',
                    'confidence': 0.9,
                    'supporting_timeframes': confluence_signals.get('strong_buy_components', []),
                    'reason': 'Strong bullish confluence across all timeframes'
                }
            
            # Multi-Timeframe Buy
            elif confluence_signals.get('multi_timeframe_buy', False):
                buy_strength = len(confluence_signals.get('buy_components', []))
                confidence = min(0.8, 0.4 + (buy_strength * 0.2))
                
                return {
                    'recommendation': 'BUY',
                    'confidence': confidence,
                    'supporting_timeframes': confluence_signals.get('buy_components', []),
                    'reason': f'Bullish confluence across {buy_strength} timeframes'
                }
            
            # Confluence Sell
            elif confluence_signals.get('confluence_sell', False):
                sell_strength = len(confluence_signals.get('sell_components', []))
                confidence = min(0.8, 0.4 + (sell_strength * 0.2))
                
                return {
                    'recommendation': 'SELL',
                    'confidence': confidence,
                    'supporting_timeframes': confluence_signals.get('sell_components', []),
                    'reason': f'Bearish confluence across {sell_strength} timeframes'
                }
            
            # Neutral/Hold
            else:
                net_signal = confluence_signals.get('confluence_strength', {}).get('net_signal', 0)
                
                if net_signal > 0:
                    reason = 'Mild bullish bias but insufficient confluence'
                elif net_signal < 0:
                    reason = 'Mild bearish bias but insufficient confluence'
                else:
                    reason = 'No clear directional bias across timeframes'
                
                return {
                    'recommendation': 'HOLD',
                    'confidence': 0.3,
                    'supporting_timeframes': [],
                    'reason': reason
                }
                
        except Exception as e:
            logger.error(f"Error determining final recommendation: {e}")
            return {
                'recommendation': 'HOLD',
                'confidence': 0.0,
                'supporting_timeframes': [],
                'reason': f'Error in recommendation logic: {str(e)}'
            }
    
    def get_confluence_summary(self, analysis_results: Dict) -> str:
        """
        Generate a human-readable summary of the confluence analysis.
        
        Args:
            analysis_results: Results from analyze_multi_timeframe
            
        Returns:
            String containing analysis summary
        """
        try:
            symbol = analysis_results.get('symbol', 'Unknown')
            recommendation = analysis_results.get('final_recommendation', 'HOLD')
            confidence = analysis_results.get('confidence_score', 0.0)
            supporting_timeframes = analysis_results.get('supporting_timeframes', [])
            
            summary = f"Multi-Timeframe Analysis for {symbol}:\n"
            summary += f"Final Recommendation: {recommendation} (Confidence: {confidence:.1%})\n"
            
            if supporting_timeframes:
                summary += f"Supporting Evidence: {', '.join(supporting_timeframes)}\n"
            
            # Add timeframe-specific insights
            timeframe_analysis = analysis_results.get('timeframe_analysis', {})
            for tf, analysis in timeframe_analysis.items():
                if 'error' not in analysis:
                    signals = analysis.get('signals', {})
                    key_signals = []
                    
                    # Extract key signals for summary
                    if signals.get('trend_bullish'):
                        key_signals.append('Bullish Trend')
                    if signals.get('RSI_oversold_bounce'):
                        key_signals.append('RSI Bounce')
                    if signals.get('MACD_bullish_crossover'):
                        key_signals.append('MACD Bullish Cross')
                    if signals.get('bullish_engulfing'):
                        key_signals.append('Bullish Engulfing')
                    if signals.get('volume_breakout'):
                        key_signals.append('Volume Breakout')
                    
                    if key_signals:
                        summary += f"{tf.upper()}: {', '.join(key_signals)}\n"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating confluence summary: {e}")
            return f"Error generating summary for {analysis_results.get('symbol', 'Unknown')}: {str(e)}"
