"""
Swing Trading Signal Analyzer
==============================

This module implements precision-first swing trading signals with:
1. Trend filters (ADX, 200 SMA)
2. Multi-timeframe confirmation
3. Volatility gates (ATR-based)
4. Volume confirmation (OBV, Z-score)
5. Entry patterns (pullbacks, breakouts, etc.)
6. Exit rules (stop-loss, take-profit, time-stops)
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, List, Tuple, Optional
from utils.logger import setup_logging
from datetime import datetime, timedelta

logger = setup_logging()


class SwingTradingSignalAnalyzer:
    """
    Analyzes stocks for swing trading opportunities with strict gates and filters
    """
    
    def __init__(self):
        """Initialize the swing trading signal analyzer"""
        
        # Signal thresholds
        self.thresholds = {
            'adx_min': 20,              # Minimum ADX for trend strength
            'adx_max': 50,              # Maximum ADX (too high = exhausted trend)
            'atr_percentile_min': 20,   # Minimum ATR percentile (avoid dead stocks)
            'atr_percentile_max': 80,   # Maximum ATR percentile (avoid extreme volatility)
            'volume_zscore_min': 1.0,   # Minimum volume Z-score for breakout
            'rsi_pullback_min': 40,     # Minimum RSI for pullback entry
            'rsi_pullback_max': 60,     # Maximum RSI for pullback entry
            'bb_squeeze_threshold': 0.05, # Bollinger Band squeeze threshold
            'macd_zero_buffer': 0.1     # Buffer above zero line for MACD crosses
        }
        
        # Risk management parameters
        self.risk_params = {
            'atr_multiplier_sl': 1.5,   # ATR multiplier for stop-loss
            'atr_multiplier_tp1': 1.0,  # ATR multiplier for first take-profit
            'atr_multiplier_tp2': 2.5,  # ATR multiplier for second take-profit
            'atr_multiplier_trail': 3.0, # ATR multiplier for trailing stop
            'time_stop_bars': 15,        # Exit if no progress after N bars
            'max_risk_per_trade': 0.01   # Maximum 1% risk per trade
        }
    
    def calculate_trend_filter(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate trend filter signals
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with trend filter results
        """
        try:
            # Calculate ADX for trend strength
            adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
            plus_di = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
            minus_di = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            # Calculate 200 SMA for long-term trend
            sma_200 = talib.SMA(df['Close'], timeperiod=200)
            
            # Calculate 50 SMA for medium-term trend
            sma_50 = talib.SMA(df['Close'], timeperiod=50)
            
            # Calculate 20 EMA for short-term trend
            ema_20 = talib.EMA(df['Close'], timeperiod=20)
            
            # Current values
            current_adx = adx.iloc[-1]
            current_price = df['Close'].iloc[-1]
            current_sma_200 = sma_200.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            current_ema_20 = ema_20.iloc[-1]
            
            # Trend conditions
            strong_trend = (
                self.thresholds['adx_min'] <= current_adx <= self.thresholds['adx_max']
            )
            
            bullish_trend = (
                current_price > current_sma_200 and
                current_sma_50 > current_sma_200 and
                current_ema_20 > current_sma_50 and
                plus_di.iloc[-1] > minus_di.iloc[-1]
            )
            
            # Trend slope (momentum)
            sma_200_slope = (sma_200.iloc[-1] - sma_200.iloc[-5]) / sma_200.iloc[-5] * 100 if len(sma_200) > 5 else 0
            
            return {
                'passed': strong_trend and bullish_trend,
                'adx': current_adx,
                'price_vs_sma200': (current_price - current_sma_200) / current_sma_200 * 100,
                'sma50_vs_sma200': (current_sma_50 - current_sma_200) / current_sma_200 * 100,
                'sma200_slope': sma_200_slope,
                'bullish_trend': bullish_trend,
                'strong_trend': strong_trend,
                'reason': f"ADX={current_adx:.1f}, Price {'above' if bullish_trend else 'below'} 200SMA"
            }
            
        except Exception as e:
            logger.error(f"Error calculating trend filter: {e}")
            return {'passed': False, 'reason': f"Error: {str(e)}"}
    
    def calculate_multi_timeframe_confirmation(self, daily_df: pd.DataFrame, 
                                              weekly_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Check multi-timeframe alignment
        
        Args:
            daily_df: Daily timeframe data
            weekly_df: Weekly timeframe data (optional)
            
        Returns:
            Dictionary with MTF confirmation results
        """
        try:
            # If no weekly data provided, resample daily to weekly
            if weekly_df is None or weekly_df.empty:
                weekly_df = daily_df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
            
            if len(weekly_df) < 50:
                return {
                    'passed': False,
                    'reason': 'Insufficient weekly data for MTF analysis'
                }
            
            # Calculate weekly indicators
            weekly_sma_20 = talib.SMA(weekly_df['Close'], timeperiod=20)
            weekly_sma_50 = talib.SMA(weekly_df['Close'], timeperiod=50)
            weekly_rsi = talib.RSI(weekly_df['Close'], timeperiod=14)
            
            # Calculate daily indicators
            daily_sma_20 = talib.SMA(daily_df['Close'], timeperiod=20)
            daily_rsi = talib.RSI(daily_df['Close'], timeperiod=14)
            
            # Check alignment
            weekly_trend_up = (
                weekly_sma_20.iloc[-1] > weekly_sma_50.iloc[-1] and
                weekly_df['Close'].iloc[-1] > weekly_sma_20.iloc[-1] and
                weekly_rsi.iloc[-1] > 50
            )
            
            daily_trend_up = (
                daily_sma_20.iloc[-1] > daily_sma_20.iloc[-5] and
                daily_df['Close'].iloc[-1] > daily_sma_20.iloc[-1] and
                daily_rsi.iloc[-1] > 40
            )
            
            mtf_aligned = weekly_trend_up and daily_trend_up
            
            return {
                'passed': mtf_aligned,
                'weekly_trend_up': weekly_trend_up,
                'daily_trend_up': daily_trend_up,
                'weekly_rsi': weekly_rsi.iloc[-1] if not weekly_rsi.empty else None,
                'daily_rsi': daily_rsi.iloc[-1] if not daily_rsi.empty else None,
                'reason': f"Weekly {'UP' if weekly_trend_up else 'DOWN'}, Daily {'UP' if daily_trend_up else 'DOWN'}"
            }
            
        except Exception as e:
            logger.error(f"Error in MTF confirmation: {e}")
            return {'passed': False, 'reason': f"Error: {str(e)}"}
    
    def calculate_volatility_gate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volatility gate using ATR
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volatility gate results
        """
        try:
            # Calculate ATR
            atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            
            # Calculate ATR as percentage of price
            atr_pct = (atr / df['Close']) * 100
            
            # Get current ATR percentile over last 100 days
            current_atr = atr.iloc[-1]
            atr_percentile = (atr.iloc[-100:] < current_atr).sum() / min(100, len(atr)) * 100
            
            # Calculate historical volatility
            returns = df['Close'].pct_change()
            volatility = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized
            
            # Check if volatility is in acceptable range
            volatility_ok = (
                self.thresholds['atr_percentile_min'] <= atr_percentile <= self.thresholds['atr_percentile_max']
            )
            
            # Check for volatility expansion/contraction
            recent_atr_mean = atr.iloc[-5:].mean()
            older_atr_mean = atr.iloc[-20:-5].mean()
            volatility_expanding = recent_atr_mean > older_atr_mean * 1.1
            
            return {
                'passed': volatility_ok,
                'atr': current_atr,
                'atr_pct': atr_pct.iloc[-1],
                'atr_percentile': atr_percentile,
                'volatility': volatility,
                'volatility_expanding': volatility_expanding,
                'reason': f"ATR percentile={atr_percentile:.0f}% ({'OK' if volatility_ok else 'OUT OF RANGE'})"
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility gate: {e}")
            return {'passed': False, 'reason': f"Error: {str(e)}"}
    
    def calculate_volume_confirmation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volume confirmation signals
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with volume confirmation results
        """
        try:
            # Calculate OBV
            obv = talib.OBV(df['Close'], df['Volume'])
            
            # Calculate OBV trend (using linear regression slope)
            obv_window = 20
            if len(obv) >= obv_window:
                x = np.arange(obv_window)
                y = obv.iloc[-obv_window:].values
                slope = np.polyfit(x, y, 1)[0]
                obv_trending_up = slope > 0
            else:
                obv_trending_up = False
            
            # Calculate volume Z-score
            volume_mean = df['Volume'].iloc[-20:].mean()
            volume_std = df['Volume'].iloc[-20:].std()
            
            if volume_std > 0:
                current_volume_zscore = (df['Volume'].iloc[-1] - volume_mean) / volume_std
            else:
                current_volume_zscore = 0
            
            # Calculate volume moving average
            volume_ma = df['Volume'].rolling(window=20).mean()
            volume_above_ma = df['Volume'].iloc[-1] > volume_ma.iloc[-1]
            
            # Check for volume spike on price breakout
            price_breakout = df['Close'].iloc[-1] > df['High'].iloc[-2:-6].max()
            volume_spike = current_volume_zscore > self.thresholds['volume_zscore_min']
            
            # Volume confirmation conditions
            volume_confirmed = (
                obv_trending_up or 
                (volume_spike and price_breakout) or
                (volume_above_ma and current_volume_zscore > 0.5)
            )
            
            return {
                'passed': volume_confirmed,
                'obv_trending_up': obv_trending_up,
                'volume_zscore': current_volume_zscore,
                'volume_above_ma': volume_above_ma,
                'price_breakout': price_breakout,
                'volume_spike': volume_spike,
                'reason': f"OBV {'UP' if obv_trending_up else 'DOWN'}, Volume Z-score={current_volume_zscore:.1f}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume confirmation: {e}")
            return {'passed': False, 'reason': f"Error: {str(e)}"}
    
    def detect_entry_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect various entry patterns for swing trading
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with detected patterns and signals
        """
        patterns = {}
        
        try:
            # 1. Pullback to rising 20 EMA
            ema_20 = talib.EMA(df['Close'], timeperiod=20)
            rsi = talib.RSI(df['Close'], timeperiod=14)
            
            pullback_to_ema = (
                df['Low'].iloc[-1] <= ema_20.iloc[-1] * 1.02 and  # Near EMA
                df['Close'].iloc[-1] > ema_20.iloc[-1] and        # Closed above EMA
                ema_20.iloc[-1] > ema_20.iloc[-5] and            # EMA rising
                self.thresholds['rsi_pullback_min'] <= rsi.iloc[-1] <= self.thresholds['rsi_pullback_max'] and
                df['Close'].iloc[-1] > df['Open'].iloc[-1]       # Bullish candle
            )
            
            patterns['pullback_to_ema'] = {
                'detected': pullback_to_ema,
                'strength': 0.8 if pullback_to_ema else 0,
                'description': 'Pullback to rising 20 EMA with bullish reversal'
            }
            
            # 2. Bollinger Band squeeze breakout
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
            bb_width = (bb_upper - bb_lower) / bb_middle
            bb_squeeze = bb_width.iloc[-1] < bb_width.iloc[-20:].quantile(0.25)
            bb_breakout = df['Close'].iloc[-1] > bb_upper.iloc[-1] and bb_squeeze
            
            patterns['bb_squeeze_breakout'] = {
                'detected': bb_breakout,
                'strength': 0.7 if bb_breakout else 0,
                'description': 'Bollinger Band squeeze with upward breakout'
            }
            
            # 3. MACD signal cross above zero
            macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            macd_bullish_cross = (
                macd.iloc[-1] > macd_signal.iloc[-1] and
                macd.iloc[-2] <= macd_signal.iloc[-2] and
                macd.iloc[-1] > self.thresholds['macd_zero_buffer']
            )
            
            patterns['macd_bullish_cross'] = {
                'detected': macd_bullish_cross,
                'strength': 0.75 if macd_bullish_cross else 0,
                'description': 'MACD signal cross above zero line'
            }
            
            # 4. Higher-low swing structure
            recent_lows = df['Low'].iloc[-20:]
            swing_lows = []
            for i in range(1, len(recent_lows) - 1):
                if recent_lows.iloc[i] < recent_lows.iloc[i-1] and recent_lows.iloc[i] < recent_lows.iloc[i+1]:
                    swing_lows.append((i, recent_lows.iloc[i]))
            
            higher_low_structure = False
            if len(swing_lows) >= 2:
                higher_low_structure = swing_lows[-1][1] > swing_lows[-2][1]
            
            patterns['higher_low_structure'] = {
                'detected': higher_low_structure,
                'strength': 0.85 if higher_low_structure else 0,
                'description': 'Higher-low swing structure formed'
            }
            
            # 5. Volume-supported breakout
            resistance = df['High'].iloc[-20:-1].max()
            breakout_with_volume = (
                df['Close'].iloc[-1] > resistance and
                df['Volume'].iloc[-1] > df['Volume'].iloc[-20:].mean() * 1.5
            )
            
            patterns['volume_breakout'] = {
                'detected': breakout_with_volume,
                'strength': 0.9 if breakout_with_volume else 0,
                'description': 'Resistance breakout with volume confirmation'
            }
            
            # Calculate overall pattern strength
            detected_patterns = [p for p in patterns.values() if p['detected']]
            overall_strength = sum(p['strength'] for p in detected_patterns) / len(patterns) if detected_patterns else 0
            
            return {
                'patterns': patterns,
                'detected_count': len(detected_patterns),
                'overall_strength': overall_strength,
                'strongest_pattern': max(patterns.items(), key=lambda x: x[1]['strength'])[0] if detected_patterns else None
            }
            
        except Exception as e:
            logger.error(f"Error detecting entry patterns: {e}")
            return {
                'patterns': {},
                'detected_count': 0,
                'overall_strength': 0,
                'strongest_pattern': None
            }
    
    def calculate_exit_rules(self, df: pd.DataFrame, entry_price: float = None) -> Dict[str, Any]:
        """
        Calculate exit rules (stop-loss, take-profit, time-stop)
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price (uses current price if not provided)
            
        Returns:
            Dictionary with exit levels and rules
        """
        try:
            if entry_price is None:
                entry_price = df['Close'].iloc[-1]
            
            # Calculate ATR for position sizing
            atr = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
            current_atr = atr.iloc[-1]
            
            # Find recent swing low for initial stop
            recent_lows = df['Low'].iloc[-20:]
            swing_low = recent_lows.min()
            
            # Calculate stop-loss levels
            atr_stop = entry_price - (current_atr * self.risk_params['atr_multiplier_sl'])
            swing_stop = swing_low * 0.98  # 2% below swing low
            
            # Use the higher stop (tighter risk)
            stop_loss = max(atr_stop, swing_stop)
            
            # Calculate take-profit levels
            take_profit_1 = entry_price + (current_atr * self.risk_params['atr_multiplier_tp1'])
            take_profit_2 = entry_price + (current_atr * self.risk_params['atr_multiplier_tp2'])
            
            # Calculate trailing stop level (chandelier exit)
            trailing_stop_distance = current_atr * self.risk_params['atr_multiplier_trail']
            
            # Calculate risk-reward ratios
            risk = entry_price - stop_loss
            reward_1 = take_profit_1 - entry_price
            reward_2 = take_profit_2 - entry_price
            
            risk_reward_1 = reward_1 / risk if risk > 0 else 0
            risk_reward_2 = reward_2 / risk if risk > 0 else 0
            
            # Position sizing based on risk
            risk_amount = 10000 * self.risk_params['max_risk_per_trade']  # Assuming $10,000 account
            position_size = risk_amount / risk if risk > 0 else 0
            
            return {
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit_1': take_profit_1,
                'take_profit_2': take_profit_2,
                'trailing_stop_distance': trailing_stop_distance,
                'risk_per_share': risk,
                'risk_reward_1': risk_reward_1,
                'risk_reward_2': risk_reward_2,
                'position_size': position_size,
                'time_stop_bars': self.risk_params['time_stop_bars'],
                'atr': current_atr,
                'exit_strategy': f"SL: {stop_loss:.2f}, TP1: {take_profit_1:.2f} (1:{risk_reward_1:.1f}), TP2: {take_profit_2:.2f} (1:{risk_reward_2:.1f})"
            }
            
        except Exception as e:
            logger.error(f"Error calculating exit rules: {e}")
            return {
                'entry_price': entry_price or 0,
                'stop_loss': 0,
                'take_profit_1': 0,
                'take_profit_2': 0,
                'error': str(e)
            }
    
    def analyze_swing_opportunity(self, symbol: str, daily_df: pd.DataFrame, 
                                 weekly_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Comprehensive swing trading analysis with all gates and filters
        
        Args:
            symbol: Stock symbol
            daily_df: Daily timeframe data
            weekly_df: Weekly timeframe data (optional)
            
        Returns:
            Dictionary with complete swing trading analysis
        """
        logger.info(f"Analyzing swing trading opportunity for {symbol}")
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'gates_passed': {},
            'all_gates_passed': False,
            'signal_strength': 0,
            'entry_patterns': {},
            'exit_rules': {},
            'recommendation': 'HOLD',
            'confidence': 0,
            'reasons': []
        }
        
        try:
            # Check if we have enough data
            if len(daily_df) < 200:
                analysis['recommendation'] = 'INSUFFICIENT_DATA'
                analysis['reasons'].append('Need at least 200 days of data for analysis')
                return analysis
            
            # 1. Trend Filter
            trend_filter = self.calculate_trend_filter(daily_df)
            analysis['gates_passed']['trend_filter'] = trend_filter['passed']
            if trend_filter['passed']:
                analysis['reasons'].append(f"✓ Trend Filter: {trend_filter['reason']}")
            else:
                analysis['reasons'].append(f"✗ Trend Filter: {trend_filter['reason']}")
            
            # 2. Multi-Timeframe Confirmation
            mtf_confirmation = self.calculate_multi_timeframe_confirmation(daily_df, weekly_df)
            analysis['gates_passed']['mtf_confirmation'] = mtf_confirmation['passed']
            if mtf_confirmation['passed']:
                analysis['reasons'].append(f"✓ MTF Confirmation: {mtf_confirmation['reason']}")
            else:
                analysis['reasons'].append(f"✗ MTF Confirmation: {mtf_confirmation['reason']}")
            
            # 3. Volatility Gate
            volatility_gate = self.calculate_volatility_gate(daily_df)
            analysis['gates_passed']['volatility_gate'] = volatility_gate['passed']
            if volatility_gate['passed']:
                analysis['reasons'].append(f"✓ Volatility Gate: {volatility_gate['reason']}")
            else:
                analysis['reasons'].append(f"✗ Volatility Gate: {volatility_gate['reason']}")
            
            # 4. Volume Confirmation
            volume_confirmation = self.calculate_volume_confirmation(daily_df)
            analysis['gates_passed']['volume_confirmation'] = volume_confirmation['passed']
            if volume_confirmation['passed']:
                analysis['reasons'].append(f"✓ Volume Confirmation: {volume_confirmation['reason']}")
            else:
                analysis['reasons'].append(f"✗ Volume Confirmation: {volume_confirmation['reason']}")
            
            # Check if all gates passed
            all_gates_passed = all(analysis['gates_passed'].values())
            analysis['all_gates_passed'] = all_gates_passed
            
            # If all gates passed, look for entry patterns
            if all_gates_passed:
                # 5. Detect Entry Patterns
                entry_patterns = self.detect_entry_patterns(daily_df)
                analysis['entry_patterns'] = entry_patterns
                
                if entry_patterns['detected_count'] > 0:
                    analysis['reasons'].append(
                        f"✓ Entry Patterns: {entry_patterns['detected_count']} patterns detected, "
                        f"strongest: {entry_patterns['strongest_pattern']}"
                    )
                    
                    # 6. Calculate Exit Rules
                    exit_rules = self.calculate_exit_rules(daily_df)
                    analysis['exit_rules'] = exit_rules
                    analysis['reasons'].append(f"✓ Exit Strategy: {exit_rules['exit_strategy']}")
                    
                    # Calculate overall signal strength
                    gate_score = sum(analysis['gates_passed'].values()) / len(analysis['gates_passed'])
                    pattern_score = entry_patterns['overall_strength']
                    analysis['signal_strength'] = (gate_score * 0.6 + pattern_score * 0.4)
                    
                    # Set recommendation based on signal strength
                    if analysis['signal_strength'] >= 0.7:
                        analysis['recommendation'] = 'STRONG_BUY'
                        analysis['confidence'] = min(95, analysis['signal_strength'] * 100)
                    elif analysis['signal_strength'] >= 0.5:
                        analysis['recommendation'] = 'BUY'
                        analysis['confidence'] = min(80, analysis['signal_strength'] * 100)
                    else:
                        analysis['recommendation'] = 'WEAK_BUY'
                        analysis['confidence'] = min(65, analysis['signal_strength'] * 100)
                else:
                    analysis['recommendation'] = 'WAIT'
                    analysis['reasons'].append("⚠ All gates passed but no entry patterns detected - wait for setup")
            else:
                analysis['recommendation'] = 'NO_SIGNAL'
                failed_gates = [k for k, v in analysis['gates_passed'].items() if not v]
                analysis['reasons'].append(f"⚠ Failed gates: {', '.join(failed_gates)}")
            
            # Add detailed metrics
            analysis['detailed_metrics'] = {
                'trend': trend_filter,
                'mtf': mtf_confirmation,
                'volatility': volatility_gate,
                'volume': volume_confirmation
            }

            # Structured observability log for gates and recommendation
            try:
                logger.info({
                    'event': 'swing_analysis_result',
                    'symbol': symbol,
                    'gates_passed': analysis['gates_passed'],
                    'all_gates_passed': analysis['all_gates_passed'],
                    'recommendation': analysis['recommendation'],
                    'signal_strength': round(analysis.get('signal_strength', 0), 3),
                    'reasons_count': len(analysis.get('reasons', []))
                })
            except Exception:
                pass
            
        except Exception as e:
            logger.error(f"Error in swing trading analysis for {symbol}: {e}")
            analysis['recommendation'] = 'ERROR'
            analysis['reasons'].append(f"Analysis error: {str(e)}")
        
        return analysis


# Helper function to get swing trading signals
def get_swing_trading_signals(symbol: str, daily_df: pd.DataFrame, 
                             weekly_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Get swing trading signals for a symbol
    
    Args:
        symbol: Stock symbol
        daily_df: Daily timeframe data
        weekly_df: Weekly timeframe data (optional)
        
    Returns:
        Swing trading analysis results
    """
    analyzer = SwingTradingSignalAnalyzer()
    return analyzer.analyze_swing_opportunity(symbol, daily_df, weekly_df)
