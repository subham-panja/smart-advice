"""
Trade Logic Module
File: scripts/trade_logic.py

This module contains logic for calculating entry/exit points, support/resistance,
and trade timing based on technical indicators.
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any
import logging
from config import SWING_PATTERNS, RECOMMENDATION_THRESHOLDS, RISK_MANAGEMENT

logger = logging.getLogger(__name__)

class TradeLogic:
    """Class for trade-level analysis and recommendations."""
    
    def __init__(self):
        pass

    def analyze(self, symbol: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute buy/sell recommendations and timing for a stock.
        """
        try:
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
            
            # Get latest values with fallbacks
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
            
            # Generate buy/sell recommendations
            recommendation_data = self._generate_buy_sell_recommendations(
                current_price, current_sma_20, current_sma_50, current_ema_12, current_ema_26,
                current_rsi, current_atr, current_bb_upper, current_bb_lower,
                support_level, resistance_level
            )
            
            # Calculate days to target
            days_to_target = self._estimate_days_to_target(
                current_price, recommendation_data['sell_price'], current_atr
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                current_price, current_sma_20, current_sma_50, current_rsi,
                recommendation_data['recommendation']
            )
            
            return {
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
        except Exception as e:
            logger.error(f"Error in TradeLogic.analyze for {symbol}: {e}")
            return {'error': str(e), 'recommendation': 'HOLD'}

    def _find_support_resistance(self, data: pd.DataFrame, level_type: str) -> float:
        """Find support or resistance levels using pivot points."""
        try:
            if len(data) < 10:
                return data['Low'].min() if level_type == 'support' else data['High'].max()
            
            lookback = min(20, len(data))
            recent_data = data.tail(lookback)
            
            if level_type == 'support':
                levels = []
                for i in range(2, len(recent_data) - 2):
                    val = recent_data['Low'].iloc[i]
                    if (val < recent_data['Low'].iloc[i-1] and val < recent_data['Low'].iloc[i+1] and
                        val < recent_data['Low'].iloc[i-2] and val < recent_data['Low'].iloc[i+2]):
                        levels.append(val)
                return max(levels) if levels else recent_data['Low'].min()
            else:
                levels = []
                for i in range(2, len(recent_data) - 2):
                    val = recent_data['High'].iloc[i]
                    if (val > recent_data['High'].iloc[i-1] and val > recent_data['High'].iloc[i+1] and
                        val > recent_data['High'].iloc[i-2] and val > recent_data['High'].iloc[i+2]):
                        levels.append(val)
                return min(levels) if levels else recent_data['High'].max()
        except:
            return data['Low'].min() if level_type == 'support' else data['High'].max()

    def _generate_buy_sell_recommendations(self, current_price: float, sma_20: float, sma_50: float,
                                         ema_12: float, ema_26: float, rsi: float, atr: float,
                                         bb_upper: float, bb_lower: float, support: float,
                                         resistance: float) -> Dict[str, Any]:
        """Generate recommendations based on technical indicators."""
        recommendation = 'HOLD'
        entry_timing = 'WAIT'
        buy_price = current_price
        sell_price = current_price
        stop_loss = current_price * 0.95
        risk_reward_ratio = 0
        
        if atr < current_price * 0.005: atr = current_price * 0.02
        if abs(support - current_price) < current_price * 0.02: support = current_price * 0.97
        if abs(resistance - current_price) < current_price * 0.02: resistance = current_price * 1.05
        
        ma_cross_bullish = ema_12 > ema_26 and sma_20 > sma_50
        price_above_ma = current_price > sma_20 and current_price > sma_50
        exit_rules = SWING_PATTERNS.get('exit_rules', {})
        atr_sl_mult = exit_rules.get('atr_stop_multiplier', 1.5)
        
        # Determine RSI range from patterns (pullback_to_ema as proxy for neutral zone)
        pullback_config = next((p for p in SWING_PATTERNS.get('entry_patterns', []) if p['name'] == 'pullback_to_ema'), {})
        rsi_min, rsi_max = pullback_config.get('rsi_range', [40, 60])
        
        rsi_neutral = rsi_min <= rsi <= rsi_max
        near_support = abs(current_price - support) / current_price < 0.05
        
        if ma_cross_bullish and price_above_ma and rsi_neutral:
            recommendation = 'BUY'
            if current_price > (resistance * 0.98):
                entry_timing = 'IMMEDIATE'
                sell_price = resistance * 1.04
            elif near_support:
                entry_timing = 'IMMEDIATE'
                sell_price = current_price * 1.06
            else:
                entry_timing = 'WAIT_FOR_DIP'
                buy_price = max(support * 1.01, current_price * 0.98)
                sell_price = buy_price * 1.08
            stop_loss = buy_price * 0.96
        elif rsi < 30 and near_support:
            recommendation = 'BUY'
            entry_timing = 'WAIT_FOR_BREAKOUT'
            buy_price = support * 1.005
            sell_price = buy_price * 1.08
            stop_loss = support * 0.96
        elif current_price < bb_lower:
            recommendation = 'BUY'
            entry_timing = 'IMMEDIATE'
            sell_price = current_price * 1.08
            stop_loss = bb_lower * 0.95
        
        if recommendation == 'HOLD':
            targets = self._calculate_dynamic_hold_targets(current_price, atr, rsi, support, resistance, bb_upper, bb_lower)
            buy_price, sell_price, stop_loss = targets['buy_price'], targets['sell_price'], targets['stop_loss']
        
        if recommendation == 'BUY':
            risk = abs(buy_price - stop_loss)
            reward = abs(sell_price - buy_price)
            min_rr = RECOMMENDATION_THRESHOLDS.get('min_risk_reward_ratio', 1.8)
            if risk_reward_ratio < min_rr:
                vol_pct = (atr / current_price) * 100
                min_ratio = self._calculate_dynamic_risk_reward_ratio(vol_pct, rsi)
                # Ensure we at least meet the global minimum
                min_ratio = max(min_ratio, min_rr)
                sell_price = buy_price + (risk * min_ratio)
                risk_reward_ratio = min_ratio

        return {
            'recommendation': recommendation, 'entry_timing': entry_timing,
            'buy_price': buy_price, 'sell_price': sell_price, 'stop_loss': stop_loss,
            'risk_reward_ratio': round(risk_reward_ratio, 2)
        }

    def _estimate_days_to_target(self, current_price: float, target_price: float, atr: float) -> int:
        """Estimate days to target."""
        if not target_price or not current_price or current_price <= 0: return 30
        pct_move = abs(target_price - current_price) / current_price
        if pct_move < 0.01: return 7
        
        daily_vol = (atr / current_price) if current_price > 0 else 0.02
        if pct_move <= 0.05: base = 15 + (pct_move * 200)
        elif pct_move <= 0.10: base = 25 + ((pct_move - 0.05) * 600)
        else: base = 55 + ((pct_move - 0.10) * 300)
        
        adj = 1.0
        if daily_vol > 0.03: adj = 0.7
        elif daily_vol < 0.015: adj = 1.3
        
        return int(max(7, min(base * adj, 120)))

    def _calculate_confidence(self, price: float, sma20: float, sma50: float, rsi: float, rec: str) -> float:
        conf = 0.5
        if rec == 'BUY':
            if price > sma20 > sma50: conf += 0.2
            elif price > sma20: conf += 0.1
            if 40 < rsi < 60: conf += 0.2
        return round(max(0.0, min(1.0, conf)), 2)

    def _calculate_dynamic_hold_targets(self, price: float, atr: float, rsi: float, support: float, resistance: float, bb_u: float, bb_l: float) -> Dict[str, float]:
        vol_pct = (atr / price) * 100
        if vol_pct > 4.0: bd, sp, sd = 0.10, 0.22, 0.15
        elif vol_pct > 2.5: bd, sp, sd = 0.06, 0.15, 0.10
        else: bd, sp, sd = 0.04, 0.12, 0.06
        
        if rsi > 65: bd += 0.03; sp *= 0.85
        elif rsi < 35: bd *= 0.7; sp += 0.03
        
        buy = support * 1.005 if abs(price - support) / price < 0.05 else price * (1 - bd)
        sell = min(resistance * 1.02, price * (1 + sp)) if abs(resistance - price) / price < 0.08 else price * (1 + sp)
        return {'buy_price': buy, 'sell_price': sell, 'stop_loss': price * (1 - sd)}

    def _calculate_dynamic_risk_reward_ratio(self, vol_pct: float, rsi: float) -> float:
        ratio = 2.0
        if vol_pct > 4.0: ratio = 2.0
        elif vol_pct > 2.5: ratio = 2.5
        else: ratio = 3.0
        if rsi > 65: ratio *= 1.15
        elif rsi < 35: ratio *= 0.85
        return round(max(1.5, min(ratio, 4.0)), 2)
