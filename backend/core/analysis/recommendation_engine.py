"""
Recommendation Engine Module
============================

Combines technical, fundamental, and sentiment analysis to generate
comprehensive stock recommendations.
Extracted from analyzer.py for better modularity.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from utils.logger import setup_logging
from config import ANALYSIS_WEIGHTS, RECOMMENDATION_THRESHOLDS

logger = setup_logging()

class RecommendationEngine:
    """
    Combines all analysis types to generate final recommendations.
    """
    
    def __init__(self):
        """Initialize the recommendation engine."""
        pass
    
    def generate_buy_sell_recommendations(self, current_price: float, sma_20: float, sma_50: float,
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
            
            # Generate recommendations - ONLY BUY OR HOLD, NEVER SELL
            if ma_cross_bullish and price_above_ma and rsi_oversold_recovery:
                recommendation = 'BUY'
                if breakout_potential:
                    entry_timing = 'IMMEDIATE'
                    buy_price = current_price
                    sell_price = resistance * 1.04  # Target 4% above resistance
                elif near_support:
                    entry_timing = 'IMMEDIATE'
                    buy_price = current_price
                    sell_price = current_price * 1.06  # Target 6% gain
                else:
                    entry_timing = 'WAIT_FOR_DIP'
                    buy_price = max(support * 1.01, current_price * 0.98)
                    sell_price = buy_price * 1.08  # Target 8% gain
                
                stop_loss = buy_price * 0.96  # 4% stop loss
                
            elif ma_cross_bearish or price_below_ma or rsi_overbought or near_resistance:
                recommendation = 'HOLD'
                entry_timing = 'WAIT'
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
                recommendation = 'HOLD'
                entry_timing = 'WAIT'
                buy_price = current_price * 0.92  # Wait for 8% correction
                sell_price = current_price * 1.10  # Target 10% gain
                stop_loss = current_price * 0.85   # 15% stop loss
                
            elif current_price < bb_lower:
                recommendation = 'BUY'
                entry_timing = 'IMMEDIATE'
                buy_price = current_price
                sell_price = current_price * 1.08  # Target 8% above lower band
                stop_loss = bb_lower * 0.95
            
            # For HOLD recommendations, set dynamic targets
            if recommendation == 'HOLD':
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
            
            # Ensure minimum risk-reward ratio
            if risk_reward_ratio < 2.0 and recommendation == 'BUY' and buy_price and stop_loss:
                volatility_pct = (atr / current_price) * 100
                min_ratio = self._calculate_dynamic_risk_reward_ratio(volatility_pct, rsi)
                
                risk = abs(buy_price - stop_loss)
                sell_price = buy_price + (risk * min_ratio)
                risk_reward_ratio = min_ratio
            
            # Ensure prices are realistic
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
                'sell_price': current_price * 1.15,
                'stop_loss': current_price * 0.92,
                'risk_reward_ratio': 1.87
            }
    
    def combine_analysis_results(self, result: Dict[str, Any], consider_backtest: bool = True, 
                                keep_reason_as_list: bool = False) -> Dict[str, Any]:
        """
        Combine technical, fundamental, and sentiment analysis results.
        
        Args:
            result: Analysis results dictionary
            consider_backtest: Whether to consider backtest results
            keep_reason_as_list: Whether to keep reason as list
            
        Returns:
            Updated results dictionary with combined recommendation
        """
        try:
            technical_score = result['technical_score']
            fundamental_score = result['fundamental_score']
            sentiment_score = result['sentiment_score']

            # Optional volume analysis payload (if upstream provides it)
            volume_info = result.get('volume_analysis') or result.get('volume') or {}
            vol_signal = volume_info.get('overall_signal') or volume_info.get('signal')
            vol_confidence = float(volume_info.get('confidence', volume_info.get('strength', 0.0)) or 0.0)

            # Get configurable weights and thresholds
            base_technical_weight = ANALYSIS_WEIGHTS.get('technical', 0.5)
            fundamental_weight = ANALYSIS_WEIGHTS.get('fundamental', 0.3)
            sentiment_weight = ANALYSIS_WEIGHTS.get('sentiment', 0.2)

            # Rebalance to favor technical when volume is supportive (swing context)
            # If bullish volume with decent confidence, boost technical weight slightly
            technical_weight = base_technical_weight
            reasons = result.get('reason', []) if keep_reason_as_list else []
            if isinstance(reasons, str):
                reasons = [reasons]

            if vol_signal in ('bullish', 'neutral') and vol_confidence:
                # Scale boost between 0 and ~0.15 depending on confidence
                boost = min(0.15, 0.20 * max(0.0, min(1.0, vol_confidence)))
                if vol_signal == 'bullish':
                    technical_weight = min(0.80, base_technical_weight + boost)
                    reasons.append(f"Technical weight boosted by volume ({vol_signal}, conf={vol_confidence:.2f})")
                else:  # neutral
                    technical_weight = min(0.75, base_technical_weight + boost * 0.5)
                    reasons.append(f"Technical weight slightly boosted by neutral volume (conf={vol_confidence:.2f})")
            elif vol_signal == 'bearish' and vol_confidence:
                # If bearish volume, slightly reduce technical reliance to avoid false positives
                reduction = min(0.10, 0.15 * max(0.0, min(1.0, vol_confidence)))
                technical_weight = max(0.35, base_technical_weight - reduction)
                reasons.append(f"Technical weight reduced due to bearish volume (conf={vol_confidence:.2f})")

            # Normalize weights across the available components (here: tech/fund/sent)
            total_weight = technical_weight + fundamental_weight + sentiment_weight
            if total_weight == 0:
                total_weight = 1.0
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
                'technical': round(technical_weight, 3),
                'fundamental': round(fundamental_weight, 3),
                'sentiment': round(sentiment_weight, 3)
            }
            
            # Get thresholds
            strong_buy_threshold = RECOMMENDATION_THRESHOLDS.get('strong_buy_combined', 0.3)
            buy_threshold = RECOMMENDATION_THRESHOLDS.get('buy_combined', 0.2)
            technical_strong_threshold = RECOMMENDATION_THRESHOLDS.get('technical_strong_buy', 0.5)
            
            # Apply recommendation logic
            if consider_backtest:
                backtest_cagr = result.get('backtest', {}).get('combined_metrics', {}).get('avg_cagr', 0)
                trade_plan = result.get('trade_plan', {})
                days_to_target = trade_plan.get('days_to_target', 0) if trade_plan else 0
                
                # Flexible backtest requirements
                strong_analysis_override = (
                    (technical_score > 0.3 and fundamental_score > 0.3) or
                    (combined_score > 0.3) or
                    (technical_score > 0.4) or
                    (fundamental_score > 0.5)
                )
                
                if strong_analysis_override:
                    backtest_condition = True
                elif days_to_target > 30:
                    min_backtest_return = 1.0
                    backtest_condition = backtest_cagr >= min_backtest_return
                else:
                    min_backtest_return = RECOMMENDATION_THRESHOLDS.get('min_backtest_return', 0.0)
                    backtest_condition = backtest_cagr >= min_backtest_return
            else:
                backtest_condition = True
            
            # Enhanced recommendation logic
            technical_minimum = RECOMMENDATION_THRESHOLDS.get('technical_minimum', -0.1)
            fundamental_minimum = RECOMMENDATION_THRESHOLDS.get('fundamental_minimum', -0.2)

            # Add a small gate for volume if required
            require_volume = RECOMMENDATION_THRESHOLDS.get('volume_confirmation_required', False)
            volume_ok = True
            if require_volume:
                # Consider bullish or high-confidence neutral as acceptable
                volume_ok = (vol_signal == 'bullish' and vol_confidence >= 0.6) or \
                            (vol_signal == 'neutral' and vol_confidence >= 0.7)
                if not volume_ok:
                    reasons.append("Volume confirmation not sufficient for recommendation")
            
            # Sector filter - avoid weak sector regimes
            sector_ok = True
            sector_filter_enabled = RECOMMENDATION_THRESHOLDS.get('sector_filter_enabled', True)
            if sector_filter_enabled:
                sector_analysis = result.get('sector_analysis', {})
                sector_score = sector_analysis.get('sector_score', 0)
                sector_name = sector_analysis.get('sector', 'Unknown')
                
                # Minimum sector score threshold (configurable)
                min_sector_score = RECOMMENDATION_THRESHOLDS.get('min_sector_score', -0.2)
                
                if sector_name != 'Unknown' and sector_score < min_sector_score:
                    sector_ok = False
                    reasons.append(f"Weak sector regime ({sector_name}: score {sector_score:.2f})")
                elif sector_name == 'Unknown':
                    # If sector is unknown, use a more lenient approach
                    logger.warning(f"Sector unknown for stock, applying lenient filter")
                    sector_ok = True  # Don't block if we can't identify sector

            # Centralized gate enforcement (trend, volatility, volume, MTF) when available
            require_all_gates = RECOMMENDATION_THRESHOLDS.get('require_all_gates', False)
            gates_ok = True
            gates_source = None
            if require_all_gates:
                # Look for gates in common locations produced by swing analysis modules
                gates = None
                if isinstance(result.get('gates_passed'), dict):
                    gates = result.get('gates_passed')
                    gates_source = 'root.gates_passed'
                elif isinstance(result.get('swing_analysis', {}), dict) and isinstance(result['swing_analysis'].get('gates_passed'), dict):
                    gates = result['swing_analysis']['gates_passed']
                    gates_source = 'swing_analysis.gates_passed'
                elif isinstance(result.get('swing', {}), dict) and isinstance(result['swing'].get('gates_passed'), dict):
                    gates = result['swing']['gates_passed']
                    gates_source = 'swing.gates_passed'

                if isinstance(gates, dict) and gates:
                    gates_ok = all(bool(v) for v in gates.values())
                    if not gates_ok:
                        failed = [k for k, v in gates.items() if not v]
                        reasons.append(f"Gate check failed ({gates_source}): {', '.join(failed)}")
                else:
                    # If gate details are not present, do not block recommendations
                    gates_ok = True
            
            # Strong buy conditions (tighter)
            if (
                combined_score >= strong_buy_threshold and
                technical_score >= technical_strong_threshold and
                backtest_condition and volume_ok and gates_ok and sector_ok
            ):
                result['is_recommended'] = True
                result['recommendation_strength'] = 'STRONG_BUY'
                if not keep_reason_as_list:
                    result['reason'] = (
                        f"Combined score {combined_score:.2f} >= strong threshold {strong_buy_threshold:.2f} "
                        f"and technical {technical_score:.2f} >= {technical_strong_threshold:.2f} with all gates passing"
                    )
                else:
                    reasons.append(
                        f"Strong BUY: combined {combined_score:.2f} >= {strong_buy_threshold:.2f}, "
                        f"technical {technical_score:.2f} >= {technical_strong_threshold:.2f}, gates OK"
                    )
            
            # Regular buy conditions (tighter)
            elif (
                combined_score >= buy_threshold and
                technical_score >= technical_minimum and
                fundamental_score >= fundamental_minimum and
                backtest_condition and volume_ok and gates_ok and sector_ok
            ):
                result['is_recommended'] = True
                result['recommendation_strength'] = 'BUY'
                if not keep_reason_as_list:
                    result['reason'] = (
                        f"BUY: combined {combined_score:.2f} >= {buy_threshold:.2f}, "
                        f"technical {technical_score:.2f} >= {technical_minimum:.2f}, "
                        f"fundamental {fundamental_score:.2f} >= {fundamental_minimum:.2f}, gates OK"
                    )
                else:
                    reasons.append(
                        f"BUY conditions met at stricter thresholds and gates OK"
                    )
            
            # Borderline cases - use fundamentals as tie-breaker
            elif (
                0.9 * buy_threshold <= combined_score < buy_threshold and
                technical_score >= technical_minimum * 0.9 and
                backtest_condition and volume_ok and gates_ok and sector_ok
            ):
                # Check for strong fundamentals as tie-breaker
                fundamental_details = result.get('fundamental_details', {})
                eps_growth = fundamental_details.get('eps_growth', 0)
                de_ratio = fundamental_details.get('de_ratio', float('inf'))
                profit_margins = fundamental_details.get('profit_margins', 0)
                roe = fundamental_details.get('roe', 0)
                
                # Strong fundamentals criteria for tie-breaking
                strong_fundamentals = (
                    eps_growth > 0.1 and  # EPS growth > 10%
                    de_ratio < 0.5 and  # Low debt/equity
                    profit_margins > 0.1 and  # Stable margins > 10%
                    roe > 0.15  # Good return on equity > 15%
                )
                
                if strong_fundamentals:
                    result['is_recommended'] = True
                    result['recommendation_strength'] = 'BUY'
                    if not keep_reason_as_list:
                        result['reason'] = (
                            f"BUY (Fundamental tie-breaker): Strong fundamentals with "
                            f"EPS growth {eps_growth:.1%}, D/E {de_ratio:.2f}, "
                            f"margins {profit_margins:.1%}, ROE {roe:.1%}"
                        )
                    else:
                        reasons.append(
                            f"Borderline case resolved by strong fundamentals: "
                            f"EPS growth {eps_growth:.1%}, D/E {de_ratio:.2f}"
                        )
                else:
                    result['is_recommended'] = False
                    result['recommendation_strength'] = 'HOLD'
                    if not keep_reason_as_list:
                        result['reason'] = "Borderline case - fundamentals not strong enough for BUY"
                    else:
                        reasons.append("Borderline case - fundamentals not strong enough")
            
            else:
                result['is_recommended'] = False
                result['recommendation_strength'] = 'HOLD'
                if not keep_reason_as_list:
                    result['reason'] = "Analysis does not support buying at this time"
                else:
                    reasons.append("Analysis does not support buying at this time")
            
            # Attach accumulated reasons if list mode
            if keep_reason_as_list:
                result['reason'] = reasons
            else:
                if isinstance(result.get('reason'), list):
                    result['reason'] = " ".join(result['reason'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error combining analysis results: {e}")
            result['is_recommended'] = False
            result['recommendation_strength'] = 'HOLD'
            return result
    
    def _calculate_dynamic_hold_targets(self, current_price: float, atr: float, rsi: float,
                                       support: float, resistance: float, bb_upper: float, 
                                       bb_lower: float) -> Dict[str, float]:
        """Calculate dynamic buy/sell targets for HOLD recommendations."""
        try:
            volatility_pct = (atr / current_price) * 100
            
            # Base adjustments based on volatility
            if volatility_pct > 4.0:
                buy_discount = np.random.uniform(0.08, 0.12)
                sell_premium = np.random.uniform(0.18, 0.25)
                stop_discount = np.random.uniform(0.12, 0.18)
            elif volatility_pct > 2.5:
                buy_discount = np.random.uniform(0.05, 0.08)
                sell_premium = np.random.uniform(0.12, 0.18)
                stop_discount = np.random.uniform(0.08, 0.12)
            else:
                buy_discount = np.random.uniform(0.03, 0.06)
                sell_premium = np.random.uniform(0.08, 0.15)
                stop_discount = np.random.uniform(0.05, 0.08)
            
            # Adjust based on RSI
            if rsi > 65:
                buy_discount += 0.03
                sell_premium *= 0.85
            elif rsi < 35:
                buy_discount *= 0.7
                sell_premium += 0.03
            
            # Calculate prices
            buy_price = current_price * (1 - buy_discount)
            sell_price = current_price * (1 + sell_premium)
            stop_loss = current_price * (1 - stop_discount)
            
            # Ensure bounds
            buy_price = max(buy_price, current_price * 0.85)
            sell_price = min(sell_price, current_price * 1.35)
            stop_loss = max(stop_loss, current_price * 0.75)
            
            return {
                'buy_price': buy_price,
                'sell_price': sell_price,
                'stop_loss': stop_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating dynamic HOLD targets: {e}")
            base_discount = np.random.uniform(0.04, 0.07)
            base_premium = np.random.uniform(0.10, 0.16)
            base_stop = np.random.uniform(0.08, 0.12)
            
            return {
                'buy_price': current_price * (1 - base_discount),
                'sell_price': current_price * (1 + base_premium),
                'stop_loss': current_price * (1 - base_stop)
            }
    
    def _calculate_dynamic_risk_reward_ratio(self, volatility_pct: float, rsi: float) -> float:
        """Calculate dynamic minimum risk-reward ratio."""
        try:
            if volatility_pct > 4.0:
                volatility_adjustment = np.random.uniform(1.8, 2.2)
            elif volatility_pct > 2.5:
                volatility_adjustment = np.random.uniform(2.2, 2.8)
            else:
                volatility_adjustment = np.random.uniform(2.5, 3.2)
            
            # RSI adjustment
            if rsi > 65:
                rsi_adjustment = 1.15
            elif rsi < 35:
                rsi_adjustment = 0.85
            else:
                rsi_adjustment = 1.0
                
            final_ratio = volatility_adjustment * rsi_adjustment
            final_ratio *= np.random.uniform(0.95, 1.05)
            
            return max(1.5, min(final_ratio, 4.0))
            
        except Exception as e:
            logger.error(f"Error calculating dynamic risk-reward ratio: {e}")
            return 2.5
