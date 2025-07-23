"""
Risk Management Module
File: scripts/risk_management.py

This module implements comprehensive risk management features including:
- Stop-loss calculations
- Position sizing based on account risk
- Risk-reward ratio calculations
- Maximum drawdown protection
- ATR-based stop losses
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any, Optional, Tuple, List
from utils.logger import setup_logging
from scripts.position_sizing import PositionSizer

logger = setup_logging()

class RiskManager:
    """
    Professional risk management system for swing trading.
    """
    
    def __init__(self, account_balance: float = 100000.0, max_risk_per_trade: float = 0.02, 
                 max_total_risk: float = 0.06, max_drawdown: float = 0.20):
        """
        Initialize the risk manager.
        
        Args:
            account_balance: Total account balance
            max_risk_per_trade: Maximum risk per trade (default 2%)
            max_total_risk: Maximum total portfolio risk (default 6%)
            max_drawdown: Maximum allowable drawdown (default 20%)
        """
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_risk = max_total_risk
        self.max_drawdown = max_drawdown
        self.open_positions = {}  # Track open positions for portfolio risk
        
        # Initialize advanced position sizer
        self.position_sizer = PositionSizer(
            account_balance=account_balance,
            base_risk_per_trade=max_risk_per_trade
        )
        
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              risk_per_trade: Optional[float] = None, 
                              method: str = 'fixed_risk', 
                              data: Optional[pd.DataFrame] = None,
                              **kwargs) -> Dict[str, Any]:
        """
        Calculate position size using advanced methods with backward compatibility.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_per_trade: Risk per trade (uses default if None)
            method: Position sizing method ('fixed_risk', 'atr', 'kelly', 'percent_volatility', 'market_condition')
            data: Historical price data (required for advanced methods)
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary containing position size information
        """
        try:
            risk_pct = risk_per_trade or self.max_risk_per_trade
            
            # For backward compatibility, default to original method
            if method == 'fixed_risk' or data is None:
                return self._calculate_basic_position_size(entry_price, stop_loss, risk_pct)
            
            # Use advanced position sizer for sophisticated methods
            self.position_sizer.update_account_balance(self.account_balance)
            self.position_sizer.update_risk_per_trade(risk_pct)
            
            if method == 'atr':
                result = self.position_sizer.atr_based_sizing(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    data=data,
                    atr_multiplier=kwargs.get('atr_multiplier', 2.0)
                )
            elif method == 'kelly':
                win_rate = kwargs.get('win_rate', 0.55)  # Default 55% win rate
                avg_win_loss_ratio = kwargs.get('avg_win_loss_ratio', 1.5)
                result = self.position_sizer.kelly_criterion_sizing(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    win_rate=win_rate,
                    avg_win_loss_ratio=avg_win_loss_ratio
                )
            elif method == 'percent_volatility':
                result = self.position_sizer.percent_volatility_sizing(
                    entry_price=entry_price,
                    data=data,
                    volatility_target=kwargs.get('volatility_target', 0.20)
                )
            elif method == 'market_condition':
                market_condition = kwargs.get('market_condition', 'normal')
                result = self.position_sizer.market_condition_sizing(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    data=data,
                    market_condition=market_condition
                )
            else:
                # Fall back to basic method for unknown methods
                return self._calculate_basic_position_size(entry_price, stop_loss, risk_pct)
            
            # Add additional metadata
            result['method'] = method
            result['risk_percentage'] = (result['risk_amount'] / self.account_balance) * 100
            result['position_percentage'] = (result['position_value'] / self.account_balance) * 100
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating position size with method {method}: {e}")
            # Fall back to basic method on error
            return self._calculate_basic_position_size(entry_price, stop_loss, risk_pct)
    
    def _calculate_basic_position_size(self, entry_price: float, stop_loss: float, 
                                     risk_pct: float) -> Dict[str, Any]:
        """
        Original basic position size calculation for backward compatibility.
        """
        try:
            risk_amount = self.account_balance * risk_pct
            
            # Calculate risk per share
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                return {
                    'position_size': 0,
                    'risk_amount': 0,
                    'error': 'Invalid stop loss - must be different from entry price'
                }
            
            # Calculate position size
            position_size = int(risk_amount / risk_per_share)
            
            # Calculate actual risk amount
            actual_risk = position_size * risk_per_share
            
            # Calculate position value
            position_value = position_size * entry_price
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'risk_amount': actual_risk,
                'risk_per_share': risk_per_share,
                'method': 'fixed_risk',
                'risk_percentage': (actual_risk / self.account_balance) * 100,
                'position_percentage': (position_value / self.account_balance) * 100
            }
            
        except Exception as e:
            logger.error(f"Error in basic position size calculation: {e}")
            return {
                'position_size': 0,
                'risk_amount': 0,
                'error': str(e)
            }
    
    def calculate_stop_loss(self, data: pd.DataFrame, entry_price: float, 
                          method: str = 'atr') -> Dict[str, Any]:
        """
        Calculate stop loss based on different methods.
        
        Args:
            data: Historical price data
            entry_price: Entry price for the trade
            method: Method to use ('atr', 'support', 'percentage', 'combined')
            
        Returns:
            Dictionary containing stop loss information
        """
        try:
            if data.empty:
                return {'stop_loss': entry_price * 0.95, 'method': 'default_5pct'}
            
            # Calculate ATR for volatility-based stop loss
            atr = ta.ATR(data['High'].values, data['Low'].values, 
                        data['Close'].values, timeperiod=14)
            current_atr = atr[-1] if not pd.isna(atr[-1]) else entry_price * 0.02
            
            # Calculate support level
            support_level = self.find_support_level(data)
            
            stop_losses = {}
            
            if method == 'atr' or method == 'combined':
                # ATR-based stop loss (2 ATR below entry)
                atr_stop = entry_price - (current_atr * 2)
                stop_losses['atr'] = atr_stop
                
            if method == 'support' or method == 'combined':
                # Support-based stop loss (2% below support)
                support_stop = support_level * 0.98
                stop_losses['support'] = support_stop
                
            if method == 'percentage' or method == 'combined':
                # Percentage-based stop loss (5% below entry)
                pct_stop = entry_price * 0.95
                stop_losses['percentage'] = pct_stop
            
            # Choose the most conservative (highest) stop loss for long positions
            if method == 'combined':
                stop_loss = max(stop_losses.values())
                chosen_method = max(stop_losses, key=stop_losses.get)
            else:
                stop_loss = stop_losses.get(method, entry_price * 0.95)
                chosen_method = method
            
            # Ensure stop loss is reasonable (not too close to entry)
            min_stop_distance = entry_price * 0.02  # Minimum 2% stop distance
            if entry_price - stop_loss < min_stop_distance:
                stop_loss = entry_price - min_stop_distance
                chosen_method = 'min_distance_adjusted'
            
            return {
                'stop_loss': stop_loss,
                'method': chosen_method,
                'atr_value': current_atr,
                'support_level': support_level,
                'stop_distance_pct': ((entry_price - stop_loss) / entry_price) * 100,
                'all_stops': stop_losses
            }
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return {
                'stop_loss': entry_price * 0.95,
                'method': 'error_default',
                'error': str(e)
            }
    
    def find_support_level(self, data: pd.DataFrame, lookback: int = 20) -> float:
        """
        Find the nearest support level.
        
        Args:
            data: Historical price data
            lookback: Number of periods to look back
            
        Returns:
            Support level price
        """
        try:
            if len(data) < lookback:
                return data['Low'].min()
            
            # Get recent low prices
            recent_lows = data['Low'].tail(lookback)
            
            # Find local minima (support levels)
            support_levels = []
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows.iloc[i] < recent_lows.iloc[i-1] and 
                    recent_lows.iloc[i] < recent_lows.iloc[i+1] and
                    recent_lows.iloc[i] < recent_lows.iloc[i-2] and 
                    recent_lows.iloc[i] < recent_lows.iloc[i+2]):
                    support_levels.append(recent_lows.iloc[i])
            
            # Return the highest support level (most recent/relevant)
            if support_levels:
                return max(support_levels)
            else:
                return recent_lows.min()
                
        except Exception as e:
            logger.error(f"Error finding support level: {e}")
            return data['Low'].min() if not data.empty else 0
    
    def calculate_profit_targets(self, entry_price: float, stop_loss: float, 
                               risk_reward_ratios: list = [2, 3]) -> Dict[str, Any]:
        """
        Calculate profit targets based on risk-reward ratios.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_reward_ratios: List of risk-reward ratios to calculate
            
        Returns:
            Dictionary containing profit targets
        """
        try:
            risk_per_share = abs(entry_price - stop_loss)
            
            targets = {}
            for ratio in risk_reward_ratios:
                target_price = entry_price + (risk_per_share * ratio)
                targets[f'target_{ratio}x'] = {
                    'price': target_price,
                    'profit_per_share': risk_per_share * ratio,
                    'risk_reward_ratio': ratio
                }
            
            return {
                'targets': targets,
                'risk_per_share': risk_per_share,
                'entry_price': entry_price,
                'stop_loss': stop_loss
            }
            
        except Exception as e:
            logger.error(f"Error calculating profit targets: {e}")
            return {'targets': {}, 'error': str(e)}
    
    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, 
                                  target_price: float) -> float:
        """
        Calculate risk-reward ratio for a trade.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            target_price: Target price
            
        Returns:
            Risk-reward ratio
        """
        try:
            risk = abs(entry_price - stop_loss)
            reward = abs(target_price - entry_price)
            
            if risk <= 0:
                return 0
            
            return reward / risk
            
        except Exception as e:
            logger.error(f"Error calculating risk-reward ratio: {e}")
            return 0
    
    def validate_trade(self, position_size: int, entry_price: float, 
                      current_portfolio_risk: float = 0) -> Dict[str, Any]:
        """
        Validate if a trade meets risk management criteria.
        
        Args:
            position_size: Proposed position size
            entry_price: Entry price
            current_portfolio_risk: Current portfolio risk percentage
            
        Returns:
            Dictionary with validation results
        """
        try:
            position_value = position_size * entry_price
            position_risk_pct = (position_value / self.account_balance) * 100
            
            # Check individual position risk
            if position_risk_pct > (self.max_risk_per_trade * 100):
                return {
                    'valid': False,
                    'reason': f'Position risk ({position_risk_pct:.2f}%) exceeds max per trade ({self.max_risk_per_trade*100:.2f}%)'
                }
            
            # Check total portfolio risk
            total_risk = current_portfolio_risk + position_risk_pct
            if total_risk > (self.max_total_risk * 100):
                return {
                    'valid': False,
                    'reason': f'Total portfolio risk ({total_risk:.2f}%) would exceed maximum ({self.max_total_risk*100:.2f}%)'
                }
            
            # Check minimum position size
            min_position_value = self.account_balance * 0.01  # Minimum 1% of account
            if position_value < min_position_value:
                return {
                    'valid': False,
                    'reason': f'Position too small (${position_value:.2f}), minimum is ${min_position_value:.2f}'
                }
            
            return {
                'valid': True,
                'position_risk_pct': position_risk_pct,
                'total_portfolio_risk': total_risk,
                'position_value': position_value
            }
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return {
                'valid': False,
                'reason': f'Validation error: {str(e)}'
            }
    
    def calculate_pivot_points(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate pivot points for support and resistance levels.
        
        Args:
            data: Historical OHLC data
            
        Returns:
            Dictionary with pivot points
        """
        try:
            if len(data) < 1:
                return {}
            
            # Use previous day's data for pivot calculation
            high = data['High'].iloc[-1]
            low = data['Low'].iloc[-1]
            close = data['Close'].iloc[-1]
            
            # Calculate pivot point
            pivot = (high + low + close) / 3
            
            # Calculate resistance levels
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            # Calculate support levels
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'resistance_1': r1,
                'resistance_2': r2,
                'resistance_3': r3,
                'support_1': s1,
                'support_2': s2,
                'support_3': s3
            }
            
        except Exception as e:
            logger.error(f"Error calculating pivot points: {e}")
            return {}
    
    def evaluate_recommendation_risk(self, symbol: str, entry_price: float, 
                                   technical_score: float, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive risk evaluation for stock recommendation.
        
        Args:
            symbol: Stock symbol
            entry_price: Proposed entry price
            technical_score: Technical analysis score (0-1)
            data: Historical price data
            
        Returns:
            Dictionary with comprehensive risk assessment
        """
        try:
            # Calculate optimal stop loss
            stop_loss_info = self.calculate_stop_loss(data, entry_price, method='combined')
            stop_loss = stop_loss_info['stop_loss']
            
            # Calculate position size based on risk
            position_info = self.calculate_position_size(entry_price, stop_loss)
            
            # Calculate profit targets
            target_info = self.calculate_profit_targets(entry_price, stop_loss)
            
            # Calculate volatility metrics
            atr_values = ta.ATR(data['High'].values, data['Low'].values, 
                              data['Close'].values, timeperiod=14)
            current_atr = atr_values[-1] if not pd.isna(atr_values[-1]) else 0
            volatility_pct = (current_atr / entry_price) * 100
            
            # Risk-adjusted score based on technical score and volatility
            volatility_penalty = min(volatility_pct / 5.0, 0.3)  # Max 30% penalty for high volatility
            risk_adjusted_score = technical_score * (1 - volatility_penalty)
            
            # Market risk assessment
            market_risk = self.assess_market_conditions(data)
            
            # Generate recommendation with risk context
            risk_recommendation = self.generate_risk_recommendation(
                risk_adjusted_score, volatility_pct, market_risk
            )
            
            return {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'stop_loss_method': stop_loss_info['method'],
                'stop_distance_pct': stop_loss_info.get('stop_distance_pct', 0),
                'position_size': position_info['position_size'],
                'position_value': position_info['position_value'],
                'risk_amount': position_info['risk_amount'],
                'risk_percentage': position_info['risk_percentage'],
                'profit_targets': target_info['targets'],
                'volatility_pct': volatility_pct,
                'technical_score': technical_score,
                'risk_adjusted_score': risk_adjusted_score,
                'market_risk_level': market_risk['risk_level'],
                'risk_recommendation': risk_recommendation,
                'risk_notes': self.generate_risk_notes(volatility_pct, stop_loss_info, position_info)
            }
            
        except Exception as e:
            logger.error(f"Error in risk evaluation for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'risk_recommendation': 'AVOID - Risk evaluation failed'
            }
    
    def assess_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall market conditions for risk management.
        
        Args:
            data: Historical price data
            
        Returns:
            Dictionary with market risk assessment
        """
        try:
            # Calculate recent volatility
            returns = data['Close'].pct_change().dropna()
            recent_volatility = returns.tail(20).std() * np.sqrt(252)  # Annualized
            
            # Calculate trend strength
            sma_20 = data['Close'].rolling(20).mean()
            sma_50 = data['Close'].rolling(50).mean()
            trend_strength = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(20).mean()
            recent_volume_ratio = data['Volume'].tail(5).mean() / avg_volume.iloc[-1]
            
            # Determine risk level
            if recent_volatility > 0.4:  # High volatility
                risk_level = 'HIGH'
            elif recent_volatility > 0.25:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            return {
                'risk_level': risk_level,
                'volatility_annualized': recent_volatility,
                'trend_strength': trend_strength,
                'volume_ratio': recent_volume_ratio,
                'trend_direction': 'BULLISH' if trend_strength > 0.02 else 'BEARISH' if trend_strength < -0.02 else 'NEUTRAL'
            }
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            return {
                'risk_level': 'HIGH',
                'error': str(e)
            }
    
    def generate_risk_recommendation(self, risk_adjusted_score: float, 
                                   volatility_pct: float, market_risk: Dict[str, Any]) -> str:
        """
        Generate risk-based recommendation.
        
        Args:
            risk_adjusted_score: Risk-adjusted technical score
            volatility_pct: Current volatility percentage
            market_risk: Market risk assessment
            
        Returns:
            Risk recommendation string
        """
        try:
            # Base recommendation on risk-adjusted score
            if risk_adjusted_score >= 0.7:
                base_rec = 'STRONG_BUY'
            elif risk_adjusted_score >= 0.6:
                base_rec = 'BUY'
            elif risk_adjusted_score >= 0.4:
                base_rec = 'HOLD'
            else:
                base_rec = 'AVOID'
            
            # Adjust for market conditions
            if market_risk['risk_level'] == 'HIGH':
                if base_rec == 'STRONG_BUY':
                    base_rec = 'BUY'  # Downgrade in high-risk environment
                elif base_rec == 'BUY':
                    base_rec = 'HOLD'
            
            # Adjust for high volatility
            if volatility_pct > 8.0:  # Very high volatility
                if base_rec in ['STRONG_BUY', 'BUY']:
                    base_rec += '_WITH_CAUTION'
            
            return base_rec
            
        except Exception as e:
            logger.error(f"Error generating risk recommendation: {e}")
            return 'HOLD'
    
    def generate_risk_notes(self, volatility_pct: float, stop_loss_info: Dict[str, Any], 
                          position_info: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable risk management notes.
        
        Args:
            volatility_pct: Current volatility percentage
            stop_loss_info: Stop loss calculation info
            position_info: Position sizing info
            
        Returns:
            List of risk management notes
        """
        notes = []
        
        try:
            # Volatility notes
            if volatility_pct > 8.0:
                notes.append(f"⚠️ High volatility ({volatility_pct:.1f}%) - Consider smaller position size")
            elif volatility_pct > 5.0:
                notes.append(f"⚡ Moderate volatility ({volatility_pct:.1f}%) - Monitor closely")
            else:
                notes.append(f"✅ Low volatility ({volatility_pct:.1f}%) - Favorable risk profile")
            
            # Stop loss notes
            stop_distance = stop_loss_info.get('stop_distance_pct', 0)
            if stop_distance > 8.0:
                notes.append(f"🛑 Wide stop loss ({stop_distance:.1f}%) - Higher risk per share")
            elif stop_distance > 5.0:
                notes.append(f"🎯 Moderate stop loss ({stop_distance:.1f}%) - Standard risk")
            else:
                notes.append(f"🔒 Tight stop loss ({stop_distance:.1f}%) - Limited downside risk")
            
            # Position size notes
            risk_pct = position_info.get('risk_percentage', 0)
            if risk_pct > 2.5:
                notes.append(f"📊 High position risk ({risk_pct:.1f}%) - Consider reducing size")
            elif risk_pct > 1.5:
                notes.append(f"📊 Standard position risk ({risk_pct:.1f}%)")
            else:
                notes.append(f"📊 Conservative position risk ({risk_pct:.1f}%)")
            
            # Add method note
            method = stop_loss_info.get('method', 'unknown')
            notes.append(f"📋 Stop loss method: {method.replace('_', ' ').title()}")
            
            return notes
            
        except Exception as e:
            logger.error(f"Error generating risk notes: {e}")
            return ["⚠️ Risk analysis partially unavailable"]
    
    def calculate_atr_position_size(self, entry_price: float, stop_loss: float, 
                                  data: pd.DataFrame, atr_multiplier: float = 2.0) -> Dict[str, Any]:
        """
        Convenience method for ATR-based position sizing.
        """
        return self.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            method='atr',
            data=data,
            atr_multiplier=atr_multiplier
        )
    
    def calculate_kelly_position_size(self, entry_price: float, stop_loss: float,
                                    win_rate: float = 0.55, avg_win_loss_ratio: float = 1.5) -> Dict[str, Any]:
        """
        Convenience method for Kelly Criterion position sizing.
        """
        return self.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            method='kelly',
            win_rate=win_rate,
            avg_win_loss_ratio=avg_win_loss_ratio
        )
    
    def calculate_volatility_position_size(self, entry_price: float, data: pd.DataFrame,
                                         volatility_target: float = 0.20) -> Dict[str, Any]:
        """
        Convenience method for volatility-based position sizing.
        """
        return self.calculate_position_size(
            entry_price=entry_price,
            stop_loss=None,  # Not needed for volatility method
            method='percent_volatility',
            data=data,
            volatility_target=volatility_target
        )
    
    def calculate_market_condition_position_size(self, entry_price: float, stop_loss: float,
                                               data: pd.DataFrame, market_condition: str = 'normal') -> Dict[str, Any]:
        """
        Convenience method for market condition adjusted position sizing.
        """
        return self.calculate_position_size(
            entry_price=entry_price,
            stop_loss=stop_loss,
            method='market_condition',
            data=data,
            market_condition=market_condition
        )
    
    def get_optimal_position_size(self, entry_price: float, stop_loss: float,
                                data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get optimal position size by comparing multiple methods.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            data: Historical price data
            **kwargs: Additional parameters for specific methods
            
        Returns:
            Dictionary with optimal sizing recommendation and comparison of methods
        """
        try:
            methods = ['fixed_risk', 'atr', 'kelly', 'market_condition']
            results = {}
            
            # Test each method
            for method in methods:
                try:
                    if method == 'kelly':
                        result = self.calculate_position_size(
                            entry_price, stop_loss, method=method,
                            win_rate=kwargs.get('win_rate', 0.55),
                            avg_win_loss_ratio=kwargs.get('avg_win_loss_ratio', 1.5)
                        )
                    elif method == 'market_condition':
                        result = self.calculate_position_size(
                            entry_price, stop_loss, method=method, data=data,
                            market_condition=kwargs.get('market_condition', 'normal')
                        )
                    else:
                        result = self.calculate_position_size(
                            entry_price, stop_loss, method=method, data=data
                        )
                    
                    if 'error' not in result:
                        results[method] = result
                        
                except Exception as e:
                    logger.warning(f"Error with {method} position sizing: {e}")
                    continue
            
            if not results:
                # Fallback to basic method
                return self._calculate_basic_position_size(entry_price, stop_loss, self.max_risk_per_trade)
            
            # Choose the most conservative approach (smallest position size)
            optimal_method = min(results.keys(), key=lambda x: results[x]['position_size'])
            optimal_result = results[optimal_method]
            
            # Add comparison data
            optimal_result['method_comparison'] = {
                method: {
                    'position_size': results[method]['position_size'],
                    'risk_amount': results[method]['risk_amount']
                }
                for method in results
            }
            optimal_result['recommended_method'] = optimal_method
            optimal_result['methods_tested'] = list(results.keys())
            
            return optimal_result
            
        except Exception as e:
            logger.error(f"Error in optimal position sizing: {e}")
            return self._calculate_basic_position_size(entry_price, stop_loss, self.max_risk_per_trade)
    
    def update_account_settings(self, account_balance: Optional[float] = None,
                              max_risk_per_trade: Optional[float] = None,
                              max_total_risk: Optional[float] = None,
                              max_drawdown: Optional[float] = None):
        """
        Update risk management settings.
        """
        if account_balance is not None:
            self.account_balance = account_balance
            self.position_sizer.update_account_balance(account_balance)
        
        if max_risk_per_trade is not None:
            self.max_risk_per_trade = max_risk_per_trade
            self.position_sizer.update_risk_per_trade(max_risk_per_trade)
        
        if max_total_risk is not None:
            self.max_total_risk = max_total_risk
            
        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
            
        logger.info(f"Updated risk settings - Balance: ${self.account_balance:,.2f}, "
                   f"Risk per trade: {self.max_risk_per_trade*100:.1f}%")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current risk management settings.
        
        Returns:
            Dictionary with risk management summary
        """
        return {
            'account_balance': self.account_balance,
            'max_risk_per_trade': self.max_risk_per_trade * 100,
            'max_total_risk': self.max_total_risk * 100,
            'max_drawdown': self.max_drawdown * 100,
            'open_positions': len(self.open_positions),
            'risk_management_active': True,
            'position_sizer_methods': ['fixed_risk', 'atr', 'kelly', 'percent_volatility', 'market_condition']
        }
