"""
Advanced Position Sizing Module
File: scripts/position_sizing.py

This module implements sophisticated position sizing strategies for optimal risk management:
- Volatility-adjusted position sizing using ATR
- Kelly Criterion position sizing
- Fixed fractional position sizing
- Volatility targeting position sizing
- Dynamic position sizing based on market conditions
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any, Optional, Tuple
from utils.logger import setup_logging

logger = setup_logging()
class PositionSizer:
    """
    Advanced position sizing system for professional trading.
    
    This class implements multiple position sizing methods to optimize risk-adjusted returns
    while maintaining proper capital preservation.
    """
    
    def __init__(self, account_balance: float = 100000.0, base_risk_per_trade: float = 0.02, volatility_factor_enabled: bool = False):
        """
        Initialize the position sizer.
        
        Args:
            account_balance: Total account balance
            base_risk_per_trade: Base risk percentage per trade (default 2%)
            volatility_factor_enabled: Whether to enable volatility-based risk adjustments
        """
        self.account_balance = account_balance
        self.base_risk_per_trade = base_risk_per_trade
        self.volatility_factor_enabled = volatility_factor_enabled
        
    def volatility_adjusted_sizing(self, data: pd.DataFrame, entry_price: float, 
                                 stop_loss: float, target_volatility: float = 0.15,
                                 risk_adjustment_factor: float = 1.0) -> Dict[str, Any]:
        """
        Calculate position size based on volatility targeting.
        
        This method adjusts position size to maintain consistent portfolio volatility
        regardless of individual stock volatility.
        
        Args:
            data: Historical price data
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            target_volatility: Target portfolio volatility (default 15% annually)
            
        Returns:
            Dictionary containing position sizing information
        """
        try:
            if len(data) < 20:
                return self._default_sizing(entry_price, stop_loss)
            
            # Calculate stock's volatility using ATR
            atr_values = ta.ATR(data['High'].values, data['Low'].values, 
                              data['Close'].values, timeperiod=14)
            current_atr = atr_values[-1] if not pd.isna(atr_values[-1]) else entry_price * 0.02
            
            # Calculate annualized volatility
            stock_volatility = (current_atr / entry_price) * np.sqrt(252)  # Daily to annual
            
            if stock_volatility <= 0:
                return self._default_sizing(entry_price, stop_loss)
            
            # Calculate volatility adjustment factor
            volatility_adjustment = target_volatility / stock_volatility
            volatility_adjustment = max(0.25, min(4.0, volatility_adjustment))  # Limit to 0.25x - 4x
            
            # Adjust base risk by volatility
            adjusted_risk = self.base_risk_per_trade * volatility_adjustment
            adjusted_risk = max(0.005, min(0.05, adjusted_risk))  # Keep between 0.5% - 5%
            
            # Calculate position size
            risk_amount = self.account_balance * adjusted_risk
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                return self._default_sizing(entry_price, stop_loss)
            
            position_size = int(risk_amount / risk_per_share)
            position_value = position_size * entry_price
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'risk_amount': position_size * risk_per_share,
                'risk_percentage': (position_size * risk_per_share / self.account_balance) * 100,
                'method': 'volatility_adjusted',
                'stock_volatility': stock_volatility,
                'target_volatility': target_volatility,
                'volatility_adjustment': volatility_adjustment,
                'adjusted_risk': adjusted_risk,
                'atr_value': current_atr
            }
            
        except Exception as e:
            logger.error(f"Error in volatility adjusted sizing: {e}")
            return self._default_sizing(entry_price, stop_loss)
    
    def kelly_criterion_sizing(self, win_rate: float, avg_win: float, avg_loss: float,
                              entry_price: float, stop_loss: float) -> Dict[str, Any]:
        """
        Calculate position size using Kelly Criterion.
        
        The Kelly Criterion optimizes position size to maximize long-term growth.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade percentage
            avg_loss: Average losing trade percentage (positive value)
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            
        Returns:
            Dictionary containing Kelly-based position sizing
        """
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_win <= 0 or avg_loss <= 0:
                return self._default_sizing(entry_price, stop_loss)
            
            # Kelly percentage = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            
            # Apply Kelly with safety factor (typically 25-50% of full Kelly)
            safe_kelly = max(0, kelly_pct * 0.25)  # Use 25% of Kelly for safety
            safe_kelly = min(safe_kelly, 0.10)  # Cap at 10% of account
            
            # Calculate position size
            position_value = self.account_balance * safe_kelly
            position_size = int(position_value / entry_price)
            
            # Calculate actual risk
            risk_per_share = abs(entry_price - stop_loss)
            actual_risk = position_size * risk_per_share
            
            return {
                'position_size': position_size,
                'position_value': position_size * entry_price,
                'risk_amount': actual_risk,
                'risk_percentage': (actual_risk / self.account_balance) * 100,
                'method': 'kelly_criterion',
                'full_kelly': kelly_pct,
                'safe_kelly': safe_kelly,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            }
            
        except Exception as e:
            logger.error(f"Error in Kelly criterion sizing: {e}")
            return self._default_sizing(entry_price, stop_loss)
    
    def fixed_fractional_sizing(self, entry_price: float, stop_loss: float, 
                               risk_fraction: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate position size using fixed fractional method.
        
        This is the most common position sizing method, risking a fixed percentage
        of account balance on each trade.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_fraction: Risk fraction to use (uses base if None)
            
        Returns:
            Dictionary containing fixed fractional position sizing
        """
        try:
            risk_pct = risk_fraction or self.base_risk_per_trade
            risk_amount = self.account_balance * risk_pct
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                return {
                    'position_size': 0,
                    'error': 'Invalid stop loss - must be different from entry price',
                    'method': 'fixed_fractional'
                }
            
            position_size = int(risk_amount / risk_per_share)
            position_value = position_size * entry_price
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'risk_amount': position_size * risk_per_share,
                'risk_percentage': risk_pct * 100,
                'method': 'fixed_fractional',
                'risk_per_share': risk_per_share
            }
            
        except Exception as e:
            logger.error(f"Error in fixed fractional sizing: {e}")
            return self._default_sizing(entry_price, stop_loss)
    
    def percent_volatility_sizing(self, data: pd.DataFrame, entry_price: float,
                                target_risk_pct: float = 1.0) -> Dict[str, Any]:
        """
        Size position based on percentage volatility method.
        
        This method sizes positions to risk a fixed percentage based on
        the stock's volatility rather than a fixed stop loss.
        
        Args:
            data: Historical price data
            entry_price: Entry price for the trade
            target_risk_pct: Target risk percentage (default 1%)
            
        Returns:
            Dictionary containing percent volatility position sizing
        """
        try:
            if len(data) < 20:
                # Use 2% default volatility if insufficient data
                daily_volatility = 0.02
            else:
                # Calculate daily returns volatility
                returns = data['Close'].pct_change().dropna()
                daily_volatility = returns.tail(20).std()
            
            if daily_volatility <= 0:
                daily_volatility = 0.02  # Default 2%
            
            # Risk amount based on target risk percentage and volatility
            risk_amount = self.account_balance * (target_risk_pct / 100)
            
            # Position size based on volatility risk
            # Risk per share = entry_price * daily_volatility * volatility_multiplier
            volatility_multiplier = 2.0  # 2 standard deviations
            risk_per_share = entry_price * daily_volatility * volatility_multiplier
            
            position_size = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
            position_value = position_size * entry_price
            
            # Calculate implied stop loss
            implied_stop = entry_price - risk_per_share
            
            return {
                'position_size': position_size,
                'position_value': position_value,
                'risk_amount': position_size * risk_per_share,
                'risk_percentage': (position_size * risk_per_share / self.account_balance) * 100,
                'method': 'percent_volatility',
                'daily_volatility': daily_volatility,
                'volatility_multiplier': volatility_multiplier,
                'risk_per_share': risk_per_share,
                'implied_stop_loss': implied_stop
            }
            
        except Exception as e:
            logger.error(f"Error in percent volatility sizing: {e}")
            return self._default_sizing(entry_price, entry_price * 0.95)
    
    def market_condition_sizing(self, data: pd.DataFrame, entry_price: float, stop_loss: float,
                              market_regime: str = 'NEUTRAL') -> Dict[str, Any]:
        """
        Adjust position size based on market conditions.
        
        This method modifies the base position size based on overall market regime
        and volatility environment.
        
        Args:
            data: Historical price data
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            market_regime: Market regime ('BULL', 'BEAR', 'NEUTRAL', 'VOLATILE')
            
        Returns:
            Dictionary containing market-adjusted position sizing
        """
        try:
            # Start with base fixed fractional sizing
            base_sizing = self.fixed_fractional_sizing(entry_price, stop_loss)
            
            if 'error' in base_sizing:
                return base_sizing
            
            # Market condition adjustments
            market_adjustments = {
                'BULL': 1.2,      # Increase size in bull markets
                'BEAR': 0.7,      # Reduce size in bear markets
                'NEUTRAL': 1.0,   # No adjustment
                'VOLATILE': 0.6   # Significantly reduce in volatile markets
            }
            
            adjustment_factor = market_adjustments.get(market_regime, 1.0)
            
            # Calculate market volatility adjustment
            if len(data) >= 20:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.tail(20).std() * np.sqrt(252)  # Annualized
                
                # Additional volatility adjustment
                if volatility > 0.4:  # Very high volatility (>40% annual)
                    volatility_adj = 0.6
                elif volatility > 0.25:  # High volatility (>25% annual)
                    volatility_adj = 0.8
                elif volatility < 0.15:  # Low volatility (<15% annual)
                    volatility_adj = 1.2
                else:
                    volatility_adj = 1.0
                
                adjustment_factor *= volatility_adj
            
            # Apply adjustments
            adjusted_size = int(base_sizing['position_size'] * adjustment_factor)
            adjusted_value = adjusted_size * entry_price
            adjusted_risk = adjusted_size * abs(entry_price - stop_loss)
            
            return {
                'position_size': adjusted_size,
                'position_value': adjusted_value,
                'risk_amount': adjusted_risk,
                'risk_percentage': (adjusted_risk / self.account_balance) * 100,
                'method': 'market_condition_adjusted',
                'market_regime': market_regime,
                'adjustment_factor': adjustment_factor,
                'base_position_size': base_sizing['position_size'],
                'volatility': returns.tail(20).std() * np.sqrt(252) if len(data) >= 20 else None
            }
            
        except Exception as e:
            logger.error(f"Error in market condition sizing: {e}")
            return self._default_sizing(entry_price, stop_loss)
    
    def optimal_sizing_recommendation(self, data: pd.DataFrame, entry_price: float, 
                                    stop_loss: float, strategy_win_rate: Optional[float] = None,
                                    avg_win: Optional[float] = None, avg_loss: Optional[float] = None,
                                    market_regime: str = 'NEUTRAL') -> Dict[str, Any]:
        """
        Provide optimal position sizing recommendation based on multiple methods.
        
        This method evaluates different sizing approaches and recommends the most
        appropriate one based on available data and market conditions.
        
        Args:
            data: Historical price data
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            strategy_win_rate: Historical win rate for the strategy
            avg_win: Average winning trade percentage
            avg_loss: Average losing trade percentage
            market_regime: Current market regime
            
        Returns:
            Dictionary containing optimal sizing recommendation
        """
        try:
            sizing_methods = {}
            
            # 1. Fixed Fractional (baseline)
            sizing_methods['fixed_fractional'] = self.fixed_fractional_sizing(entry_price, stop_loss)
            
            # 2. Volatility Adjusted
            sizing_methods['volatility_adjusted'] = self.volatility_adjusted_sizing(
                data, entry_price, stop_loss
            )
            
            # 3. Market Condition Adjusted
            sizing_methods['market_adjusted'] = self.market_condition_sizing(
                data, entry_price, stop_loss, market_regime
            )
            
            # 4. Kelly Criterion (if we have performance data)
            if all(x is not None for x in [strategy_win_rate, avg_win, avg_loss]):
                sizing_methods['kelly'] = self.kelly_criterion_sizing(
                    strategy_win_rate, avg_win, avg_loss, entry_price, stop_loss
                )
            
            # 5. Percent Volatility
            sizing_methods['percent_volatility'] = self.percent_volatility_sizing(data, entry_price)
            
            # Evaluate and rank methods
            method_scores = self._score_sizing_methods(sizing_methods, data, market_regime)
            
            # Select best method
            best_method = max(method_scores.items(), key=lambda x: x[1]['score'])
            recommended_method = best_method[0]
            recommended_sizing = sizing_methods[recommended_method]
            
            return {
                'recommended_method': recommended_method,
                'recommended_sizing': recommended_sizing,
                'all_methods': sizing_methods,
                'method_scores': method_scores,
                'selection_reason': best_method[1]['reason']
            }
            
        except Exception as e:
            logger.error(f"Error in optimal sizing recommendation: {e}")
            return {
                'recommended_method': 'fixed_fractional',
                'recommended_sizing': self._default_sizing(entry_price, stop_loss),
                'error': str(e)
            }
    
    def _score_sizing_methods(self, methods: Dict[str, Dict], data: pd.DataFrame, 
                            market_regime: str) -> Dict[str, Dict]:
        """
        Score different sizing methods based on appropriateness for current conditions.
        
        Args:
            methods: Dictionary of sizing methods and their results
            data: Historical price data
            market_regime: Current market regime
            
        Returns:
            Dictionary with scores and reasons for each method
        """
        scores = {}
        
        try:
            # Calculate market characteristics
            has_sufficient_data = len(data) >= 50
            
            if has_sufficient_data:
                returns = data['Close'].pct_change().dropna()
                volatility = returns.tail(20).std() * np.sqrt(252)
                high_volatility = volatility > 0.3
            else:
                high_volatility = False
            
            for method_name, method_result in methods.items():
                if 'error' in method_result:
                    scores[method_name] = {'score': 0, 'reason': f"Error in {method_name}"}
                    continue
                
                score = 50  # Base score
                reasons = []
                
                # Scoring logic
                if method_name == 'volatility_adjusted':
                    if has_sufficient_data:
                        score += 20
                        reasons.append("Good data availability")
                    if high_volatility:
                        score += 15
                        reasons.append("High volatility environment")
                
                elif method_name == 'market_adjusted':
                    if market_regime != 'NEUTRAL':
                        score += 15
                        reasons.append(f"Clear market regime: {market_regime}")
                    if has_sufficient_data:
                        score += 10
                        reasons.append("Good data for market analysis")
                
                elif method_name == 'kelly':
                    score += 25  # Kelly is theoretically optimal
                    reasons.append("Theoretically optimal for long-term growth")
                
                elif method_name == 'fixed_fractional':
                    score += 10  # Always reliable baseline
                    reasons.append("Reliable baseline method")
                
                elif method_name == 'percent_volatility':
                    if not has_sufficient_data:
                        score += 10
                        reasons.append("Good for limited data")
                
                # Risk-based adjustments
                risk_pct = method_result.get('risk_percentage', 0)
                if 0.5 <= risk_pct <= 3.0:  # Reasonable risk range
                    score += 10
                    reasons.append("Reasonable risk level")
                elif risk_pct > 5.0:  # Too risky
                    score -= 20
                    reasons.append("Risk level too high")
                
                scores[method_name] = {
                    'score': score,
                    'reason': '; '.join(reasons)
                }
            
            return scores
            
        except Exception as e:
            logger.error(f"Error scoring sizing methods: {e}")
            return {method: {'score': 50, 'reason': 'Default scoring'} for method in methods.keys()}
    
    def _default_sizing(self, entry_price: float, stop_loss: float) -> Dict[str, Any]:
        """
        Fallback default sizing method.
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            
        Returns:
            Dictionary containing default position sizing
        """
        try:
            risk_amount = self.account_balance * self.base_risk_per_trade
            risk_per_share = abs(entry_price - stop_loss)
            
            if risk_per_share <= 0:
                risk_per_share = entry_price * 0.05  # Default 5% risk per share
            
            position_size = int(risk_amount / risk_per_share)
            
            return {
                'position_size': position_size,
                'position_value': position_size * entry_price,
                'risk_amount': position_size * risk_per_share,
                'risk_percentage': (position_size * risk_per_share / self.account_balance) * 100,
                'method': 'default_fallback'
            }
            
        except Exception as e:
            logger.error(f"Error in default sizing: {e}")
            return {
                'position_size': 0,
                'position_value': 0,
                'risk_amount': 0,
                'risk_percentage': 0,
                'method': 'error_fallback',
                'error': str(e)
            }
    
    def update_account_balance(self, new_balance: float):
        """Update account balance for position sizing calculations."""
        self.account_balance = new_balance
        logger.info(f"Account balance updated to ${new_balance:,.2f}")
    
    def update_risk_per_trade(self, new_risk: float):
        """Update base risk per trade for position sizing calculations."""
        self.base_risk_per_trade = new_risk
        logger.info(f"Base risk per trade updated to {new_risk*100:.2f}%")
    
    def get_sizing_summary(self) -> Dict[str, Any]:
        """
        Get summary of position sizer configuration.
        
        Returns:
            Dictionary with position sizer summary
        """
        return {
            'account_balance': self.account_balance,
            'base_risk_per_trade': self.base_risk_per_trade * 100,
            'available_methods': [
                'volatility_adjusted',
                'kelly_criterion', 
                'fixed_fractional',
                'percent_volatility',
                'market_condition_adjusted'
            ],
            'recommended_approach': 'Use optimal_sizing_recommendation() for best results'
        }
