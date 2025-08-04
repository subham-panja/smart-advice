"""
Backtest Metrics Module
=======================

Calculates various performance metrics for backtesting results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.logger import setup_logging

logger = setup_logging()

class BacktestMetrics:
    """
    Calculates performance metrics for backtesting results.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_returns_metrics(self, returns: List[float]) -> Dict[str, float]:
        """
        Calculate return-based metrics.
        
        Args:
            returns: List of returns
            
        Returns:
            Dictionary containing return metrics
        """
        try:
            if not returns:
                return {}
            
            returns_array = np.array(returns)
            
            metrics = {
                'total_return': np.sum(returns_array),
                'mean_return': np.mean(returns_array),
                'median_return': np.median(returns_array),
                'std_return': np.std(returns_array),
                'min_return': np.min(returns_array),
                'max_return': np.max(returns_array),
                'skewness': self._calculate_skewness(returns_array),
                'kurtosis': self._calculate_kurtosis(returns_array)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating return metrics: {e}")
            return {}
    
    def calculate_risk_metrics(self, portfolio_values: List[float], 
                              risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate risk-based metrics.
        
        Args:
            portfolio_values: List of portfolio values over time
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            Dictionary containing risk metrics
        """
        try:
            if len(portfolio_values) < 2:
                return {}
            
            # Calculate returns from portfolio values
            returns = []
            for i in range(1, len(portfolio_values)):
                ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(ret)
            
            if not returns:
                return {}
            
            returns_array = np.array(returns)
            
            # Calculate metrics
            volatility = np.std(returns_array) * np.sqrt(252)  # Annualized volatility
            mean_return = np.mean(returns_array) * 252  # Annualized return
            
            # Sharpe ratio
            sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            
            # Calmar ratio
            calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Value at Risk (VaR) at 95% confidence
            var_95 = np.percentile(returns_array, 5)
            
            # Conditional Value at Risk (CVaR)
            cvar_95 = np.mean(returns_array[returns_array <= var_95])
            
            metrics = {
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'var_95': var_95,
                'cvar_95': cvar_95,
                'downside_deviation': self._calculate_downside_deviation(returns_array)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate trade-based metrics.
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary containing trade metrics
        """
        try:
            if not trades:
                return {}
            
            # Separate buy and sell trades
            buy_trades = [t for t in trades if t.get('action') == 'BUY']
            sell_trades = [t for t in trades if t.get('action') == 'SELL']
            
            # Calculate trade pairs
            trade_pairs = []
            buy_idx = 0
            
            for sell_trade in sell_trades:
                if buy_idx < len(buy_trades):
                    buy_trade = buy_trades[buy_idx]
                    profit_loss = (sell_trade['price'] - buy_trade['price']) * sell_trade['shares']
                    return_pct = (sell_trade['price'] - buy_trade['price']) / buy_trade['price'] * 100
                    
                    trade_pairs.append({
                        'buy_date': buy_trade['date'],
                        'sell_date': sell_trade['date'],
                        'buy_price': buy_trade['price'],
                        'sell_price': sell_trade['price'],
                        'shares': sell_trade['shares'],
                        'profit_loss': profit_loss,
                        'return_pct': return_pct,
                        'holding_days': (sell_trade['date'] - buy_trade['date']).days
                    })
                    
                    buy_idx += 1
            
            if not trade_pairs:
                return {'total_trades': len(trades), 'complete_trades': 0}
            
            # Calculate metrics
            profits = [tp['profit_loss'] for tp in trade_pairs]
            returns = [tp['return_pct'] for tp in trade_pairs]
            holding_periods = [tp['holding_days'] for tp in trade_pairs]
            
            winning_trades = [p for p in profits if p > 0]
            losing_trades = [p for p in profits if p <= 0]
            
            metrics = {
                'total_trades': len(trades),
                'complete_trades': len(trade_pairs),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trade_pairs) * 100 if trade_pairs else 0,
                'average_win': np.mean(winning_trades) if winning_trades else 0,
                'average_loss': np.mean(losing_trades) if losing_trades else 0,
                'largest_win': max(profits) if profits else 0,
                'largest_loss': min(profits) if profits else 0,
                'profit_factor': sum(winning_trades) / abs(sum(losing_trades)) if losing_trades and sum(losing_trades) != 0 else float('inf'),
                'average_holding_period': np.mean(holding_periods) if holding_periods else 0,
                'trade_pairs': trade_pairs
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating trade metrics: {e}")
            return {}
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown from portfolio values."""
        try:
            if len(portfolio_values) < 2:
                return 0
            
            peak = portfolio_values[0]
            max_drawdown = 0
            
            for value in portfolio_values[1:]:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0
    
    def _calculate_downside_deviation(self, returns: np.ndarray, target_return: float = 0) -> float:
        """Calculate downside deviation."""
        try:
            downside_returns = returns[returns < target_return]
            if len(downside_returns) == 0:
                return 0
            
            downside_variance = np.mean((downside_returns - target_return) ** 2)
            return np.sqrt(downside_variance)
            
        except Exception as e:
            logger.error(f"Error calculating downside deviation: {e}")
            return 0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns."""
        try:
            if len(returns) < 3:
                return 0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0
            
            skewness = np.mean(((returns - mean_return) / std_return) ** 3)
            return skewness
            
        except Exception as e:
            logger.error(f"Error calculating skewness: {e}")
            return 0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns."""
        try:
            if len(returns) < 4:
                return 0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0
            
            kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
            return kurtosis
            
        except Exception as e:
            logger.error(f"Error calculating kurtosis: {e}")
            return 0
