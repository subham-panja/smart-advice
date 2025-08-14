"""
Backtest Engine Module
=====================

Core backtesting functionality for testing trading strategies.
Extracted from analyzer.py for better organization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.logger import setup_logging

logger = setup_logging()

class BacktestEngine:
    """
    Core engine for backtesting trading strategies.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.reset()
    
    def reset(self):
        """Reset the backtest state."""
        self.cash = self.initial_capital
        self.position = 0
        self.trades = []
        self.portfolio_values = []
        self._data_aligned = None  # store last aligned OHLCV for per-trade analysis
    
    def run_backtest(self, data: pd.DataFrame, strategy_signals: pd.Series) -> Dict[str, Any]:
        """
        Run backtest on historical data with strategy signals.
        
        Args:
            data: Historical OHLCV data
            strategy_signals: Series with BUY/SELL/HOLD signals
            
        Returns:
            Dictionary containing backtest results
        """
        try:
            if data.empty or strategy_signals.empty:
                return {'error': 'Empty data or signals'}
            
            logger.info("Starting backtest execution")
            self.reset()
            
            # Align data and signals
            aligned_data = data.align(strategy_signals, join='inner', axis=0)
            if aligned_data[0].empty:
                return {'error': 'No aligned data between prices and signals'}
            
            data_aligned, signals_aligned = aligned_data

            # Keep aligned data for metrics
            self._data_aligned = data_aligned[['Open','High','Low','Close']].copy()
            
            # Execute trades based on signals
            for i, (date, row) in enumerate(data_aligned.iterrows()):
                current_price = row['Close']
                signal = signals_aligned.iloc[i] if i < len(signals_aligned) else 'HOLD'
                
                # Calculate current portfolio value
                portfolio_value = self.cash + (self.position * current_price)
                self.portfolio_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'cash': self.cash,
                    'position': self.position,
                    'price': current_price
                })
                
                # Execute trades
                if signal == 'BUY' and self.position == 0 and self.cash > current_price:
                    # Enter long position
                    shares_to_buy = int(self.cash * 0.95 / current_price)  # Use 95% of cash
                    if shares_to_buy > 0:
                        cost = shares_to_buy * current_price
                        self.cash -= cost
                        self.position = shares_to_buy
                        
                        self.trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': current_price,
                            'shares': shares_to_buy,
                            'value': cost
                        })
                        
                elif signal == 'SELL' and self.position > 0:
                    # Exit long position
                    proceeds = self.position * current_price
                    self.cash += proceeds
                    
                    self.trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': current_price,
                        'shares': self.position,
                        'value': proceeds
                    })
                    
                    self.position = 0
            
            # Calculate final portfolio value
            final_price = data_aligned['Close'].iloc[-1]
            final_portfolio_value = self.cash + (self.position * final_price)
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(final_portfolio_value, data_aligned.index)
            results['trades'] = self.trades
            results['portfolio_history'] = self.portfolio_values
            
            logger.info("Backtest execution completed")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_metrics(self, final_value: float, date_range: pd.DatetimeIndex) -> Dict[str, Any]:
        """
        Calculate performance metrics from backtest results.
        
        Args:
            final_value: Final portfolio value
            date_range: Date range of the backtest
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            # Basic metrics
            total_return = (final_value - self.initial_capital) / self.initial_capital * 100
            
            # Calculate CAGR
            days_in_period = (date_range[-1] - date_range[0]).days
            years = days_in_period / 365.25
            cagr = ((final_value / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
            
            # Calculate win rate
            buy_sell_pairs = []
            buy_price = None
            
            # Build a quick map of trade dates for MAE/time-in-trade
            prices_df = self._data_aligned if isinstance(self._data_aligned, pd.DataFrame) else None

            for trade in self.trades:
                if trade['action'] == 'BUY':
                    buy_price = trade['price']
                    buy_date = trade['date']
                elif trade['action'] == 'SELL' and buy_price is not None:
                    sell_price = trade['price']
                    sell_date = trade['date']
                    profit_loss = sell_price - buy_price
                    ret_pct = (profit_loss / buy_price) * 100

                    # Compute time-in-trade and MAE using lows between buy and sell
                    time_in_trade = (sell_date - buy_date).days if isinstance(sell_date, pd.Timestamp) else 0
                    mae_pct = 0.0
                    if prices_df is not None and buy_date in prices_df.index and sell_date in prices_df.index:
                        window = prices_df.loc[buy_date:sell_date]
                        if not window.empty and 'Low' in window.columns:
                            min_price = float(window['Low'].min())
                            mae_pct = ((min_price - buy_price) / buy_price) * 100
                    
                    buy_sell_pairs.append({
                        'buy_date': buy_date,
                        'sell_date': sell_date,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'profit_loss': profit_loss,
                        'return_pct': ret_pct,
                        'time_in_trade_days': time_in_trade,
                        'mae_pct': mae_pct
                    })
                    buy_price = None
            
            total_trades = len(buy_sell_pairs)
            winning_trades = len([trade for trade in buy_sell_pairs if trade['profit_loss'] > 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate maximum drawdown
            max_drawdown = 0
            peak_value = self.initial_capital
            
            for portfolio_point in self.portfolio_values:
                value = portfolio_point['portfolio_value']
                if value > peak_value:
                    peak_value = value
                drawdown = (peak_value - value) / peak_value * 100
                max_drawdown = max(max_drawdown, drawdown)
            
            # Calculate Sharpe ratio (simplified)
            if buy_sell_pairs:
                returns = [trade['return_pct'] for trade in buy_sell_pairs]
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0

                # Additional swing metrics
                avg_time_in_trade = float(np.mean([t['time_in_trade_days'] for t in buy_sell_pairs])) if buy_sell_pairs else 0.0
                avg_mae = float(np.mean([t['mae_pct'] for t in buy_sell_pairs])) if buy_sell_pairs else 0.0
            else:
                sharpe_ratio = 0
                avg_time_in_trade = 0.0
                avg_mae = 0.0
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': final_value,
                'total_return': round(total_return, 2),
                'cagr': round(cagr, 2),
                'win_rate': round(win_rate, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe_ratio, 2),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'period_days': days_in_period,
                'avg_time_in_trade_days': round(avg_time_in_trade, 2),
                'avg_mae_pct': round(avg_mae, 2),
                'buy_sell_pairs': buy_sell_pairs
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {
                'error': str(e),
                'initial_capital': self.initial_capital,
                'final_capital': final_value,
                'total_return': 0,
                'cagr': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'total_trades': len(self.trades)
            }
    
    def generate_trading_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on simple moving average crossover.
        This is a placeholder strategy for demonstration.
        
        Args:
            data: Historical OHLCV data
            
        Returns:
            Series with trading signals
        """
        try:
            if data.empty:
                return pd.Series()
            
            # Simple moving average crossover strategy
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            
            signals = pd.Series(index=data.index, data='HOLD')
            
            # Generate signals
            for i in range(1, len(data)):
                if (data['SMA_20'].iloc[i] > data['SMA_50'].iloc[i] and 
                    data['SMA_20'].iloc[i-1] <= data['SMA_50'].iloc[i-1]):
                    signals.iloc[i] = 'BUY'
                elif (data['SMA_20'].iloc[i] < data['SMA_50'].iloc[i] and 
                      data['SMA_20'].iloc[i-1] >= data['SMA_50'].iloc[i-1]):
                    signals.iloc[i] = 'SELL'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return pd.Series()
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current backtest state.
        
        Returns:
            Dictionary containing backtest summary
        """
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'current_position': self.position,
            'total_trades': len(self.trades),
            'portfolio_points': len(self.portfolio_values)
        }
