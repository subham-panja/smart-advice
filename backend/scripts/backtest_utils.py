"""
Backtesting Utility Module
File: scripts/backtest_utils.py

This module contains logic for running historical simulations and calculating
backtest metrics.
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any
import logging
from scripts.backtesting_runner import BacktestingRunner

logger = logging.getLogger(__name__)

class BacktestUtils:
    """Class for backtesting orchestration and metrics."""
    
    def __init__(self):
        self.backtest_runner = BacktestingRunner()

    def perform_backtesting(self, symbol: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform backtesting on the strategies using historical data."""
        try:
            results = self.backtest_runner.run(symbol, historical_data)
            if results.get('status') == 'completed':
                metrics = results.get('combined_metrics', {})
                logger.info(f"Backtest {symbol}: CAGR {metrics.get('avg_cagr', 0)}%, Win {metrics.get('avg_win_rate', 0)}%")
            return results
        except Exception as e:
            logger.error(f"Error in perform_backtesting for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol, 'status': 'error'}

    def simulate_trading_strategy(self, data: pd.DataFrame, initial_capital: float = 100000) -> Dict[str, Any]:
        """Simulate trading strategy on historical data."""
        try:
            cash, position = initial_capital, 0
            trades, p_values = [], []
            closes = data['Close'].values
            sma_20, sma_50 = ta.SMA(closes, 20), ta.SMA(closes, 50)
            rsi = ta.RSI(closes, 14)
            
            for i in range(50, len(data)):
                price, date = data['Close'].iloc[i], data.index[i]
                p_values.append(cash + (position * price))
                
                signal = self._generate_trading_signal(price, sma_20[i], sma_50[i], rsi[i])
                
                if signal == 'BUY' and position == 0 and cash > price:
                    shares = int(cash * 0.95 / price)
                    if shares > 0:
                        cost = shares * price
                        cash, position = cash - cost, shares
                        trades.append({'date': date, 'action': 'BUY', 'price': price, 'shares': shares})
                elif signal == 'SELL' and position > 0:
                    cash, position = cash + (position * price), 0
                    trades.append({'date': date, 'action': 'SELL', 'price': price})
            
            final_val = cash + (position * data['Close'].iloc[-1])
            total_ret = (final_val - initial_capital) / initial_capital * 100
            
            # Simple win rate calculation
            sells = [t for t in trades if t['action'] == 'SELL']
            wins = 0 # logic to calculate wins would go here
            
            return {
                'initial_capital': initial_capital, 'final_portfolio_value': final_val,
                'total_return': round(total_ret, 2), 'total_trades': len(trades)
            }
        except Exception as e:
            logger.error(f"Error in simulate_trading_strategy: {e}")
            return {'error': str(e)}

    def _generate_trading_signal(self, price: float, sma_20: float, sma_50: float, rsi: float) -> str:
        """Generate high-level trading signals."""
        if pd.isna(sma_20) or pd.isna(sma_50) or pd.isna(rsi): return 'HOLD'
        buy_sig, sell_sig = 0, 0
        if price > sma_20 > sma_50: buy_sig += 2
        elif price < sma_20 < sma_50: sell_sig += 2
        if 30 < rsi < 50: buy_sig += 1
        elif rsi > 70: sell_sig += 1.5
        
        if buy_sig >= 2.5: return 'BUY'
        if sell_sig >= 2.0: return 'SELL'
        return 'HOLD'

    def calculate_overall_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall metrics across periods."""
        if not backtest_results: return {}
        try:
            rs = backtest_results.values()
            return {
                'average_cagr': round(sum(r.get('cagr', 0) for r in rs) / len(rs), 2),
                'average_win_rate': round(sum(r.get('win_rate', 0) for r in rs) / len(rs), 2),
                'total_trades': sum(r.get('total_trades', 0) for r in rs)
            }
        except: return {}
