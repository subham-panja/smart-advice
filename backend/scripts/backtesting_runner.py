import pandas as pd
import numpy as np
import importlib
import logging
from typing import List, Dict, Any, Optional
from scripts.backtesting import BacktestingEngine
from scripts.strategies.base_strategy import BacktraderStrategy

logger = logging.getLogger(__name__)

class BacktestingRunner:
    """Evaluates multiple strategies and calculates CAGR, Win Rate, Expectancy, and Profit Factor."""
    
    def __init__(self, initial_cash: float = 100000.0, commission: float = 0.001):
        self.initial_cash = initial_cash
        self.commission = commission
        
    def run(self, symbol: str, df: pd.DataFrame, strategy_classes: Optional[List[str]] = None) -> Dict[str, Any]:
        if len(df) < 60: return {'symbol': symbol, 'status': 'insufficient_data'}
        
        if strategy_classes is None:
            strategy_classes = ['MA_Crossover_50_200', 'RSI_Overbought_Oversold', 'MACD_Signal_Crossover', 'Bollinger_Band_Breakout']

        results = {}
        for name in strategy_classes:
            try:
                engine = BacktestingEngine(self.initial_cash, self.commission)
                bt = engine.run_backtest(self._create_strategy(name), df)
                results[name] = self._calc_metrics(bt, df)
            except Exception as e:
                logger.error(f"Backtest {name} error: {e}")

        combined = self._combine(results)
        return {'symbol': symbol, 'status': 'completed', 'strategy_results': results, 'combined_metrics': combined}

    def _create_strategy(self, name: str):
        mapping = {
            'MA_Crossover_50_200': 'scripts.strategies.ma_crossover_50_200',
            'RSI_Overbought_Oversold': 'scripts.strategies.rsi_overbought_oversold',
            'MACD_Signal_Crossover': 'scripts.strategies.macd_signal_crossover',
            'Bollinger_Band_Breakout': 'scripts.strategies.bollinger_band_breakout'
        }
        class BTStrategy(BacktraderStrategy):
            def _execute_strategy_logic(self, data):
                mod = importlib.import_module(mapping[name])
                return getattr(mod, name)()._execute_strategy_logic(data)
        return BTStrategy

    def _calc_metrics(self, bt: dict, df: pd.DataFrame) -> dict:
        t = bt.get('trade_analysis', {})
        won, lost, total = t.get('won', {}).get('total', 0), t.get('lost', {}).get('total', 0), t.get('total', {}).get('total', 0)
        p_won, p_lost = t.get('won', {}).get('pnl', {}).get('total', 0), abs(t.get('lost', {}).get('pnl', {}).get('total', 0))
        
        wr = (won / total * 100) if total > 0 else 0
        pf = (p_won / p_lost) if p_lost > 0 else (999.0 if p_won > 0 else 0)
        exp = ((wr/100 * (p_won/won if won > 0 else 0)) - ((1-wr/100) * (p_lost/lost if lost > 0 else 0))) if total > 0 else 0
        
        years = len(df) / 365.25
        cagr = ((bt['final_portfolio_value'] / bt['initial_cash']) ** (1/years) - 1) * 100 if years > 0 else 0
        
        return {'cagr': round(cagr, 2), 'win_rate': round(wr, 2), 'expectancy': round(exp, 2), 'profit_factor': round(min(pf, 999.0), 2), 'total_trades': total}

    def _combine(self, results: dict) -> dict:
        res = [r for r in results.values()]
        if not res: return {}
        return {
            'avg_cagr': round(sum(r['cagr'] for r in res) / len(res), 2),
            'avg_win_rate': round(sum(r['win_rate'] for r in res) / len(res), 2),
            'avg_expectancy': round(sum(r['expectancy'] for r in res) / len(res), 2),
            'avg_profit_factor': round(sum(r['profit_factor'] for r in res) / len(res), 2)
        }
