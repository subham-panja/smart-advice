import pandas as pd
import talib as ta
import logging
from typing import Dict, Any
from config import SWING_TRADING_GATES

logger = logging.getLogger(__name__)

class SwingTradingSignalAnalyzer:
    """Analyzes stocks for swing trading using Trend, Volatility, and Volume gates."""
    
    def __init__(self):
        t = SWING_TRADING_GATES.get('TREND_GATE', {}).get('params', {})
        self.adx_min = t.get('adx_min', 20)
        self.sma_period = t.get('sma_period', 200)

    def analyze_swing_opportunity(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        if len(df) < self.sma_period: return {'symbol': symbol, 'all_gates_passed': False}
        
        c = df['Close'].iloc[-1]
        
        # Gates
        adx = ta.ADX(df['High'], df['Low'], df['Close'], 14).iloc[-1]
        pdi, mdi = ta.PLUS_DI(df['High'], df['Low'], df['Close'], 14).iloc[-1], ta.MINUS_DI(df['High'], df['Low'], df['Close'], 14).iloc[-1]
        sma = ta.SMA(df['Close'], self.sma_period).iloc[-1]
        
        trend_ok = adx > self.adx_min and c > sma and pdi > mdi
        
        v_mean = df['Volume'].tail(20).mean()
        vol_ok = df['Volume'].iloc[-1] > v_mean * 1.2
        
        atr = ta.ATR(df['High'], df['Low'], df['Close'], 14)
        v_ok = 20 <= (atr.iloc[-100:] < atr.iloc[-1]).sum() <= 80 # Volatility percentile

        gates = {'trend': trend_ok, 'volume': vol_ok, 'volatility': v_ok}
        all_ok = all(gates.values())
        
        res = {'symbol': symbol, 'all_gates_passed': all_ok, 'gates': gates}
        if all_ok:
            ema20 = ta.EMA(df['Close'], 20).iloc[-1]
            if df['Low'].iloc[-1] <= ema20 * 1.02 and c > ema20:
                res.update({'recommendation': 'BUY', 'pattern': 'EMA_Pullback'})
        
        return res
