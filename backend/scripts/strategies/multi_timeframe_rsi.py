import pandas as pd
import talib as ta
from scripts.strategies.base_strategy import BaseStrategy, TechnicalIndicatorMixin

# Initialize logger only when class is instantiated to avoid import hangs
logger = None

class MultiTimeframeRSI(BaseStrategy, TechnicalIndicatorMixin):
    """
    Multi-Timeframe RSI Confluence Strategy
    
    This strategy checks RSI across multiple timeframes (daily, weekly) to find high-probability setups.
    - Daily RSI for entry timing
    - Weekly RSI for trend confirmation
    """
    
    def __init__(self, params: dict = None):
        super().__init__(params)
        # Initialize logger when strategy is instantiated
        global logger
        if logger is None:
            from utils.logger import setup_logging
            logger = setup_logging()
        
        self.params = params or {
            'daily_rsi_period': 14,
            'weekly_rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'weekly_rsi_bullish': 50,
            'weekly_rsi_bearish': 50
        }
    
    def _resample_to_weekly(self, data: pd.DataFrame) -> pd.DataFrame:
        """Resample daily data to weekly data."""
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        weekly_data = data.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return weekly_data

    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the multi-timeframe RSI strategy.
        
        Args:
            data: Daily OHLCV data
            
        Returns:
            1 for buy signal, -1 for sell signal, 0 for no signal
        """
        try:
            # Resample to weekly data
            weekly_data = self._resample_to_weekly(data.copy())
            
            if len(data) < self.params['daily_rsi_period'] or len(weekly_data) < self.params['weekly_rsi_period']:
                return 0

            # Calculate daily and weekly RSI
            daily_rsi = self.calculate_rsi(data['Close'], self.params['daily_rsi_period']).iloc[-1]
            weekly_rsi = self.calculate_rsi(weekly_data['Close'], self.params['weekly_rsi_period']).iloc[-1]

            # Bullish confluence
            daily_oversold = daily_rsi < self.params['rsi_oversold']
            weekly_uptrend = weekly_rsi > self.params['weekly_rsi_bullish']
            
            if daily_oversold and weekly_uptrend:
                logger.info(f"{self.name}: BUY signal - Daily RSI ({daily_rsi:.2f}) oversold in weekly uptrend (RSI {weekly_rsi:.2f})")
                return 1

            # Bearish confluence (for selling or avoiding buys)
            daily_overbought = daily_rsi > self.params['rsi_overbought']
            weekly_downtrend = weekly_rsi < self.params['weekly_rsi_bearish']
            
            if daily_overbought and weekly_downtrend:
                logger.info(f"{self.name}: SELL signal - Daily RSI ({daily_rsi:.2f}) overbought in weekly downtrend (RSI {weekly_rsi:.2f})")
                return -1

            return 0

        except Exception as e:
            logger.error(f"{self.name}: Error executing strategy: {e}")
            return 0

