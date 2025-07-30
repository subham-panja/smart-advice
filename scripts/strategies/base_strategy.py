"""
Base Strategy Class for Technical Analysis
File: scripts/strategies/base_strategy.py

This module provides the abstract base class for all trading strategies.
Each strategy should inherit from BaseStrategy and implement the run_strategy method.
"""

import backtrader as bt
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from utils.logger import setup_logging
from utils.enhanced_volume_confirmation import volume_confirmator
from utils.volume_analysis import get_enhanced_volume_confirmation

logger = setup_logging()

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class provides common functionality and enforces a consistent interface
    for all trading strategies in the system.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy with optional parameters.
        
        Args:
            params: Dictionary of strategy-specific parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def _execute_strategy_logic(self, data: pd.DataFrame) -> int:
        """
        Execute the core strategy logic without volume filtering.
        This method should be implemented by each strategy.
        
        Args:
            data: DataFrame with OHLCV data, indexed by date
                  Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        
        Returns:
            int: 1 for positive signal (buy), -1 for negative signal (sell/no buy)
        """
        pass
    
    def run_strategy(self, data: pd.DataFrame) -> int:
        """
        Execute the trading strategy with automatic volume filtering.
        This method calls the strategy logic and applies enhanced volume confirmation.
        
        Args:
            data: DataFrame with OHLCV data, indexed by date
                  Columns: ['Open', 'High', 'Low', 'Close', 'Volume']
        
        Returns:
            int: 1 for positive signal (buy), -1 for negative signal (sell/no buy)
        """
        try:
            # Execute the core strategy logic
            raw_signal = self._execute_strategy_logic(data)
            
            # Skip volume filtering for no signal or insufficient data
            if raw_signal == 0 or len(data) < 20:
                return raw_signal
            
            # Apply enhanced volume filtering using new system
            filtered_signal, filter_reason = volume_confirmator.filter_signal_by_volume(
                raw_signal, data, require_confirmation=True
            )
            
            # Log volume filtering result if signal was changed
            if filtered_signal != raw_signal:
                self.log_signal(filtered_signal, filter_reason, data)
            
            return filtered_signal
            
        except Exception as e:
            logger.error(f"{self.name}: Error in run_strategy: {e}")
            return -1
    
    def _get_volume_filtering_parameters(self) -> Dict[str, Any]:
        """
        Get volume filtering parameters based on strategy type.
        Different strategies require different volume confirmation thresholds.
        
        Returns:
            Dictionary with volume filtering parameters
        """
        # Default parameters
        params = {
            'min_volume_factor': 0.8,
            'breakout': False,
            'level': None
        }
        
        # Strategy-specific volume filtering parameters
        strategy_params = {
            # Breakout strategies need higher volume confirmation
            'Volume_Breakout': {'min_volume_factor': 1.5, 'breakout': True},
            'Bollinger_Band_Breakout': {'min_volume_factor': 1.3, 'breakout': True},
            'Support_Resistance_Breakout': {'min_volume_factor': 1.2, 'breakout': True},
            'Keltner_Channels_Breakout': {'min_volume_factor': 1.2, 'breakout': True},
            
            # Gap and channel strategies
            'Gap_Trading': {'min_volume_factor': 1.5, 'breakout': True},
            'Channel_Trading': {'min_volume_factor': 1.1},
            
            # Moving average crossovers (reduced volume requirements for MA strategies)
            'MA_Crossover_50_200': {'min_volume_factor': 0.6},
            'SMA_Crossover_20_50': {'min_volume_factor': 0.6},
            'EMA_Crossover_12_26': {'min_volume_factor': 1.1},
            'DEMA_Crossover': {'min_volume_factor': 1.1},
            'TEMA_Crossover': {'min_volume_factor': 1.1},
            
            # MACD strategies
            'MACD_Signal_Crossover': {'min_volume_factor': 1.1},
            'MACD_Zero_Line_Crossover': {'min_volume_factor': 1.0},
            
            # Oscillator strategies (more lenient volume requirements)
            'RSI_Overbought_Oversold': {'min_volume_factor': 0.8},
            'Stochastic_Overbought_Oversold': {'min_volume_factor': 0.8},
            'Williams_Percent_R_Overbought_Oversold': {'min_volume_factor': 0.8},
            'CCI_Crossover': {'min_volume_factor': 0.8},
            
            # Pattern recognition strategies
            'Chart_Patterns': {'min_volume_factor': 0.9},
            'Fibonacci_Retracement': {'min_volume_factor': 1.0},
            
            # Volume-based strategies (already volume-focused)
            'Volume_Profile': {'min_volume_factor': 0.7},
            'On_Balance_Volume': {'min_volume_factor': 0.7},
            'Volume_Price_Trend': {'min_volume_factor': 0.7},
            
            # Candlestick patterns
            'Candlestick_Hammer': {'min_volume_factor': 1.0},
            'Candlestick_Bullish_Engulfing': {'min_volume_factor': 1.1},
            'Candlestick_Doji': {'min_volume_factor': 0.9},
            
            # Ichimoku strategies
            'Ichimoku_Cloud_Breakout': {'min_volume_factor': 1.2, 'breakout': True},
            'Ichimoku_Kijun_Tenkan_Crossover': {'min_volume_factor': 1.0},
        }
        
        # Update with strategy-specific parameters if available
        if self.name in strategy_params:
            params.update(strategy_params[self.name])
        
        return params
    
    def validate_data(self, data: pd.DataFrame, min_periods: int = 1) -> bool:
        """
        Validate that the data contains the required columns and sufficient data points.
        
        Args:
            data: DataFrame to validate
            min_periods: Minimum number of data points required
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if data.empty:
            logger.warning(f"{self.name}: Empty data provided")
            return False
            
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"{self.name}: Missing columns: {missing_columns}")
            return False
            
        if len(data) < min_periods:
            logger.warning(f"{self.name}: Insufficient data points. Required: {min_periods}, Got: {len(data)}")
            return False
            
        return True
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get a parameter value with optional default.
        
        Args:
            key: Parameter key
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        return self.params.get(key, default)
    
    def log_signal(self, signal: int, reason: str, data: pd.DataFrame) -> None:
        """
        Log the signal with context information.
        
        Args:
            signal: Signal value (1 or -1)
            reason: Reason for the signal
            data: Data used for the signal
        """
        signal_type = "BUY" if signal == 1 else "SELL/NO_BUY"
        latest_close = data['Close'].iloc[-1] if not data.empty else "N/A"
        
        logger.info(f"{self.name}: {signal_type} signal - {reason} (Latest close: {latest_close})")
    
    def apply_volume_filtering(self, signal: int, data: pd.DataFrame, 
                              signal_type: str = 'bullish', 
                              breakout: bool = False, 
                              level: float = None,
                              min_volume_factor: float = 0.8) -> Dict[str, Any]:
        """
        Apply enhanced volume confirmation filtering to trading signals.
        
        Args:
            signal: Original signal (1, -1, or 0)
            data: DataFrame with OHLCV data
            signal_type: 'bullish' or 'bearish' signal type
            breakout: Whether this is a breakout signal
            level: Support/resistance level if applicable
            min_volume_factor: Minimum volume factor to accept signal
            
        Returns:
            Dictionary with filtered signal and volume analysis
        """
        try:
            if signal == 0 or len(data) < 20:
                return {
                    'signal': signal,
                    'volume_filtered': False,
                    'volume_factor': 1.0,
                    'reason': 'No signal or insufficient data'
                }
            
            # Get enhanced volume confirmation
            volume_analysis = get_enhanced_volume_confirmation(
                data, signal_type, breakout, level
            )
            
            volume_factor = volume_analysis['factor']
            volume_strength = volume_analysis['strength']
            
            # Apply volume filtering
            if volume_factor >= min_volume_factor:
                filtered_signal = signal
                volume_filtered = False
                reason = f"Volume confirmation passed: {volume_strength} (factor: {volume_factor})"
            else:
                filtered_signal = 0  # Filter out weak volume signals
                volume_filtered = True
                reason = f"Signal filtered due to weak volume: {volume_strength} (factor: {volume_factor})"
            
            return {
                'signal': filtered_signal,
                'original_signal': signal,
                'volume_filtered': volume_filtered,
                'volume_factor': volume_factor,
                'volume_strength': volume_strength,
                'volume_details': volume_analysis.get('details', []),
                'vwap_context': volume_analysis.get('vwap_context', ''),
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"{self.name}: Error in volume filtering: {e}")
            return {
                'signal': signal,
                'volume_filtered': False,
                'volume_factor': 1.0,
                'reason': f'Volume filtering error: {e}'
            }
    
    def get_volume_confirmation_strength(self, data: pd.DataFrame, signal_type: str = 'bullish') -> float:
        """
        Get volume confirmation strength for signal quality assessment.
        
        Args:
            data: DataFrame with OHLCV data
            signal_type: 'bullish' or 'bearish' signal type
            
        Returns:
            float: Volume confirmation strength (0.0 to 2.0+)
        """
        try:
            if len(data) < 20:
                return 1.0
            
            volume_analysis = get_enhanced_volume_confirmation(data, signal_type)
            return volume_analysis.get('factor', 1.0)
            
        except Exception as e:
            logger.error(f"{self.name}: Error getting volume confirmation strength: {e}")
            return 1.0


class BacktraderStrategyMeta(type(ABC), type(bt.Strategy)):
    """Metaclass to resolve conflicts between ABC and bt.Strategy."""
    pass

class BacktraderStrategy(BaseStrategy, bt.Strategy, metaclass=BacktraderStrategyMeta):
    """
    Base class for strategies that can be used with Backtrader.
    
    This class bridges the gap between our simple strategy interface
    and Backtrader's more complex strategy system.
    """
    
    def __init__(self):
        # Initialize BaseStrategy (ABC) part
        BaseStrategy.__init__(self)
        # Initialize Backtrader part
        bt.Strategy.__init__(self)
        # Backtrader strategy initialization
        self.data_close = self.datas[0].close
        self.data_open = self.datas[0].open
        self.data_high = self.datas[0].high
        self.data_low = self.datas[0].low
        self.data_volume = self.datas[0].volume
        
    def next(self):
        """
        Backtrader's next method - called for each bar.
        
        This method converts backtrader data to our DataFrame format
        and calls the run_strategy method.
        """
        try:
            # Convert backtrader data to DataFrame format
            lookback_period = getattr(self, 'lookback_period', 250)
            
            # Check if we have enough data available
            available_data = len(self.data_close)
            if available_data < 200:  # Skip if insufficient data for meaningful analysis
                return
            
            # Get the required amount of historical data (use all available data up to lookback_period)
            data_length = min(lookback_period, available_data)
            data_dict = {
                'Open': [self.data_open[-i] for i in range(data_length, 0, -1)],
                'High': [self.data_high[-i] for i in range(data_length, 0, -1)],
                'Low': [self.data_low[-i] for i in range(data_length, 0, -1)],
                'Close': [self.data_close[-i] for i in range(data_length, 0, -1)],
                'Volume': [self.data_volume[-i] for i in range(data_length, 0, -1)]
            }
            
            # Create DataFrame
            df = pd.DataFrame(data_dict)
            
            # Ensure we have enough data for the strategy
            if len(df) < 200:
                return  # Skip this iteration if insufficient data
            
            # Run the strategy
            signal = self.run_strategy(df)
            
            # Execute trades based on signal
            if signal == 1 and not self.position:
                self.buy()
            elif signal == -1 and self.position:
                self.sell()
                
        except Exception as e:
            logger.error(f"{self.name}: Error in next() method: {e}")


class TechnicalIndicatorMixin:
    """
    Mixin class providing common technical indicator calculations.
    """
    
    @staticmethod
    def calculate_sma(data: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD."""
        ema_fast = data.ewm(span=fast_period).mean()
        ema_slow = data.ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
