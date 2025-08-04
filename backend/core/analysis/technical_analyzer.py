"""
Technical Analysis Module
========================

This module handles all technical analysis related functionality.
Extracted from the original analyzer.py to improve maintainability.
"""

import pandas as pd
import numpy as np
import talib as ta
from typing import Dict, Any
from utils.logger import setup_logging

logger = setup_logging()

class TechnicalAnalyzer:
    """
    Handles technical analysis of stock data using various indicators.
    """
    
    def __init__(self):
        """Initialize the technical analyzer."""
        pass
    
    def calculate_technical_indicators(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate various technical indicators for the given historical data.
        
        Args:
            historical_data: DataFrame with OHLCV data
            
        Returns:
            Dictionary containing calculated technical indicators
        """
        try:
            if historical_data.empty:
                return {}
            
            current_price = historical_data['Close'].iloc[-1]
            
            # Calculate technical indicators
            sma_20 = ta.SMA(historical_data['Close'].values, timeperiod=20)
            sma_50 = ta.SMA(historical_data['Close'].values, timeperiod=50)
            ema_12 = ta.EMA(historical_data['Close'].values, timeperiod=12)
            ema_26 = ta.EMA(historical_data['Close'].values, timeperiod=26)
            rsi = ta.RSI(historical_data['Close'].values, timeperiod=14)
            atr = ta.ATR(historical_data['High'].values, historical_data['Low'].values, 
                        historical_data['Close'].values, timeperiod=14)
            
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(historical_data['Close'].values, 
                                                     timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            
            # Get latest values with fallbacks
            indicators = {
                'current_price': current_price,
                'sma_20': sma_20[-1] if not pd.isna(sma_20[-1]) else current_price,
                'sma_50': sma_50[-1] if not pd.isna(sma_50[-1]) else current_price,
                'ema_12': ema_12[-1] if not pd.isna(ema_12[-1]) else current_price,
                'ema_26': ema_26[-1] if not pd.isna(ema_26[-1]) else current_price,
                'rsi': rsi[-1] if not pd.isna(rsi[-1]) else 50,
                'atr': atr[-1] if not pd.isna(atr[-1]) else current_price * 0.02,
                'bb_upper': bb_upper[-1] if not pd.isna(bb_upper[-1]) else current_price * 1.05,
                'bb_lower': bb_lower[-1] if not pd.isna(bb_lower[-1]) else current_price * 0.95,
                'bb_middle': bb_middle[-1] if not pd.isna(bb_middle[-1]) else current_price
            }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}
    
    def find_support_resistance(self, data: pd.DataFrame, level_type: str) -> float:
        """
        Find support or resistance levels using pivot points.
        
        Args:
            data: Historical price data
            level_type: 'support' or 'resistance'
            
        Returns:
            Support or resistance level
        """
        try:
            if len(data) < 10:
                if level_type == 'support':
                    return data['Low'].min()
                else:
                    return data['High'].max()
            
            # Look for pivot points in the last 20 days
            lookback = min(20, len(data))
            recent_data = data.tail(lookback)
            
            if level_type == 'support':
                # Find local minima (support levels)
                levels = []
                for i in range(2, len(recent_data) - 2):
                    if (recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-1] and 
                        recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+1] and
                        recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i-2] and 
                        recent_data['Low'].iloc[i] < recent_data['Low'].iloc[i+2]):
                        levels.append(recent_data['Low'].iloc[i])
                
                return max(levels) if levels else recent_data['Low'].min()
            
            else:  # resistance
                # Find local maxima (resistance levels)
                levels = []
                for i in range(2, len(recent_data) - 2):
                    if (recent_data['High'].iloc[i] > recent_data['High'].iloc[i-1] and 
                        recent_data['High'].iloc[i] > recent_data['High'].iloc[i+1] and
                        recent_data['High'].iloc[i] > recent_data['High'].iloc[i-2] and 
                        recent_data['High'].iloc[i] > recent_data['High'].iloc[i+2]):
                        levels.append(recent_data['High'].iloc[i])
                
                return min(levels) if levels else recent_data['High'].max()
                
        except Exception as e:
            logger.error(f"Error finding {level_type} level: {e}")
            if level_type == 'support':
                return data['Low'].min()
            else:
                return data['High'].max()
    
    def calculate_confidence(self, current_price: float, sma_20: float, sma_50: float, 
                            rsi: float, recommendation: str) -> float:
        """
        Calculate confidence level for the recommendation.
        
        Args:
            current_price: Current stock price
            sma_20: 20-day Simple Moving Average
            sma_50: 50-day Simple Moving Average
            rsi: Relative Strength Index
            recommendation: Buy/Sell/Hold recommendation
            
        Returns:
            Confidence level (0.0 to 1.0)
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Price relative to moving averages
            if recommendation == 'BUY':
                if current_price > sma_20 > sma_50:
                    confidence += 0.2
                elif current_price > sma_20:
                    confidence += 0.1
                    
                # RSI signals
                if 40 < rsi < 60:
                    confidence += 0.2
                elif 30 < rsi < 70:
                    confidence += 0.1
                    
            elif recommendation == 'SELL':
                if current_price < sma_20 < sma_50:
                    confidence += 0.2
                elif current_price < sma_20:
                    confidence += 0.1
                    
                # RSI signals
                if rsi > 70:
                    confidence += 0.2
                elif rsi < 30:
                    confidence -= 0.1
            
            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            return round(confidence, 2)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
