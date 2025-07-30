"""
Fundamental Analysis Module
File: scripts/fundamental_analysis.py

This module performs fundamental analysis on stocks by evaluating various financial metrics.
"""

import requests
import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional
from utils.logger import setup_logging
from config import MAX_RETRIES, REQUEST_DELAY, BACKOFF_MULTIPLIER
import numpy as np
import time
import random

logger = setup_logging()

class FundamentalAnalysis:
    """
    Perform fundamental analysis on stocks.
    """
    
    @staticmethod
    def get_financial_data_from_yfinance(symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get financial data from yfinance API with retry mechanism.
        
        Args:
            symbol: Stock symbol (NSE format)
        
        Returns:
            Dictionary containing financial metrics or None if error
        """
        # Add .NS suffix for NSE stocks if not present
        if '.NS' not in symbol and '.BO' not in symbol:
            symbol = f"{symbol}.NS"
        
        for attempt in range(MAX_RETRIES):
            try:
                # Add progressive delay with jitter to avoid overwhelming the API
                if attempt > 0:
                    base_delay = REQUEST_DELAY * (BACKOFF_MULTIPLIER ** attempt)
                    jitter = random.uniform(0, base_delay * 0.3)  # Add up to 30% jitter
                    total_delay = base_delay + jitter
                    time.sleep(total_delay)
                    logger.info(f"Retry attempt {attempt + 1} for fundamental data of {symbol} after {total_delay:.2f}s delay")
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                if not info or 'regularMarketPrice' not in info:
                    if attempt == MAX_RETRIES - 1:
                        logger.warning(f"No financial info available for {symbol} after {MAX_RETRIES} attempts")
                        return None
                    continue
                
                # Extract key financial metrics with safe defaults
                financial_data = {
                    'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                    'pb_ratio': info.get('priceToBook'),
                    'de_ratio': info.get('debtToEquity'),
                    'eps_growth': info.get('earningsGrowth'),
                    'revenue_growth': info.get('revenueGrowth'),
                    'dividend_yield': info.get('dividendYield'),
                    'market_cap': info.get('marketCap'),
                    'current_ratio': info.get('currentRatio'),
                    'roe': info.get('returnOnEquity'),
                    'profit_margins': info.get('profitMargins'),
                    'beta': info.get('beta'),
                    'price_to_sales': info.get('priceToSalesTrailing12Months')
                }
                
                logger.debug(f"Successfully fetched fundamental data for {symbol}")
                return financial_data
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Categorize errors for better handling
                if any(keyword in error_msg for keyword in ['http', 'curl', 'connection', '401', 'unauthorized', 'timeout', 'timed out']):
                    logger.warning(f"Network/timeout error for fundamental data of {symbol} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                    if attempt == MAX_RETRIES - 1:
                        logger.error(f"Failed to fetch fundamental data for {symbol} after {MAX_RETRIES} attempts")
                        return None
                    continue
                else:
                    logger.error(f"Non-retryable error fetching fundamental data for {symbol}: {e}")
                    return None
        
        return None
    
    @staticmethod
    def calculate_fundamental_score(financial_data: Dict[str, Any]) -> float:
        """
        Calculate fundamental analysis score based on financial metrics.
        
        Args:
            financial_data: Dictionary containing financial metrics
            
        Returns:
            Score between -1 and 1
        """
        score = 0
        total_weight = 0
        
        # P/E Ratio Analysis (Weight: 20%)
        pe_ratio = financial_data.get('pe_ratio')
        if pe_ratio and pe_ratio > 0:
            if pe_ratio < 15:
                score += 1.0 * 0.2  # Excellent
            elif pe_ratio < 25:
                score += 0.5 * 0.2  # Good
            elif pe_ratio < 35:
                score += 0 * 0.2    # Neutral
            else:
                score += -0.5 * 0.2 # Poor
            total_weight += 0.2
        
        # P/B Ratio Analysis (Weight: 15%)
        pb_ratio = financial_data.get('pb_ratio')
        if pb_ratio and pb_ratio > 0:
            if pb_ratio < 1.5:
                score += 1.0 * 0.15  # Excellent
            elif pb_ratio < 3:
                score += 0.5 * 0.15  # Good
            elif pb_ratio < 5:
                score += 0 * 0.15    # Neutral
            else:
                score += -0.5 * 0.15 # Poor
            total_weight += 0.15
        
        # Debt to Equity Analysis (Weight: 15%)
        de_ratio = financial_data.get('de_ratio')
        if de_ratio is not None and de_ratio >= 0:
            if de_ratio < 0.3:
                score += 1.0 * 0.15  # Excellent
            elif de_ratio < 0.6:
                score += 0.5 * 0.15  # Good
            elif de_ratio < 1.0:
                score += 0 * 0.15    # Neutral
            else:
                score += -0.5 * 0.15 # Poor
            total_weight += 0.15
        
        # EPS Growth Analysis (Weight: 20%)
        eps_growth = financial_data.get('eps_growth')
        if eps_growth is not None:
            if eps_growth > 0.15:     # 15%+ growth
                score += 1.0 * 0.2   # Excellent
            elif eps_growth > 0.1:   # 10-15% growth
                score += 0.5 * 0.2   # Good
            elif eps_growth > 0:     # Positive growth
                score += 0.25 * 0.2  # Fair
            else:
                score += -0.5 * 0.2  # Poor
            total_weight += 0.2
        
        # Revenue Growth Analysis (Weight: 15%)
        revenue_growth = financial_data.get('revenue_growth')
        if revenue_growth is not None:
            if revenue_growth > 0.1:   # 10%+ growth
                score += 1.0 * 0.15  # Excellent
            elif revenue_growth > 0.05: # 5-10% growth
                score += 0.5 * 0.15  # Good
            elif revenue_growth > 0:   # Positive growth
                score += 0.25 * 0.15 # Fair
            else:
                score += -0.5 * 0.15 # Poor
            total_weight += 0.15
        
        # Dividend Yield Analysis (Weight: 10%)
        dividend_yield = financial_data.get('dividend_yield')
        if dividend_yield is not None:
            if dividend_yield > 0.03:   # 3%+ yield
                score += 0.5 * 0.1   # Good
            elif dividend_yield > 0.01: # 1-3% yield
                score += 0.25 * 0.1  # Fair
            else:
                score += 0 * 0.1     # Neutral
            total_weight += 0.1
        
        # Current Ratio Analysis (Weight: 5%)
        current_ratio = financial_data.get('current_ratio')
        if current_ratio and current_ratio > 0:
            if current_ratio > 2:
                score += 0.5 * 0.05  # Good liquidity
            elif current_ratio > 1:
                score += 0.25 * 0.05 # Fair liquidity
            else:
                score += -0.5 * 0.05 # Poor liquidity
            total_weight += 0.05
        
        # Normalize score based on available metrics
        if total_weight > 0:
            normalized_score = score / total_weight
            # Ensure score is between -1 and 1
            normalized_score = max(-1, min(1, normalized_score))
            
            # Apply slight positive bias to neutral scores for better recommendations
            if -0.1 <= normalized_score <= 0.1:
                normalized_score = max(0.05, normalized_score + 0.05)
            
            return normalized_score
        else:
            return 0.1  # Default to slightly positive when no data available
    
    @staticmethod
    def perform_fundamental_analysis(symbol: str) -> float:
        """
        Perform comprehensive fundamental analysis using real financial data.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            float: Score based on fundamental metrics (-1 to 1)
        """
        try:
            # Get real financial data from yfinance
            financial_data = FundamentalAnalysis.get_financial_data_from_yfinance(symbol)
            
            if not financial_data:
                logger.warning(f"No financial data available for {symbol}, using default positive score")
                return 0.1  # Default to slightly positive
            
            # Calculate fundamental score
            fundamental_score = FundamentalAnalysis.calculate_fundamental_score(financial_data)
            
            # Log the analysis details
            logger.info(f"Fundamental analysis for {symbol} - Score: {fundamental_score:.3f}")
            logger.debug(f"Financial metrics for {symbol}: PE={financial_data.get('pe_ratio')}, "
                        f"PB={financial_data.get('pb_ratio')}, DE={financial_data.get('de_ratio')}, "
                        f"EPS_Growth={financial_data.get('eps_growth')}, Rev_Growth={financial_data.get('revenue_growth')}")
            
            return fundamental_score
            
        except Exception as e:
            logger.error(f"Error performing fundamental analysis for {symbol}: {e}")
            return 0.1  # Return slightly positive score on error


