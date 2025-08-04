# scripts/market_microstructure.py
import pandas as pd
from typing import Dict, Any
from utils.logger import setup_logging
logger = setup_logging()

class MarketMicrostructureAnalyzer:
    """
    Analyzes Level 2/3 order book data to extract microstructure insights.
    
    This class is designed to process high-frequency order book data to calculate
    metrics like order imbalance, book depth, and spread, which can be used
    as predictive signals.
    """

    def __init__(self):
        """
        Initializes the analyzer. In a real scenario, this would connect
        to a high-frequency data feed.
        """
        logger.info("MarketMicrostructureAnalyzer initialized. (Simulation Mode)")

    def fetch_level2_data(self, symbol: str) -> Dict[str, Any]:
        """
        Simulates fetching Level 2 order book data for a given symbol.
        
        In a real implementation, this method would connect to a data provider's API
        (e.g., via WebSocket) to get a snapshot of the order book.
        
        Args:
            symbol: The stock symbol.
            
        Returns:
            A dictionary representing the simulated order book.
        """
        logger.debug(f"Simulating Level 2 data fetch for {symbol}")
        
        # Simulate a realistic-looking order book snapshot
        simulated_book = {
            'symbol': symbol,
            'bids': [
                {'price': 100.00, 'size': 500},
                {'price': 99.99, 'size': 800},
                {'price': 99.98, 'size': 1200},
                {'price': 99.97, 'size': 1500},
                {'price': 99.96, 'size': 2000},
            ],
            'asks': [
                {'price': 100.01, 'size': 450},
                {'price': 100.02, 'size': 750},
                {'price': 100.03, 'size': 1100},
                {'price': 100.04, 'size': 1600},
                {'price': 100.05, 'size': 1800},
            ]
        }
        return simulated_book

    def calculate_order_imbalance(self, order_book: Dict[str, Any]) -> float:
        """
        Calculates the Order Imbalance Ratio (OIR) from the order book.
        
        OIR = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        
        A positive OIR suggests buying pressure, while a negative OIR suggests
        selling pressure.
        
        Args:
            order_book: A dictionary representing the order book.
            
        Returns:
            The calculated Order Imbalance Ratio.
        """
        try:
            bid_volume = sum(level['size'] for level in order_book['bids'])
            ask_volume = sum(level['size'] for level in order_book['asks'])
            
            if (bid_volume + ask_volume) == 0:
                return 0.0
            
            oir = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            logger.debug(f"Order imbalance calculated for {order_book['symbol']}: {oir:.4f}")
            return oir
            
        except Exception as e:
            logger.error(f"Error calculating order imbalance: {e}")
            return 0.0

    def analyze(self, symbol: str) -> Dict[str, Any]:
        """
        Performs a full market microstructure analysis for a symbol.
        
        Args:
            symbol: The stock symbol to analyze.
            
        Returns:
            A dictionary containing microstructure analysis insights.
        """
        logger.info(f"Running market microstructure analysis for {symbol}...")
        
        # 1. Fetch order book data
        order_book = self.fetch_level2_data(symbol)
        
        # 2. Calculate order imbalance
        order_imbalance_ratio = self.calculate_order_imbalance(order_book)
        
        # 3. Generate predictive signals (placeholder logic)
        signal = "NEUTRAL"
        if order_imbalance_ratio > 0.15:
            signal = "SHORT_TERM_BUY"
        elif order_imbalance_ratio < -0.15:
            signal = "SHORT_TERM_SELL"
            
        analysis_result = {
            'symbol': symbol,
            'order_imbalance_ratio': round(order_imbalance_ratio, 4),
            'short_term_signal': signal,
            'spread': order_book['asks'][0]['price'] - order_book['bids'][0]['price'],
            'total_bid_volume': sum(level['size'] for level in order_book['bids']),
            'total_ask_volume': sum(level['size'] for level in order_book['asks'])
        }
        
        logger.info(f"Microstructure analysis for {symbol} complete. Signal: {signal} (OIR: {order_imbalance_ratio:.4f})")
        return analysis_result

if __name__ == '__main__':
    # Example usage of the analyzer
    microstructure_analyzer = MarketMicrostructureAnalyzer()
    
    symbol = "RELIANCE.NS"
    analysis = microstructure_analyzer.analyze(symbol)
    
    print("=== Market Microstructure Analysis Example ===")
    print(f"Symbol: {analysis['symbol']}")
    print(f"  - Bid-Ask Spread: {analysis['spread']:.2f}")
    print(f"  - Order Imbalance Ratio: {analysis['order_imbalance_ratio']}")
    print(f"  - Total Bid Volume: {analysis['total_bid_volume']}")
    print(f"  - Total Ask Volume: {analysis['total_ask_volume']}")
    print(f"  - Short-term Signal: {analysis['short_term_signal']}")

