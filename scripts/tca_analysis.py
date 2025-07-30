# scripts/tca_analysis.py

import numpy as np
from utils.logger import setup_logging
logger = setup_logging()

class TransactionCostAnalyzer:
    """
    A framework for analyzing and estimating transaction costs.
    
    This module provides methods to calculate potential trading costs, including
    brokerage fees, taxes, and slippage.
    """
    def __init__(self, brokerage_rate=0.0005, stt_rate=0.001, slippage_pct=0.0005):
        """
        Args:
            brokerage_rate (float): The brokerage fee as a percentage of trade value.
            stt_rate (float): The Securities Transaction Tax as a percentage.
            slippage_pct (float): The estimated slippage as a percentage.
        """
        self.brokerage_rate = brokerage_rate
        self.stt_rate = stt_rate
        self.slippage_pct = slippage_pct

    def estimate_trade_costs(self, trade_value: float):
        """
        Estimates the total cost of a single trade.
        
        Args:
            trade_value (float): The total value of the trade.
            
        Returns:
            A dictionary containing the breakdown of estimated costs.
        """
        brokerage = trade_value * self.brokerage_rate
        stt = trade_value * self.stt_rate
        slippage_cost = trade_value * self.slippage_pct
        
        total_cost = brokerage + stt + slippage_cost
        
        logger.debug(f"TCA: Trade Value={trade_value:.2f}, Brokerage={brokerage:.2f}, STT={stt:.2f}, Slippage={slippage_cost:.2f}, Total={total_cost:.2f}")
        
        return {
            'trade_value': trade_value,
            'brokerage': brokerage,
            'stt': stt,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'cost_as_pct_of_value': (total_cost / trade_value) * 100 if trade_value > 0 else 0
        }

if __name__ == '__main__':
    tca = TransactionCostAnalyzer()
    trade_value = 100000  # Example trade of 1 lakh
    costs = tca.estimate_trade_costs(trade_value)
    
    print("\n=== Transaction Cost Analysis Example ===")
    print(f"Estimated costs for a trade of {trade_value:.2f}:")
    for cost_type, value in costs.items():
        print(f"  - {cost_type.replace('_', ' ').title()}: {value:.2f}")

