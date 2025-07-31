# scripts/tca_analysis.py

import numpy as np
from utils.logger import setup_logging

logger = setup_logging()

class TransactionCostAnalyzer:
    """
    A framework for analyzing and estimating transaction costs.
    Enhanced to provide more realistic and detailed cost estimations.
    """
    def __init__(self, brokerage_rate=0.0005, stt_rate=0.001, slippage_pct=0.0005, exchange_fee_rate=0.0000325):
        self.brokerage_rate = brokerage_rate
        self.stt_rate = stt_rate
        self.slippage_pct = slippage_pct
        self.exchange_fee_rate = exchange_fee_rate

    def estimate_trade_costs(self, trade_value: float, is_buy: bool = True):
        """
        Estimates the total cost of a single trade, considering buy/sell side.
        """
        brokerage = trade_value * self.brokerage_rate
        stt = trade_value * self.stt_rate if is_buy else 0  # STT is typically on delivery
        exchange_fees = trade_value * self.exchange_fee_rate
        slippage_cost = trade_value * self.slippage_pct

        total_cost = brokerage + stt + exchange_fees + slippage_cost

        logger.debug(f"TCA: Trade Value={trade_value:.2f}, Brokerage={brokerage:.2f}, STT={stt:.2f}, Slippage={slippage_cost:.2f}, Total={total_cost:.2f}")

        return {
            'trade_value': trade_value,
            'brokerage': brokerage,
            'stt': stt,
            'exchange_fees': exchange_fees,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'cost_as_pct_of_value': (total_cost / trade_value) * 100 if trade_value > 0 else 0
        }

    def estimate_round_trip_costs(self, trade_value: float):
        """
        Estimates the total cost for a complete buy-sell round trip.
        """
        buy_costs = self.estimate_trade_costs(trade_value, is_buy=True)
        sell_costs = self.estimate_trade_costs(trade_value, is_buy=False)
        
        total_round_trip_cost = buy_costs['total_cost'] + sell_costs['total_cost']
        
        return {
            'buy_costs': buy_costs,
            'sell_costs': sell_costs,
            'total_round_trip_cost': total_round_trip_cost,
            'round_trip_cost_pct': (total_round_trip_cost / trade_value) * 100 if trade_value > 0 else 0,
            'breakeven_profit_required': total_round_trip_cost
        }

    def analyze_trade_efficiency(self, expected_profit: float, trade_value: float):
        """
        Analyzes if a trade is efficient given expected profit vs transaction costs.
        """
        round_trip = self.estimate_round_trip_costs(trade_value)
        total_costs = round_trip['total_round_trip_cost']
        
        efficiency_ratio = expected_profit / total_costs if total_costs > 0 else 0
        net_profit = expected_profit - total_costs
        
        # Determine trade recommendation based on efficiency
        if efficiency_ratio >= 3.0:
            recommendation = "HIGHLY_EFFICIENT"
        elif efficiency_ratio >= 2.0:
            recommendation = "EFFICIENT"
        elif efficiency_ratio >= 1.5:
            recommendation = "MODERATELY_EFFICIENT"
        elif efficiency_ratio >= 1.0:
            recommendation = "BARELY_PROFITABLE"
        else:
            recommendation = "INEFFICIENT"
        
        return {
            'expected_profit': expected_profit,
            'total_costs': total_costs,
            'net_profit': net_profit,
            'efficiency_ratio': efficiency_ratio,
            'recommendation': recommendation,
            'cost_breakdown': round_trip
        }


