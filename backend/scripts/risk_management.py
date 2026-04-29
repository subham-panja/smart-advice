import pandas as pd
import numpy as np
import talib as ta
import logging
from typing import Dict, Any, Optional
from config import RISK_MANAGEMENT

logger = logging.getLogger(__name__)

class RiskManager:
    """Handles stop-loss, position sizing, and risk-reward evaluation."""
    
    def __init__(self, account_balance: float = 100000):
        s = RISK_MANAGEMENT.get('account_settings', {})
        p = RISK_MANAGEMENT.get('position_sizing', {})
        self.balance = account_balance or s.get('initial_capital', 100000)
        self.risk_per_trade = p.get('risk_per_trade', 0.01)
        self.min_rr = RISK_MANAGEMENT.get('risk_reward', {}).get('min_ratio', 1.5)

    def calculate_risk_params(self, df: pd.DataFrame, entry: float) -> Dict[str, Any]:
        """Calculates optimal stop loss, position size, and targets."""
        atr = ta.ATR(df['High'], df['Low'], df['Close'], 14).iloc[-1]
        sl = entry - (atr * 2.0)
        
        # Position Sizing
        risk_amt = self.balance * self.risk_per_trade
        risk_per_share = entry - sl
        size = int(risk_amt / risk_per_share) if risk_per_share > 0 else 0
        
        # Targets
        targets = {f"{r}x": entry + (risk_per_share * r) for r in [1.5, 2.5, 3.5]}
        
        return {
            'stop_loss': round(sl, 2),
            'position_size': size,
            'position_value': round(size * entry, 2),
            'risk_amount': round(risk_amt, 2),
            'targets': {k: round(v, 2) for k, v in targets.items()},
            'risk_reward_ok': True # Simplified
        }

    def validate_trade(self, entry: float, sl: float, target: float) -> bool:
        risk = entry - sl
        reward = target - entry
        return (reward / risk >= self.min_rr) if risk > 0 else False
