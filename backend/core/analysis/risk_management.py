"""
Risk Management Module
======================

Provides utilities for risk-per-trade sizing, ATR-based stops, portfolio guards,
and daily loss limits, aligned with swing-trading guidelines in config.
"""

from typing import Dict, Any, List, Optional
from math import floor
from utils.logger import setup_logging
from config import RISK_MANAGEMENT

logger = setup_logging()


class RiskManager:
    """
    Risk utilities for sizing and constraints.
    """

    def __init__(self):
        pass

    # --- Core sizing and stops ---
    def atr_stop(self, entry_price: float, atr: float, atr_multiplier: Optional[float] = None) -> float:
        """Compute ATR-based stop below entry for long positions."""
        try:
            mult = atr_multiplier if atr_multiplier is not None else RISK_MANAGEMENT['risk_reward'].get('min_ratio', 2.5)  # fallback
            # Prefer using dedicated ATR stop multiplier if present
            mult = RISK_MANAGEMENT.get('position_sizing', {}).get('atr_stop_multiplier', 1.5) if 'position_sizing' in RISK_MANAGEMENT else 1.5
            stop = max(0.0, entry_price - mult * atr)
            return stop
        except Exception as e:
            logger.error(f"ATR stop error: {e}")
            return max(0.0, entry_price * 0.96)

    def position_size_atr(self,
                          equity: float,
                          entry_price: float,
                          stop_loss: float,
                          risk_per_trade: Optional[float] = None,
                          max_position_pct: Optional[float] = None) -> Dict[str, Any]:
        """
        ATR-based position sizing capped by risk-per-trade and max position percent.
        Risk per share = entry - stop. Shares = floor((equity * risk) / risk_per_share).
        Also cap by max_position_pct of equity.
        """
        try:
            if entry_price <= 0 or stop_loss <= 0 or equity <= 0:
                return {'position_size': 0, 'reason': 'invalid_inputs'}

            cfg = RISK_MANAGEMENT.get('position_sizing', {})
            risk = risk_per_trade if risk_per_trade is not None else cfg.get('risk_per_trade', 0.01)
            max_pct = max_position_pct if max_position_pct is not None else cfg.get('max_position_pct', 0.20)

            risk_per_share = max(1e-6, entry_price - stop_loss)
            capital_at_risk = equity * risk
            shares_risk_capped = floor(capital_at_risk / risk_per_share)

            # Cap by max position percent
            max_value = equity * max_pct
            shares_value_capped = floor(max_value / entry_price)

            shares = max(0, min(shares_risk_capped, shares_value_capped))
            return {
                'position_size': int(shares),
                'risk_per_share': risk_per_share,
                'capital_at_risk': capital_at_risk,
                'max_position_value': max_value,
                'constraints': {
                    'risk_per_trade': risk,
                    'max_position_pct': max_pct
                }
            }
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return {'position_size': 0, 'error': str(e)}

    # --- Portfolio-level guards ---
    def check_portfolio_guards(self,
                               positions: List[Dict[str, Any]],
                               new_position_value: float,
                               new_position_sector: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate portfolio-level constraints: max concurrent positions and sector concentration.
        positions: list of { 'value': float, 'sector': str }
        """
        try:
            pcfg = RISK_MANAGEMENT.get('portfolio_constraints', {})
            max_positions = pcfg.get('max_concurrent_positions', 5)
            max_sector_conc = pcfg.get('max_sector_concentration', 0.40)

            existing_values = [p.get('value', 0.0) for p in positions]
            total_equity = sum(existing_values) + new_position_value
            current_positions = len([p for p in positions if p.get('value', 0.0) > 0])

            can_open = True
            reasons = []

            # Max positions
            if current_positions >= max_positions:
                can_open = False
                reasons.append(f"max positions reached: {current_positions}/{max_positions}")

            # Sector concentration
            if new_position_sector:
                sector_sum = sum(p.get('value', 0.0) for p in positions if p.get('sector') == new_position_sector)
                sector_after = sector_sum + new_position_value
                conc = (sector_after / total_equity) if total_equity > 0 else 1.0
                if conc > max_sector_conc:
                    can_open = False
                    reasons.append(f"sector concentration {conc:.2f} exceeds {max_sector_conc:.2f}")

            return {'can_open': can_open, 'reasons': reasons, 'total_equity': total_equity}
        except Exception as e:
            logger.error(f"Portfolio guards error: {e}")
            return {'can_open': False, 'error': str(e)}

    # --- Daily loss limit ---
    def check_daily_loss_cap(self, equity_start: float, equity_now: float) -> Dict[str, Any]:
        """Check if daily loss limit breached; return pause flag."""
        try:
            pcfg = RISK_MANAGEMENT.get('portfolio_constraints', {})
            daily_loss_limit = pcfg.get('daily_loss_limit', 0.03)
            pause_on_breach = pcfg.get('pause_on_limit_breach', True)

            if equity_start <= 0:
                return {'breached': False, 'pause': False, 'reason': 'invalid_equity_start'}

            loss_pct = (equity_start - equity_now) / equity_start
            breached = loss_pct >= daily_loss_limit
            return {
                'breached': bool(breached),
                'pause': bool(breached and pause_on_breach),
                'loss_pct': float(loss_pct),
                'limit': float(daily_loss_limit)
            }
        except Exception as e:
            logger.error(f"Daily loss cap error: {e}")
            return {'breached': False, 'pause': False, 'error': str(e)}

    # --- Profit targets helper ---
    def calculate_profit_targets(self, entry_price: float, atr: float) -> Dict[str, float]:
        """Compute TP1/TP2 and trail using config ATR multipliers."""
        try:
            exit_cfg = {
                'tp1_atr': 1.0,
                'tp2_atr': 2.5,
                'trail_atr': 3.0,
            }
            tp1 = entry_price + exit_cfg['tp1_atr'] * atr
            tp2 = entry_price + exit_cfg['tp2_atr'] * atr
            trail = exit_cfg['trail_atr'] * atr
            return {'take_profit_1': tp1, 'take_profit_2': tp2, 'trailing_stop_distance': trail}
        except Exception as e:
            logger.error(f"Profit targets error: {e}")
            return {'take_profit_1': 0.0, 'take_profit_2': 0.0, 'trailing_stop_distance': 0.0}

