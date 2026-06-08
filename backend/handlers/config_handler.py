import logging

import config

logger = logging.getLogger(__name__)


def get_strategies():
    from utils.strategy_loader import StrategyLoader

    all_strats = StrategyLoader.load_all_strategies_including_disabled()
    result = []
    for s in all_strats:
        file_name = s.pop("_file_name", "")
        result.append({**s, "file_name": file_name})
    return result


def get_trading_config():
    return config.TRADING_OPTIONS


def get_cycle_stats():
    from database import get_open_positions

    positions = get_open_positions()
    initial_cap = config.TRADING_OPTIONS.get("initial_capital", 100000.0)
    total_invested = sum(p.get("total_investment", p["quantity"] * p["entry_price"]) for p in positions)
    total_mkt_val = sum(p.get("current_price", p["entry_price"]) * p["quantity"] for p in positions)
    cash_remaining = initial_cap - total_invested
    total_equity = total_mkt_val + cash_remaining
    pnl_pct = ((total_equity - initial_cap) / initial_cap) * 100 if initial_cap > 0 else 0

    return {
        "open_positions": len(positions),
        "total_invested": round(total_invested, 2),
        "cash_remaining": round(cash_remaining, 2),
        "total_equity": round(total_equity, 2),
        "pnl_pct": round(pnl_pct, 2),
        "initial_capital": initial_cap,
    }
