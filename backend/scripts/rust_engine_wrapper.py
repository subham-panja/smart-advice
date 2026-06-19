"""
Rust Engine Wrapper
===================

Adapts the existing Python data format (pandas DataFrames, pre-computed signals)
to the Rust portfolio_engine's numpy-based interface.

Usage:
    from scripts.rust_engine_wrapper import RustPortfolioEngine
    engine = RustPortfolioEngine(strategy_config)
    result = engine.run(symbols_data, precomputed_signals, sim_start, sim_end)
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from portfolio_engine import run_backtest as _rust_run_backtest

    RUST_ENGINE_AVAILABLE = True
except ImportError:
    RUST_ENGINE_AVAILABLE = False
    logger.warning("Rust portfolio_engine not available. Install with: cd rust_engine && maturin develop --release")


class RustPortfolioEngine:
    """Drop-in replacement for PortfolioBacktestSession using Rust core."""

    def __init__(self, strategy_config: Dict[str, Any], capital_config: Optional[Dict] = None):
        if not RUST_ENGINE_AVAILABLE:
            raise RuntimeError("Rust engine not installed. Run: cd backend/rust_engine && maturin develop --release")

        self.strategy_config = strategy_config

        from config import PORTFOLIO_BACKTEST_CONFIG

        cfg = capital_config if capital_config else PORTFOLIO_BACKTEST_CONFIG

        self.initial_capital = cfg.get("initial_capital", 100000.0)
        self.brokerage = cfg.get("brokerage_charges", 0.0020)
        self.slippage = cfg.get("slippage_pct", 0.0005)

        risk_cfg = strategy_config.get("risk_management", {})
        exit_cfg = strategy_config.get("exit_rules", {})

        self._rust_config = {
            "risk_per_trade": risk_cfg.get("risk_per_trade_pct", 2.0) / 100.0,
            "max_position_pct": risk_cfg.get("max_position_pct", 10.0) / 100.0,
            "max_positions": risk_cfg.get("max_positions", 15),
            "brokerage": self.brokerage,
            "slippage": self.slippage,
            "stop_loss_pct": exit_cfg.get("stop_loss_pct", 8.0) / 100.0,
            "t1_sell_pct": exit_cfg.get("targets", [{}])[0].get("sell_percentage", 50) / 100.0,
            "t1_target_pct": exit_cfg.get("oneil_target_pct", 20.0),
            "time_stop_bars": exit_cfg.get("time_stop_bars", 20),
            "atr_stop_multiplier": exit_cfg.get("atr_stop_multiplier", 3.0),
            "trailing_stop_enabled": exit_cfg.get("trailing_stop_enabled", True),
        }

    def run(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        precomputed_signals: Dict[str, Dict],
        sim_start_date: Optional[pd.Timestamp] = None,
        sim_end_date: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Run backtest using Rust engine with pre-computed signals.

        Args:
            symbols_data: Dict[symbol, DataFrame] with OHLCV columns
            precomputed_signals: Dict[symbol, Dict[tz_naive_date, {score, swing_result}]]
            sim_start_date: Optional start date
            sim_end_date: Optional end date

        Returns:
            Dict matching PortfolioBacktestSession.run() output format
        """
        # 1. Build symbol index mapping
        symbol_list = sorted(symbols_data.keys())
        sym_to_idx = {sym: i for i, sym in enumerate(symbol_list)}

        # 2. Find common date range
        all_dates = set()
        for df in symbols_data.values():
            all_dates.update(df.index)
        common_dates = pd.DatetimeIndex(sorted(all_dates))

        if sim_start_date is not None:
            common_dates = common_dates[common_dates >= sim_start_date]
        if sim_end_date is not None:
            common_dates = common_dates[common_dates <= sim_end_date]

        if len(common_dates) < 60:
            return {"status": "failed", "reason": f"Insufficient trading days: {len(common_dates)}"}

        date_to_bar = {d: i for i, d in enumerate(common_dates)}
        num_bars = len(common_dates)

        # 3. Convert prices to numpy arrays (N_symbols x num_bars x 5)
        prices_list = []
        for sym in symbol_list:
            df = symbols_data[sym]
            arr = np.zeros((num_bars, 5), dtype=np.float64)
            for col_idx, col in enumerate(["Open", "High", "Low", "Close", "Volume"]):
                if col in df.columns:
                    aligned = df[col].reindex(common_dates)
                    arr[:, col_idx] = aligned.fillna(0).values
            prices_list.append(arr)

        # 4. Convert signals to Rust format: {symbol_idx: [(bar_idx, score), ...]}
        rust_signals = {}
        for sym, date_signals in precomputed_signals.items():
            if sym not in sym_to_idx:
                continue
            idx = sym_to_idx[sym]
            bar_scores = []
            for date_key, sig_data in date_signals.items():
                if date_key in date_to_bar:
                    bar_idx = date_to_bar[date_key]
                    score = sig_data.get("score", 0.0)
                    bar_scores.append((bar_idx, score))
            if bar_scores:
                rust_signals[idx] = bar_scores

        # 5. Run Rust engine
        logger.info(
            f"🦀 Rust engine: {len(symbol_list)} symbols, {num_bars} bars, " f"{len(rust_signals)} symbols with signals"
        )

        result = _rust_run_backtest(
            prices_list,
            rust_signals,
            self._rust_config,
            self.initial_capital,
        )

        return result
