import logging
import os
from typing import Any, Dict

import pandas as pd

from scripts.backtest_utils import BacktestUtils
from scripts.data_fetcher import get_all_nse_symbols
from scripts.fundamental_analysis import FundamentalAnalysis
from scripts.risk_management import RiskManager
from scripts.strategy_evaluator import StrategyEvaluator
from scripts.swing_trading_signals import SwingTradingSignalAnalyzer
from scripts.trade_logic import TradeLogic

os.environ.update(
    {
        k: "1"
        for k in [
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]
    }
)
logger = logging.getLogger(__name__)


class StockAnalyzer:
    """Combines technical, fundamental, and market analysis for stock selection."""

    def __init__(self):
        # Instances will be updated per call with the correct strategy config
        self.strategy_evaluator = None
        self.fundamental_analyzer = FundamentalAnalysis()
        self.risk_manager = RiskManager()
        self.trade_logic = TradeLogic()
        self.swing_analyzer = SwingTradingSignalAnalyzer()
        self.backtest_utils = BacktestUtils()

    def analyze_stock_with_data(
        self, symbol: str, name: str, hist: pd.DataFrame, app_config: Dict[str, Any], index_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Performs multi-pillar analysis on provided historical data with strict configuration."""
        try:
            if not name:
                meta = get_all_nse_symbols()
                name = meta.get(symbol, symbol)

            res = {
                "symbol": symbol,
                "company_name": name,
                "strategy_name": app_config["name"],
                "technical_score": 0.0,
                "fundamental_score": 0.0,
                "is_recommended": False,
                "reason": [],
                "detailed_analysis": {},
            }

            if hist.empty:
                raise ValueError(f"Empty data for {symbol}")

            # Technical & Swing
            if self.strategy_evaluator is None:
                self.strategy_evaluator = StrategyEvaluator(app_config)

            tech = self.strategy_evaluator.evaluate_strategies(
                symbol, hist, app_config=app_config, index_data=index_data
            )
            res["technical_score"] = tech["technical_score"]
            swing = self.swing_analyzer.analyze_swing_opportunity(symbol, hist, strategy_config=app_config)
            res["swing_analysis"] = swing

            rec_thresholds = app_config["recommendation_thresholds"]
            if rec_thresholds["require_all_gates"] and not swing["all_gates_passed"]:
                res["technical_score"] = min(res["technical_score"], 0.1)
                res["reason"].append("Gate failure")

            # Smart Money & Options
            from scripts.smart_money_tracker import SmartMoneyTracker

            dpct = SmartMoneyTracker().get_delivery_volume(symbol)
            if dpct > 40:
                res["technical_score"] += 0.05

            ana_cfg = app_config["analysis_config"]
            if ana_cfg["options_oi"]:
                from scripts.options_analyzer import analyze_oi

                oi_cfg = app_config["options_oi_config"]
                oi = analyze_oi(symbol, config=oi_cfg)
                if oi["passed"]:
                    res["technical_score"] += oi_cfg["weight"]

            # Fundamental
            if ana_cfg["fundamental_analysis"]:
                res["fundamental_score"] = self.fundamental_analyzer.perform_fundamental_analysis(symbol)

            # Combine & Trade Plan
            res = self._combine(res, app_config)
            res["trade_plan"] = self.trade_logic.analyze(symbol, hist, app_config=app_config)
            res["backtest"] = self.backtest_utils.perform_backtesting(symbol, hist, app_config=app_config)

            if res["is_recommended"]:
                res = self._combine(res, app_config, consider_bt=True)
                res["risk_management"] = self.risk_manager.calculate_risk_params(
                    hist, hist["Close"].iloc[-1], app_config=app_config
                )

            res["reason"] = " ".join(res["reason"])
            return res
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {e}")
            raise e

    def _combine(self, res: Dict[str, Any], app_config: Dict[str, Any], consider_bt: bool = False) -> Dict[str, Any]:
        weights = app_config["analysis_weights"]
        t_w = weights["technical"]
        f_w = weights["fundamental"]
        s_w = weights["sector"]

        # Calculate Sector Bonus (Strict)
        sector_score = 0.5
        if res["swing_analysis"]["gates"]["trend"]:
            sector_score = 0.8

        # Weighted Score Calculation
        score = (res["technical_score"] * t_w) + (res["fundamental_score"] * f_w) + (sector_score * s_w)
        res["combined_score"] = score

        # Floor Checks
        rec_thresholds = app_config["recommendation_thresholds"]
        tech_floor = rec_thresholds["technical_minimum"]
        fund_floor = rec_thresholds["fundamental_minimum"]

        passed_floors = True
        if res["technical_score"] < tech_floor:
            res["reason"].append(f"Low Tech Score ({res['technical_score']:.2f})")
            passed_floors = False
        if res["fundamental_score"] < fund_floor:
            res["reason"].append(f"Low Fund Score ({res['fundamental_score']:.2f})")
            passed_floors = False

        # Backtest Check
        bt_ok = True
        if consider_bt:
            # Note: backtest_utils must be updated if it uses defaults
            m = res["backtest"]["combined_metrics"]
            # Check for min_backtest_return in threshold, if missing it will crash as requested
            bt_ok = m["avg_cagr"] >= rec_thresholds["min_backtest_return"]

        buy_t = rec_thresholds["buy_combined"]
        res["is_recommended"] = bool(score >= buy_t and bt_ok and passed_floors)
        res["recommendation_strength"] = "BUY" if res["is_recommended"] else "HOLD"

        logger.warning(
            f"[{res['symbol']}] Result: {res['recommendation_strength']} | Score: {score:.2f} (Target: {buy_t}) | Tech: {res['technical_score']:.2f} | Fund: {res['fundamental_score']:.2f} | BT: {'Pass' if bt_ok else 'Fail'}"
        )
        return res


def analyze_stock(symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
    return StockAnalyzer().analyze_stock_with_data(symbol, "", None, app_config)
