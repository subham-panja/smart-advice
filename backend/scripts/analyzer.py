import os
from typing import Any, Dict

import pandas as pd

from config import ANALYSIS_CONFIG, ANALYSIS_WEIGHTS, HISTORICAL_DATA_PERIOD, RECOMMENDATION_THRESHOLDS
from scripts.backtest_utils import BacktestUtils
from scripts.data_fetcher import get_all_nse_symbols, get_historical_data
from scripts.fundamental_analysis import FundamentalAnalysis
from scripts.market_regime_detection import MarketRegimeDetection
from scripts.risk_management import RiskManager
from scripts.strategy_evaluator import StrategyEvaluator
from scripts.swing_trading_signals import SwingTradingSignalAnalyzer
from scripts.trade_logic import TradeLogic
from utils.logger import setup_logging

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
logger = setup_logging()


class StockAnalyzer:
    """Combines technical, fundamental, and market analysis for stock selection."""

    def __init__(self):
        self.strategy_evaluator = StrategyEvaluator()
        self.fundamental_analyzer = FundamentalAnalysis()
        self.risk_manager = RiskManager()
        self.trade_logic = TradeLogic()
        self.swing_analyzer = SwingTradingSignalAnalyzer()
        self.backtest_utils = BacktestUtils()

    def analyze_stock(self, symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
        """Performs multi-pillar analysis by fetching data first."""
        hist = get_historical_data(
            symbol,
            app_config.get("HISTORICAL_DATA_PERIOD", HISTORICAL_DATA_PERIOD),
            fresh=app_config.get("FRESH_DATA", False),
        )
        return self.analyze_stock_with_data(symbol, "", hist, app_config)

    def analyze_stock_with_data(
        self, symbol: str, name: str, hist: pd.DataFrame, app_config: Dict[str, Any], index_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Performs multi-pillar analysis on provided historical data."""
        try:
            if not name:
                meta = get_all_nse_symbols()
                name = meta.get(symbol, symbol)

            res = {
                "symbol": symbol,
                "company_name": name,
                "technical_score": 0.0,
                "fundamental_score": 0.0,
                "is_recommended": False,
                "reason": [],
                "detailed_analysis": {},
            }
            if hist.empty:
                return {**res, "technical_score": -1.0, "reason": ["No data"]}

            # Technical & Swing
            tech = self.strategy_evaluator.evaluate_strategies(symbol, hist, index_data=index_data)
            res["technical_score"] = (tech["technical_score"] * 1.5) - 0.5
            swing = self.swing_analyzer.analyze_swing_opportunity(symbol, hist)
            res["swing_analysis"] = swing

            if RECOMMENDATION_THRESHOLDS.get("require_all_gates", True) and not swing.get("all_gates_passed", False):
                res["technical_score"] = min(res["technical_score"], 0.1)
                res["reason"].append("Gate failure")

            # Smart Money & Options
            from scripts.smart_money_tracker import SmartMoneyTracker

            dpct = SmartMoneyTracker().get_delivery_volume(symbol)
            if dpct > 40:
                res["technical_score"] += 0.05

            if ANALYSIS_CONFIG.get("options_oi", True):
                from config import OPTIONS_OI_CONFIG
                from scripts.options_analyzer import analyze_oi

                oi = analyze_oi(symbol)
                if oi.get("passed"):
                    res["technical_score"] += OPTIONS_OI_CONFIG.get("weight", 0.15)

            # Fundamental & Regime
            if ANALYSIS_CONFIG.get("fundamental_analysis", True):
                res["fundamental_score"] = self.fundamental_analyzer.perform_fundamental_analysis(symbol)

            if ANALYSIS_CONFIG.get("market_regime_detection", True):
                mrd = MarketRegimeDetection().get_simple_regime_check()
                if not mrd["passed"]:
                    res["technical_score"] = min(res["technical_score"], -0.5)

            # Combine & Trade Plan
            res = self._combine(res)
            res["trade_plan"] = self.trade_logic.analyze(symbol, hist)
            res["backtest"] = self.backtest_utils.perform_backtesting(symbol, hist)

            if res["is_recommended"]:
                res = self._combine(res, consider_bt=True)
                res["risk_management"] = self.risk_manager.calculate_risk_params(hist, hist["Close"].iloc[-1])

            res["reason"] = " ".join(res["reason"])
            return res
        except Exception as e:
            logger.error(f"Error {symbol}: {e}")
            return {"symbol": symbol, "is_recommended": False, "error": str(e)}

    def _combine(self, res: Dict[str, Any], consider_bt: bool = False) -> Dict[str, Any]:
        # Load weights from config
        t_w = ANALYSIS_WEIGHTS.get("technical", 0.90)
        f_w = ANALYSIS_WEIGHTS.get("fundamental", 0.05)
        s_w = ANALYSIS_WEIGHTS.get("sector", 0.05)

        # Calculate Sector Bonus (Placeholder: Using RS strength as a proxy for Sector Momentum)
        sector_score = 0.5  # Neutral default
        if res.get("swing_analysis", {}).get("gates", {}).get("trend"):
            sector_score = 0.8  # Bonus if in a trending context

        # Weighted Score Calculation
        score = (res["technical_score"] * t_w) + (res["fundamental_score"] * f_w) + (sector_score * s_w)
        res["combined_score"] = score

        # Floor Checks (Kicker Logic)
        tech_floor = RECOMMENDATION_THRESHOLDS.get("technical_minimum", 0.38)
        fund_floor = RECOMMENDATION_THRESHOLDS.get("fundamental_minimum", -0.5)

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
            cagr = res.get("backtest", {}).get("combined_metrics", {}).get("avg_cagr", 0)
            bt_ok = cagr >= RECOMMENDATION_THRESHOLDS.get("min_backtest_return", 0.0)

        buy_t = RECOMMENDATION_THRESHOLDS.get("buy_combined", 0.40)
        res["is_recommended"] = bool(score >= buy_t and bt_ok and passed_floors)
        res["recommendation_strength"] = "BUY" if res["is_recommended"] else "HOLD"

        logger.info(
            f"[{res.get('symbol')}] Result: {res['recommendation_strength']} | Score: {score:.2f} (Target: {buy_t}) | Tech: {res['technical_score']:.2f} | Fund: {res['fundamental_score']:.2f} | BT: {'Pass' if bt_ok else 'Fail'}"
        )
        return res


def analyze_stock(symbol: str, app_config: Dict[str, Any]) -> Dict[str, Any]:
    return StockAnalyzer().analyze_stock(symbol, app_config)
