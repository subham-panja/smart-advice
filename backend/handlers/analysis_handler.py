import logging
from datetime import datetime

from scripts.analyzer import analyze_stock

logger = logging.getLogger(__name__)

analysis_progress = {
    "status": "idle",
    "progress": 0,
    "total": 0,
    "current_stock": "",
    "recommendations": 0,
    "message": "",
    "start_time": None,
    "verbose": False,
}


def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


def analyze(symbol: str, app_config):
    return analyze_stock(symbol.upper(), app_config)


def get_recommendations():
    from database import get_recommended_shares_with_analytics

    recommendations_raw = get_recommended_shares_with_analytics()
    recommendations = []
    for rec in recommendations_raw:
        rec_dict = dict(rec)
        rec_dict.pop("_id", None)
        recommendations.append(
            {
                "symbol": rec_dict.get("symbol", ""),
                "company_name": rec_dict.get("company_name", rec_dict.get("symbol", "")),
                "technical_score": rec_dict.get("technical_score", 0),
                "fundamental_score": rec_dict.get("fundamental_score", 0),
                "sentiment_score": rec_dict.get("sentiment_score", 0),
                "combined_score": rec_dict.get("combined_score", 0),
                "is_recommended": rec_dict.get("is_recommended", False),
                "recommendation_strength": rec_dict.get("recommendation_strength", "HOLD"),
                "reason": rec_dict.get("reason", ""),
                "recommendation_date": rec_dict.get("recommendation_date", ""),
                "buy_price": rec_dict.get("buy_price", 0),
                "sell_price": rec_dict.get("sell_price", 0),
                "expected_return_percent": rec_dict.get("expected_return_percent", 0),
                "backtest_metrics": rec_dict.get("backtest_metrics", {}),
                "detailed_analysis": rec_dict.get("detailed_analysis", {}),
                "sector_analysis": rec_dict.get("sector_analysis", {}),
                "market_regime": rec_dict.get("market_regime", ""),
            }
        )
    return recommendations


def run_analysis_thread(req_data, app_context):
    try:
        with app_context:
            from run_analysis import AutomatedStockAnalysis

            analyzer = AutomatedStockAnalysis(verbose=req_data.get("verbose", False))
            analyzer.group_name = req_data["group"]
            analysis_progress["status"] = "running"
            analysis_progress["start_time"] = datetime.now()
            analyzer.run_analysis(
                max_stocks=req_data.get("max_stocks"),
                use_all_symbols=req_data.get("all", False),
            )
            analysis_progress["status"] = "completed"
    except Exception as e:
        analysis_progress["status"] = "error"
        analysis_progress["message"] = str(e)
