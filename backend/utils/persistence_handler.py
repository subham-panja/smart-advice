import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from database import get_mongodb, insert_backtest_result

logger = logging.getLogger(__name__)


class PersistenceHandler:
    """Handles database persistence for recommendations and backtests."""

    def __init__(self, app=None):
        self.app = app

    def clear_old_data(self, days: int = 7):
        try:
            db = get_mongodb()
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            db.recommended_shares.delete_many({"recommendation_date": {"$lt": cutoff}})
            db.backtest_results.delete_many({"created_at": {"$lt": cutoff}})
        except Exception as e:
            logger.error(f"Clear error: {e}")

    def save_recommendation(self, res: Dict[str, Any]) -> bool:
        if not res.get("is_recommended") or res.get("recommendation_strength") == "HOLD":
            return True
        try:
            db = get_mongodb()
            doc = {
                "symbol": res["symbol"],
                "company_name": res.get("company_name", ""),
                "technical_score": res.get("technical_score", 0),
                "combined_score": res.get("combined_score", 0),
                "recommendation_strength": res.get("recommendation_strength"),
                "reason": res.get("reason", ""),
                "buy_price": res.get("trade_plan", {}).get("buy_price", res.get("buy_price")),
                "sell_price": res.get("trade_plan", {}).get("sell_price", res.get("sell_price")),
                "stop_loss": res.get("trade_plan", {}).get("stop_loss", 0),
                "backtest_metrics": res.get("backtest", {}).get("combined_metrics", {}),
                "recommendation_date": datetime.now(timezone.utc),
            }
            res_db = db.recommended_shares.update_one({"symbol": res["symbol"]}, {"$set": doc}, upsert=True)
            return res_db.upserted_id or db.recommended_shares.find_one({"symbol": res["symbol"]}, {"_id": 1})["_id"]
        except Exception as e:
            logger.error(f"Save error: {e}")
            return None

    def save_backtest_results(self, res: Dict[str, Any]) -> bool:
        bt = res.get("backtest", {})
        if bt.get("status") != "completed":
            return False
        try:
            m = bt.get("combined_metrics", {})
            insert_backtest_result(
                {
                    "symbol": res["symbol"],
                    "period": "Overall",
                    "cagr": m.get("avg_cagr", 0),
                    "win_rate": m.get("avg_win_rate", 0),
                    "total_trades": m.get("total_trades", 0),
                }
            )
            return True
        except Exception as e:
            logger.error(f"BT Save error: {e}")
            return False

    def create_scan_run(self, cfg, macro=None):
        try:
            return (
                get_mongodb()
                .scan_runs.insert_one({"started_at": datetime.now(timezone.utc), "macro": macro})
                .inserted_id
            )
        except Exception as e:
            logger.error(f"Scan run creation error: {e}")
            return None

    def complete_scan_run(self, rid, summary):
        try:
            get_mongodb().scan_runs.update_one(
                {"_id": rid}, {"$set": {"completed_at": datetime.now(timezone.utc), "summary": summary}}
            )
        except Exception as e:
            logger.error(f"Scan run completion error: {e}")

    def save_analysis_snapshot(self, res, scan_run_id=None):
        pass

    def save_swing_gate_results(self, symbol, gates, scan_run_id, recommended):
        pass

    def save_trade_signal(self, symbol, signal, scan_run_id):
        pass
