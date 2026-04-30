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
        logger.info(
            f"Saving Recomm attempt for {res['symbol']} | is_recomm: {res.get('is_recommended')} | strength: {res.get('recommendation_strength')}"
        )
        if not res.get("is_recommended") or res.get("recommendation_strength") == "HOLD":
            return True
        try:
            db = get_mongodb()
            doc = {
                "symbol": res["symbol"],
                "company_name": res["company_name"],
                "technical_score": res["technical_score"],
                "combined_score": res["combined_score"],
                "recommendation_strength": res["recommendation_strength"],
                "reason": res["reason"],
                "buy_price": res["trade_plan"]["buy_price"],
                "sell_price": res["trade_plan"]["sell_price"],
                "stop_loss": res["trade_plan"]["stop_loss"],
                "backtest_metrics": res["backtest"]["combined_metrics"],
                "suggested_quantity": res["risk_management"]["position_size"],
                "allocation_pct": res["risk_management"]["allocation_pct"],
                "rr_ratio": res["risk_management"]["rr_ratio"],
                "strategy_name": res["strategy_name"],
                "recommendation_date": datetime.now(timezone.utc).replace(tzinfo=None),
            }
            res_db = db.recommended_shares.update_one({"symbol": res["symbol"]}, {"$set": doc}, upsert=True)
            return res_db.upserted_id or db.recommended_shares.find_one({"symbol": res["symbol"]}, {"_id": 1})["_id"]
        except Exception as e:
            logger.error(f"Save error for {res.get('symbol')}: {e}")
            raise e

    def save_backtest_results(self, res: Dict[str, Any]) -> bool:
        bt = res["backtest"]
        if bt["status"] != "completed":
            return False
        try:
            m = bt["combined_metrics"]
            insert_backtest_result(
                {
                    "symbol": res["symbol"],
                    "strategy_name": res["strategy_name"],
                    "period": "Overall",
                    "cagr": m["avg_cagr"],
                    "win_rate": m["avg_win_rate"],
                    "total_trades": m["total_trades"],
                }
            )
            return True
        except Exception as e:
            logger.error(f"BT Save error for {res.get('symbol')}: {e}")
            raise e

    def create_scan_run(self, cfg, macro=None):
        try:
            return (
                get_mongodb()
                .scan_runs.insert_one({"started_at": datetime.now(timezone.utc), "macro": macro})
                .inserted_id
            )
        except Exception as e:
            logger.error(f"Scan run creation error: {e}")
            raise e

    def complete_scan_run(self, rid, summary):
        try:
            get_mongodb().scan_runs.update_one(
                {"_id": rid}, {"$set": {"completed_at": datetime.now(timezone.utc), "summary": summary}}
            )
        except Exception as e:
            logger.error(f"Scan run completion error: {e}")
            raise e
