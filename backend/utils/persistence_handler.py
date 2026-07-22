import logging
from datetime import timedelta, timezone
from typing import Any, Dict

from database import get_mongodb
from utils.trading_clock import trading_now

logger = logging.getLogger(__name__)


class PersistenceHandler:
    """Handles database persistence for recommendations and portfolio backtest sessions."""

    def __init__(self, app=None):
        self.app = app

    def clear_old_data(self, days: int = None):
        if days is None:
            from config import DATA_PURGE_DAYS

            days = DATA_PURGE_DAYS
        try:
            db = get_mongodb()
            cutoff = trading_now(timezone.utc) - timedelta(days=days)
            db.recommended_shares.delete_many({"recommendation_date": {"$lt": cutoff}})
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
            # Fetch the filtered_stock_id
            fs = db.filtered_stocks.find_one({"symbol": res["symbol"]}, sort=[("detected_at", -1)])
            fs_id = fs["_id"] if fs else None

            doc = {
                "symbol": res["symbol"],
                "filtered_stock_id": fs_id,
                "company_name": res["company_name"],
                "technical_score": res["technical_score"],
                "combined_score": res["combined_score"],
                "recommendation_strength": res["recommendation_strength"],
                "reason": res["reason"],
                "buy_price": res["trade_plan"]["buy_price"],
                "sell_price": res["trade_plan"]["sell_price"],
                "stop_loss": res["trade_plan"]["stop_loss"],
                "suggested_quantity": res["risk_management"]["position_size"],
                "allocation_pct": res["risk_management"]["allocation_pct"],
                "rr_ratio": res["risk_management"]["rr_ratio"],
                "strategy_name": res["strategy_name"],
                "recommendation_date": trading_now(timezone.utc)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .replace(tzinfo=None),
                "created_at": trading_now(timezone.utc).replace(tzinfo=None),
                "updated_at": trading_now(timezone.utc).replace(tzinfo=None),
            }
            rec_date = doc["recommendation_date"]
            db.recommended_shares.update_one(
                {"symbol": res["symbol"], "recommendation_date": rec_date},
                {"$set": doc},
                upsert=True,
            )
            return True
        except Exception as e:
            logger.error(f"Save error for {res.get('symbol')}: {e}")
            raise e

    def save_filtered_stock(self, symbol: str, strategy_name: str, scan_id: Any = None) -> Any:
        """Saves a stock that passed initial filters and returns its ID."""
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            res = db.filtered_stocks.insert_one(
                {
                    "symbol": symbol,
                    "strategy_name": strategy_name,
                    "scan_id": scan_id,
                    "detected_at": now,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            return res.inserted_id
        except Exception as e:
            logger.error(f"Error saving filtered stock {symbol}: {e}")
            return None

    def create_scan_run(self, cfg, macro=None):
        try:
            return (
                get_mongodb()
                .scan_runs.insert_one({"started_at": trading_now(timezone.utc), "macro": macro})
                .inserted_id
            )
        except Exception as e:
            logger.error(f"Scan run creation error: {e}")
            raise e

    def complete_scan_run(self, rid, summary):
        try:
            get_mongodb().scan_runs.update_one(
                {"_id": rid}, {"$set": {"completed_at": trading_now(timezone.utc), "summary": summary}}
            )
        except Exception as e:
            logger.error(f"Scan run completion error: {e}")
            raise e

    # ------------------------------------------------------------------
    # Portfolio Backtest Session Persistence
    # ------------------------------------------------------------------

    def create_backtest_session(
        self, strategy_name: str, strategy_config: dict, capital_config: dict, symbols: list
    ) -> Any:
        """Creates a portfolio backtest session and returns its ID."""
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            doc = {
                "session_type": "portfolio",
                "session_name": f"{strategy_name}_Portfolio_{now.strftime('%Y_%m_%d_%H%M')}",
                "strategy_name": strategy_name,
                "strategy_config_snapshot": strategy_config,
                "capital_config": capital_config,
                "symbols_tested": symbols,
                "total_symbols": len(symbols),
                "status": "running",
                "date_range": {"start_date": None, "end_date": None},
                "summary_metrics": {},
                "created_at": now,
                "started_at": now,
            }
            return db.backtest_sessions.insert_one(doc).inserted_id
        except Exception as e:
            logger.error(f"Backtest session creation error: {e}")
            raise e

    def save_portfolio_backtest_trades(self, session_id: Any, trades: list) -> bool:
        """Bulk saves portfolio backtest trades."""
        if not trades:
            return True
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            trade_docs = []
            for t in trades:
                # Convert dataclass objects to dicts if needed
                if hasattr(t, "__dataclass_fields__"):
                    from dataclasses import asdict

                    doc = asdict(t)
                elif isinstance(t, dict):
                    doc = t
                else:
                    doc = dict(t)
                doc["session_id"] = session_id
                doc.setdefault("created_at", now)
                doc.setdefault("updated_at", now)
                trade_docs.append(doc)
            db.portfolio_backtest_trades.insert_many(trade_docs)
            return True
        except Exception as e:
            logger.error(f"Portfolio trades save error: {e}")
            raise e

    def save_portfolio_backtest_snapshots(self, session_id: Any, snapshots: list) -> bool:
        """Bulk saves daily portfolio snapshots."""
        if not snapshots:
            return True
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            for s in snapshots:
                s["session_id"] = session_id
                s.setdefault("created_at", now)
                s.setdefault("updated_at", now)
            db.portfolio_backtest_daily_snapshots.insert_many(snapshots)
            return True
        except Exception as e:
            logger.error(f"Portfolio snapshots save error: {e}")
            raise e

    def complete_backtest_session(self, session_id: Any, summary: dict, date_range: dict = None):
        """Marks a backtest session as completed with summary metrics."""
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            update = {
                "status": "completed",
                "completed_at": now,
                "summary_metrics": summary,
            }
            if date_range:
                update["date_range"] = date_range
            db.backtest_sessions.update_one({"_id": session_id}, {"$set": update})
        except Exception as e:
            logger.error(f"Backtest session completion error: {e}")
            raise e

    def save_ultimate_backtest_phases(self, session_id: Any, phases: dict) -> bool:
        """Save validation, walk-forward, stress tests, diagnostics, and confidence score.

        Stores all Phase 2-6 results as a nested 'ultimate_phases' field on the
        backtest_sessions document so the full analysis is persisted for later review.
        """

        def sanitize(d):
            import numpy as np

            if isinstance(d, dict):
                return {str(k): sanitize(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [sanitize(x) for x in d]
            elif isinstance(d, (np.bool_, bool)):
                return bool(d)
            elif isinstance(d, (np.integer, int)):
                return int(d)
            elif isinstance(d, (np.floating, float)):
                return float(d)
            return d

        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            safe_phases = sanitize(phases)
            db.backtest_sessions.update_one(
                {"_id": session_id},
                {"$set": {"ultimate_phases": safe_phases, "updated_at": now}},
            )
            return True
        except Exception as e:
            logger.error(f"Ultimate phases save error: {e}")
            return False

    # ------------------------------------------------------------------
    # Walk-Forward Backtest Session Persistence
    # ------------------------------------------------------------------

    def create_walk_forward_session(
        self,
        strategy_name: str,
        strategy_config: dict,
        capital_config: dict,
        windows: list,
        mc_iterations: int,
    ) -> Any:
        """Creates a walk-forward backtest session and returns its ID."""
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            total_runs = len(windows) * mc_iterations
            doc = {
                "session_type": "walk_forward",
                "session_name": f"{strategy_name}_WalkForward_{now:%Y_%m_%d_%H%M}",
                "strategy_name": strategy_name,
                "strategy_config_snapshot": strategy_config,
                "capital_config": capital_config,
                "total_windows": len(windows),
                "mc_iterations_per_window": mc_iterations,
                "total_runs": total_runs,
                "window_definitions": [
                    {"window": i + 1, "start": str(ws), "end": str(we)} for i, (ws, we) in enumerate(windows)
                ],
                "status": "running",
                "progress": {
                    "current_window": 0,
                    "total_windows": len(windows),
                    "completed_runs": 0,
                    "total_runs": total_runs,
                    "pct_complete": 0.0,
                    "elapsed_seconds": 0,
                    "estimated_remaining_seconds": None,
                    "current_cagr_mean": None,
                },
                "summary_metrics": {},
                "created_at": now,
                "started_at": now,
            }
            return db.backtest_sessions.insert_one(doc).inserted_id
        except Exception as e:
            logger.error(f"Walk-forward session creation error: {e}")
            raise e

    def save_walk_forward_run(
        self,
        session_id: Any,
        window: int,
        mc_iteration: int,
        symbols_count: int,
        sampled_symbols: list,
        result: dict,
    ) -> bool:
        """Saves an individual MC run result to walk_forward_results collection."""
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            doc = {
                "session_id": session_id,
                "window": window,
                "mc_iteration": mc_iteration,
                "symbols_count": symbols_count,
                "sampled_symbols": sampled_symbols,
                "cagr": result.get("cagr"),
                "total_return": result.get("total_return"),
                "max_drawdown": result.get("max_drawdown"),
                "sharpe": result.get("sharpe"),
                "total_trades": result.get("total_trades"),
                "win_rate": result.get("win_rate"),
                "profit_factor": result.get("profit_factor"),
                "status": result.get("status", "success"),
                "error": result.get("error"),
                "created_at": now,
            }
            db.walk_forward_results.insert_one(doc)
            return True
        except Exception as e:
            logger.error(f"Walk-forward run save error: {e}")
            return False

    def update_walk_forward_progress(
        self,
        session_id: Any,
        current_window: int,
        completed_runs: int,
        total_runs: int,
        elapsed: float,
        cagrs_so_far: list,
    ) -> bool:
        """Updates progress field on walk-forward session document."""
        try:
            db = get_mongodb()
            pct_complete = round(completed_runs / total_runs * 100, 1) if total_runs > 0 else 0.0
            cagr_mean = round(sum(cagrs_so_far) / len(cagrs_so_far), 2) if cagrs_so_far else None

            update = {
                "progress.current_window": current_window,
                "progress.completed_runs": completed_runs,
                "progress.pct_complete": pct_complete,
                "progress.elapsed_seconds": round(elapsed, 1),
                "progress.current_cagr_mean": cagr_mean,
                "updated_at": trading_now(timezone.utc).replace(tzinfo=None),
            }

            if completed_runs > 0 and elapsed > 0:
                remaining = total_runs - completed_runs
                update["progress.estimated_remaining_seconds"] = round(elapsed / completed_runs * remaining)

            db.backtest_sessions.update_one({"_id": session_id}, {"$set": update})
            return True
        except Exception as e:
            logger.error(f"Walk-forward progress update error: {e}")
            return False

    def complete_walk_forward_session(
        self,
        session_id: Any,
        aggregated_metrics: dict,
        duration: float,
    ) -> bool:
        """Marks walk-forward session as completed with aggregated metrics."""
        try:
            db = get_mongodb()
            now = trading_now(timezone.utc).replace(tzinfo=None)
            update = {
                "status": "completed",
                "completed_at": now,
                "summary_metrics": aggregated_metrics,
                "duration_seconds": round(duration, 1),
                "progress.pct_complete": 100.0,
                "updated_at": now,
            }
            db.backtest_sessions.update_one({"_id": session_id}, {"$set": update})
            return True
        except Exception as e:
            logger.error(f"Walk-forward session completion error: {e}")
            return False
