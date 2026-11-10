from flask import Blueprint, current_app, jsonify, request

from handlers import analysis_handler

analysis_bp = Blueprint("analysis", __name__)


@analysis_bp.route("/")
def index():
    return jsonify(analysis_handler.health_check())


@analysis_bp.route("/analyze_stock/<symbol>", methods=["GET"])
def analyze_stock_endpoint(symbol):
    try:
        result = analysis_handler.analyze(symbol, current_app.config)
        return jsonify(result)
    except Exception as e:
        current_app.logger.error(f"Analysis endpoint failure for {symbol}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@analysis_bp.route("/recommendations", methods=["GET"])
def get_recommendations():
    try:
        recs = analysis_handler.get_recommendations()
        return jsonify({"status": "success", "count": len(recs), "recommendations": recs})
    except Exception as e:
        current_app.logger.error(f"Recommendations fetch failure: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@analysis_bp.route("/recommendations/<symbol>", methods=["PATCH"])
def update_recommendation_endpoint(symbol):
    try:
        data = request.get_json() or {}
        res = analysis_handler.update_recommendation(symbol, data)
        if not res:
            return jsonify({"status": "error", "error": f"Recommendation for {symbol} not found"}), 404
        return jsonify({"status": "success", "message": f"Recommendation {symbol} updated"})
    except Exception as e:
        current_app.logger.error(f"Recommendation update failure for {symbol}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@analysis_bp.route("/stock-candles/<symbol>", methods=["GET"])
def get_stock_candles(symbol):
    """Return historical candles for chart display, using local parquet cache or fetcher."""
    import os

    import pandas as pd

    import config

    try:
        clean_sym = symbol.replace(".0", "").replace(".NS", "").strip()
        days = request.args.get("days", default=35, type=int)
        cache_dir = config.DATA_CACHE_CONFIG.get("cache_dir", os.path.join(config.BACKEND_DIR, "data", "historical"))
        fpath = os.path.join(cache_dir, f"{clean_sym}.parquet")

        df = None
        if os.path.exists(fpath):
            try:
                df = pd.read_parquet(fpath)
            except Exception:
                pass

        if df is None or df.empty:
            # Fallback to general fetch
            from utils.data_cache import fetch_historical_data_cached

            df = fetch_historical_data_cached(clean_sym, period="3mo")

        if df is None or df.empty:
            return jsonify({"status": "error", "error": f"No candle data for {symbol}"}), 404

        # Clean index timezone if present
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)

        tail_df = df.tail(days)
        candles = []
        for dt, row in tail_df.iterrows():
            candles.append(
                {
                    "date": dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10],
                    "open": round(float(row.get("Open", 0)), 2),
                    "high": round(float(row.get("High", 0)), 2),
                    "low": round(float(row.get("Low", 0)), 2),
                    "close": round(float(row.get("Close", 0)), 2),
                    "volume": int(row.get("Volume", 0)),
                }
            )

        return jsonify({"status": "success", "symbol": clean_sym, "candles": candles})
    except Exception as e:
        current_app.logger.error(f"Stock candles fetch error for {symbol}: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@analysis_bp.route("/trigger-analysis", methods=["POST"])
def trigger_analysis():
    import threading

    try:
        req_data = request.get_json()
        if not req_data:
            raise ValueError("Missing analysis configuration payload.")

        req_data["group"]  # mandatory field check

        thread = threading.Thread(
            target=analysis_handler.run_analysis_thread,
            args=(req_data, current_app.app_context()),
        )
        thread.daemon = True
        thread.start()

        return jsonify({"status": "success", "message": "Analysis started."})
    except Exception as e:
        current_app.logger.error(f"Analysis trigger failure: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
