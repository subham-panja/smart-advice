import os

from dotenv import load_dotenv

load_dotenv()  # Load env variables

# System & Threading
LIBRARY_MAX_THREADS = "1"
os.environ["OMP_NUM_THREADS"] = LIBRARY_MAX_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = LIBRARY_MAX_THREADS
os.environ["MKL_NUM_THREADS"] = LIBRARY_MAX_THREADS
os.environ["VECLIB_MAXIMUM_THREADS"] = LIBRARY_MAX_THREADS
os.environ["NUMEXPR_NUM_THREADS"] = LIBRARY_MAX_THREADS
SECRET_KEY = "your_super_secret_key_here"
PERSIST_LOGGING = False
VERBOSE_LOGGING = False

# Data Purge
DATA_PURGE_DAYS = 7  # Purge recommendations and backtest results older than N days


# Database configuration
MONGODB_HOST = os.getenv("MONGODB_HOST", "127.0.0.1")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "super_advice")

# Paths & Files
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_CACHE_FILE = os.path.join(BACKEND_DIR, "data", "nse_symbols.json")
STRATEGIES_DIR = os.path.join(BACKEND_DIR, "strategies")


# Performance & Pipeline
MAX_WORKER_THREADS = 10  # Thread limit for parallel fetching
DATA_FETCH_THREADS = 4  # Reduced to avoid rate limiting
BATCH_SIZE = 8  # Batch size for processing
REQUEST_DELAY = 1.0  # Increased delay between requests
MAX_RETRIES = 1  # Max retries for failed requests
TIMEOUT_SECONDS = 10  # Request timeout
RATE_LIMIT_DELAY = 2.0  # Delay when rate limited
BACKOFF_MULTIPLIER = 1.5  # Backoff multiplier for retries
USE_MULTIPROCESSING_PIPELINE = True  # Re-enabled for high-volume EP scans
NUM_WORKER_PROCESSES = 8  # Using 8 cores for parallel analysis

# External Integrations
USE_CHARTINK = True  # Use Chartink for rapid stock screening
USE_SCREENER = False  # Fallback screener integration

MONGODB_COLLECTIONS = {
    "recommended_shares": "recommended_shares",
    "backtest_results": "backtest_results",
    "trade_signals": "trade_signals",
    "scan_runs": "scan_runs",
    "analysis_snapshots": "analysis_snapshots",
    "swing_gate_results": "swing_gate_results",
    "positions": "positions",
    "backtest_sessions": "backtest_sessions",
    "portfolio_backtest_trades": "portfolio_backtest_trades",
    "portfolio_backtest_daily_snapshots": "portfolio_backtest_snapshots",
}

TELEGRAM_CONFIG = {
    "enabled": True,
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "allowed_user_ids": [
        int(uid.strip()) for uid in (os.getenv("TELEGRAM_ALLOWED_USER_IDS") or "").split(",") if uid.strip()
    ],
}

FIVEPAISA_CONFIG = {
    "app_name": os.getenv("FIVEPAISA_APP_NAME"),
    "app_source": os.getenv("FIVEPAISA_APP_SOURCE"),
    "user_id": os.getenv("FIVEPAISA_USER_ID"),
    "password": os.getenv("FIVEPAISA_PASSWORD"),
    "api_key": os.getenv("FIVEPAISA_API_KEY"),
    "encry_key": os.getenv("FIVEPAISA_ENCRY_KEY"),
    "client_code": os.getenv("FIVEPAISA_CLIENT_CODE"),
    "redirect_url": os.getenv("FIVEPAISA_REDIRECT_URL"),
    "access_token": os.getenv("FIVEPAISA_ACCESS_TOKEN"),
}

# TRADING & EXECUTION OPTIONS (app-level, not strategy-specific)
TRADING_OPTIONS = {
    "is_paper_trading": True,
    "initial_capital": 100000.0,
    "brokerage_charges": 0.0020,
    "allow_multiple_positions_same_stock": False,
    "time_stop_days": 15,
    "time_stop_min_pnl_pct": 2.0,
    "auto_execute": True,
    "circuit_breaker": False,
}

# PORTFOLIO BACKTEST CONFIGURATION (app-level defaults)
PORTFOLIO_BACKTEST_CONFIG = {
    "enabled": True,
    "initial_capital": 100000.0,
    "brokerage_charges": 0.0020,
    "ranking_method": "combined_score",
    "save_daily_snapshots": True,
    "same_day_cash_recycling": True,
    "force_close_delisted": True,
    "auto_run_on_cycle": True,
    "auto_run_max_stocks": 1000,
    "walk_forward": {
        "enabled": False,
        "window_days": 180,
        "step_days": 90,
        "mc_iterations": 10,
        "sample_pct": 0.7,
    },
}

# DATA CACHING CONFIGURATION
DATA_CACHE_CONFIG = {
    "enabled": True,
    "format": "parquet",
    "cache_dir": os.path.join(BACKEND_DIR, "data", "historical"),
    "staleness_hours": 24,
    "periods": {
        "analysis": "2y",
        "backtest": "5y",
        "portfolio_backtest": "5y",
        "monitoring": "1mo",
    },
}

# GLOBAL POSITION MANAGEMENT
PYRAMID_COUNTS_AS_NEW_POSITION = False
