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
EPISODIC_PIVOT_MODE = True

# Database configuration
MONGODB_HOST = os.getenv("MONGODB_HOST", "127.0.0.1")
MONGODB_PORT = int(os.getenv("MONGODB_PORT", "27017"))
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "super_advice")

# Global Parameters
HISTORICAL_DATA_PERIOD = "5y"  # Data lookback for backtesting


# Module Toggles
ANALYSIS_CONFIG = {
    "technical_analysis": True,
    "fundamental_analysis": True,
    "sentiment_analysis": False,
    "sector_analysis": True,
    "market_regime_detection": False,
    "backtesting": True,
    "risk_management": True,
}

# News & Sentiment
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
ALT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
NEWS_COUNT = 20
NEWS_MAX_RETRIES = 3
NEWS_DATE_RANGE = "10d"

# Paths & Files
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_CACHE_FILE = os.path.join(BACKEND_DIR, "data", "nse_symbols.json")

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

# Chartink Delayed Episodic Pivot (EP) Scanner - Tightened
CHARTINK_CONFIG = {
    "scan_clause": "( {cash} ( [0] 1 day ago volume > [0] 1 day ago sma(volume,50) * 3 or [0] 2 day ago volume > [0] 2 day ago sma(volume,50) * 3 or [0] 3 day ago volume > [0] 3 day ago sma(volume,50) * 3 or [0] 4 day ago volume > [0] 4 day ago sma(volume,50) * 3 or [0] 5 day ago volume > [0] 5 day ago sma(volume,50) * 3 ) and [0] volume < [0] sma(volume,20) and [0] close > [0] sma(close,10) and [0] market cap > 2000 and [0] rsi(14) > 50 )",
    "cache_ttl_minutes": 60,
}

# Relative Strength (RS) Parameters
RS_CONFIG = {
    "benchmark_index": "^NSEI",  # Nifty 50
    "period": 55,  # Standard RS55 timeframe
    "threshold": 0.0,  # Positive value means outperforming
}

# Market Regime Detection (Weinstein Rule)
MARKET_REGIME_CONFIG = {
    "index": "^NSEI",  # Nifty 50
    "bull_market_rule": "latest close > sma(50)",
    "pause_buying_if_bearish": True,
}

# NSE Options OI Filter Settings
OPTIONS_OI_CONFIG = {
    "enabled": True,
    "weight": 0.15,  # Score bonus for positive OI signature
    "min_unwinding_pct": 5.0,  # Min drop in Call OI for short squeeze
    "pcr_bullish_threshold": 1.2,  # Bullish Put-Call Ratio
    "pcr_bearish_threshold": 0.7,  # Bearish Put-Call Ratio
    "expiry_type": "CURRENT",  # CURRENT, NEXT, or MONTHLY
}

MONGODB_COLLECTIONS = {
    "recommended_shares": "recommended_shares",
    "backtest_results": "backtest_results",
    "trade_signals": "trade_signals",
    "scan_runs": "scan_runs",
    "analysis_snapshots": "analysis_snapshots",
    "swing_gate_results": "swing_gate_results",
}

TELEGRAM_CONFIG = {
    "enabled": True,
    "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
    "allowed_user_ids": [int(uid.strip()) for uid in os.getenv("TELEGRAM_ALLOWED_USER_IDS").split(",") if uid.strip()],
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

# Technical Engine (Weights & Strategies)
ANALYSIS_WEIGHTS = {
    "technical": 0.90,  # Weight for technical momentum
    "fundamental": 0.05,  # Weight for basic health filters
    "sentiment": 0.00,  # Weight for news sentiment
    "sector": 0.05,  # Weight for sector strength
}

STRATEGY_CONFIG = {
    "RSI_Overbought_Oversold": {"enabled": False, "period": 14, "upper": 70, "lower": 30},
    "MACD_Signal_Crossover": {"enabled": True, "fast": 12, "slow": 26, "signal": 9},
    "ADX_Trend_Strength": {"enabled": True, "period": 14, "threshold": 15},
    "On_Balance_Volume": {"enabled": True, "lookback": 20},
    "Bollinger_Band_Squeeze": {"enabled": True, "period": 20, "std": 2, "volatility_limit": 2.5},
    "Relative_Strength_Comparison": {"enabled": True, "period": 55, "threshold": 0.0},
    "Pocket_Pivot_Entry": {"enabled": True, "max_down_days": 10},
    "Volume_Breakout": {"enabled": True, "period": 20, "threshold": 2.5},
}

RECOMMENDATION_THRESHOLDS = {
    "buy_combined": 0.30,  # Lowered for EP Turnarounds
    "technical_minimum": 0.28,  # Floor lowered for EPs
    "fundamental_minimum": -0.50,  # Floor lowered to allow EP Turnarounds
    "require_all_gates": True,  # Allow trade if core trend is strong
    "min_risk_reward_ratio": 2,  # Minimum RR requirement
}


# NOTE: WE ARE NOT USING THE BELOW IF CHARTINK IS ENABLED (Legacy Fallback Only)
STOCK_FILTERING = {
    "min_volume": 100000,
    "min_price": 20.0,
    "max_price": 3000.0,
    "min_market_cap": 5000.0,  # Cr
    "require_above_sma50": True,
    "require_above_sma200": True,
    "require_volume_spike": 2.0,  # Increased from 1.2 to 2.0 for institutional footprint
    "require_rsi_above": 60.0,  # Increased from 50 to 60 for momentum velocity
    "require_20day_breakout": False,  # Disabled to allow pullbacks (Swing Strategy)
    "require_bullish_candle": True,  # Close > Open
    "require_strong_close": 0.98,  # Close >= High * 0.98
}

# Swing Trading Gates (Safety Filters)
SWING_TRADING_GATES = {
    "TREND_GATE": {
        "enabled": True,
        "params": {
            "adx_min": 15,
            "adx_max": 50,
            "macd_zero_buffer": 0.1,
            "sma_period": 50,  # Pivot out of neglect zone
            "require_price_above_sma": False,  # Allow Turnaround EPs
            "require_sma_stack": False,
            "adx_slope_check": True,
        },
    },
    "VOLATILITY_GATE": {
        "enabled": False,
        "params": {
            "min_percentile": 20,  # Avoid "dead" stocks
            "max_percentile": 80,  # Avoid "erratic" stocks
            "lookback_days": 100,
        },
    },
    "VOLUME_GATE": {
        "enabled": True,
        "params": {
            "zscore_threshold": 0.2,
            "obv_trend_lookback": 10,
            "logic_operator": "OR",  # Either Z-Score spike OR OBV trend
        },
    },
    "MTF_GATE": {
        "enabled": True,
        "params": {
            "weekly_trend_check": False,  # EPs often start with bad weekly charts
            "weekly_sma_fast": 10,
            "weekly_sma_slow": 30,
            "rsi_alignment_min": 60,  # Increased from 50 for momentum velocity
        },
    },
}

# Exit Rules & Trade Management
SWING_PATTERNS = {
    "entry_patterns": [
        {
            "name": "pullback_to_ema",
            "enabled": True,
            "ema_period": 10,  # Catch explosive momentum on 10-EMA
            "rsi_range": [40, 70],
            "bullish_candle_required": True,
        },
        {
            "name": "bollinger_squeeze_breakout",
            "enabled": True,
            "bb_period": 20,
            "bb_std": 2,
            "squeeze_threshold": 0.05,
            "retest_required": True,
            "max_squeeze_duration_days": 20,  # Avoid dead stocks
        },
        {"name": "macd_zero_cross", "enabled": True, "fast": 12, "slow": 26, "signal": 9, "above_zero_only": True},
        {"name": "higher_low_structure", "enabled": True, "pivot_lookback": 5, "min_swings": 2},
        {"name": "volatility_contraction", "enabled": True, "min_contractions": 2, "volume_dry_up_required": True},
    ],
    "exit_rules": {
        "targets": [
            {"name": "Target 1", "atr_multiplier": 3.0, "sell_percentage": 0.5},
            {"name": "Target 2", "atr_multiplier": 4.5, "sell_percentage": 1.0},
        ],
        "atr_stop_multiplier": 1.5,
        "trail_stop_atr": 3.0,
        "time_stop_bars": 15,
        "breakeven_at_target_1": True,
    },
}

# Portfolio & Risk Limits
RISK_MANAGEMENT = {
    "position_sizing": {
        "risk_per_trade": 0.01,
        "max_position_pct": 0.10,
    },
    "portfolio_constraints": {
        "max_concurrent_positions": 10,
        "daily_loss_limit": 0.03,
    },
    "pyramiding": {
        "enabled": True,
        "max_adds": 2,
        "steps": [
            {"name": "Add 1", "trigger_step_atr": 1.5, "add_size_pct": 0.5},
            {"name": "Add 2", "trigger_step_atr": 1.5, "add_size_pct": 0.25},
        ],
    },
}

# TRADING & EXECUTION OPTIONS
TRADING_OPTIONS = {
    "is_paper_trading": True,
    "initial_capital": 100000.0,
    "brokerage_charges": 0.0020,  # 0.20% per side
    "allow_multiple_positions_same_stock": False,
    "time_stop_days": 15,  # Exit if sideways for 15 days
    "auto_execute": True,
}
