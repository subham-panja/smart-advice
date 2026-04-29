import os
import logging
from typing import Dict, List, Any
from dotenv import load_dotenv

load_dotenv() # Load env variables

# System & Threading
LIBRARY_MAX_THREADS = '1'
os.environ['OMP_NUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['MKL_NUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = LIBRARY_MAX_THREADS
SECRET_KEY = 'your_super_secret_key_here'
PERSIST_LOGGING = False

# Database configuration
MONGODB_HOST = os.getenv('MONGODB_HOST', '127.0.0.1')
MONGODB_PORT = int(os.getenv('MONGODB_PORT', '27017'))
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'super_advice')

# Global Parameters
MIN_RECOMMENDATION_SCORE = 0.50 # Minimum technical score to consider a buy
HISTORICAL_DATA_PERIOD = '5y' # Data lookback for backtesting
FILTERED_SYMBOLS_CACHE_HOURS = 72 # Cache duration for stock lists
FILTER_VALIDATION_PERIOD = '5y' # Period for validating filters


# Module Toggles
ANALYSIS_CONFIG = {
    'technical_analysis': True,
    'fundamental_analysis': True,
    'sentiment_analysis': False,
    'sector_analysis': True,
    'market_regime_detection': True,
    'backtesting': True,
    'risk_management': True,
}

# News & Sentiment
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
ALT_SENTIMENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
NEWS_COUNT = 20
NEWS_MAX_RETRIES = 3
NEWS_DATE_RANGE = '10d'

# Paths & Files
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_CACHE_FILE = os.path.join(BACKEND_DIR, 'data', 'nse_symbols.json')
SYMBOL_GROUPS_FILE = os.path.join(BACKEND_DIR, 'data', 'symbol_groups.json')

# Performance & Pipeline
MAX_WORKER_THREADS = 10 # Thread limit for parallel fetching
DATA_FETCH_THREADS = 16 # Threads specifically for data fetching
BATCH_SIZE = 8 # Batch size for processing
REQUEST_DELAY = 0.5 # Delay between requests
MAX_RETRIES = 1 # Max retries for failed requests
TIMEOUT_SECONDS = 10 # Request timeout
RATE_LIMIT_DELAY = 2.0 # Delay when rate limited
BACKOFF_MULTIPLIER = 1.5 # Backoff multiplier for retries
USE_MULTIPROCESSING_PIPELINE = True # Use multiple CPUs for analysis
NUM_WORKER_PROCESSES = 8 # Number of CPU cores to use

# External Integrations
USE_CHARTINK = True # Use Chartink for rapid stock screening
USE_SCREENER = False # Fallback screener integration

# Chartink Momentum-Breakout Query
CHARTINK_CONFIG = {
    'scan_clause': "( {cash} ( latest close > latest sma( close,50 ) and latest close > latest sma( close,200 ) and latest volume > latest sma( volume,20 ) * 2 and latest close > 1 day ago max( 20, high ) and latest rsi( 14 ) > 60 and latest close > latest open and latest close >= latest high * 0.98 and latest close > 20 and latest close < 50000 and latest volume > 100000 and market cap > 500 ) )",
    'cache_ttl_minutes': 30,
}

# Relative Strength (RS) Parameters
RS_CONFIG = {
    'benchmark_index': '^NSEI', # Nifty 50
    'period': 55, # Standard RS55 timeframe
    'threshold': 0.0, # Positive value means outperforming
}

# Market Regime Detection (Weinstein Rule)
MARKET_REGIME_CONFIG = {
    'index': '^NSEI', # Nifty 50
    'bull_market_rule': 'latest close > sma(200)',
    'pause_buying_if_bearish': True 
}

# NSE Options OI Filter Settings
OPTIONS_OI_CONFIG = {
    'enabled': True,
    'weight': 0.15, # Score bonus for positive OI signature
    'min_unwinding_pct': 5.0, # Min drop in Call OI for short squeeze
    'pcr_bullish_threshold': 1.2, # Bullish Put-Call Ratio
    'pcr_bearish_threshold': 0.7, # Bearish Put-Call Ratio
    'expiry_type': 'CURRENT', # CURRENT, NEXT, or MONTHLY
}

MONGODB_COLLECTIONS = {
    'recommended_shares': 'recommended_shares',
    'backtest_results': 'backtest_results',
    'trade_signals': 'trade_signals',
    'scan_runs': 'scan_runs',
    'analysis_snapshots': 'analysis_snapshots',
    'swing_gate_results': 'swing_gate_results'
}

TELEGRAM_CONFIG = {
    'enabled': True,
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'allowed_user_ids': [int(uid.strip()) for uid in os.getenv('TELEGRAM_ALLOWED_USER_IDS').split(',') if uid.strip()],
}

FIVEPAISA_CONFIG = {
    'app_name': os.getenv('FIVEPAISA_APP_NAME'),
    'app_source': os.getenv('FIVEPAISA_APP_SOURCE'),
    'user_id': os.getenv('FIVEPAISA_USER_ID'),
    'password': os.getenv('FIVEPAISA_PASSWORD'),
    'api_key': os.getenv('FIVEPAISA_API_KEY'),
    'encry_key': os.getenv('FIVEPAISA_ENCRY_KEY'),
    'client_code': os.getenv('FIVEPAISA_CLIENT_CODE'),
    'redirect_url': os.getenv('FIVEPAISA_REDIRECT_URL', 'https://xstream.5paisa.com/'),
    'access_token': os.getenv('FIVEPAISA_ACCESS_TOKEN'),
}

# Technical Engine (Weights & Strategies)
ANALYSIS_WEIGHTS = {
    'technical': 0.80, # Weight for technical momentum
    'fundamental': 0.10, # Weight for basic health filters
    'sentiment': 0.00, # Weight for news sentiment
    'sector': 0.10, # Weight for sector strength
}

STRATEGY_CONFIG = {
    'RSI_Overbought_Oversold': False, # Momentum continuation mode
    'MACD_Signal_Crossover': True, # Trend momentum confirmation
    'ADX_Trend_Strength': True, # Trend strength filter
    'On_Balance_Volume': True, # Volume-based confirmation
    'Bollinger_Band_Squeeze': True, # Volatility contraction pattern
    'Relative_Strength_Comparison': True, # Outperformance vs Nifty 50
    'Pocket_Pivot_Entry': True, # Early entry volume signature
}

RECOMMENDATION_THRESHOLDS = {
    'buy_combined': 0.40, # Minimum total score for a BUY
    'technical_minimum': 0.35, # Minimum technical score floor
    'fundamental_minimum': 0.05, # Minimum fundamental health floor
    'require_all_gates': True, # Allow trade if core trend is strong
    'min_risk_reward_ratio': 2, # Minimum RR requirement
}


# Legacy Filtering (Fallback) - Synchronized with Chartink Scan Clause
STOCK_FILTERING = {
    'min_volume': 100000,
    'min_price': 20.0,
    'max_price': 3000.0,
    'min_market_cap': 5000.0, # Cr
    'require_above_sma50': True,
    'require_above_sma200': True,
    'require_volume_spike': 2.0, # Increased from 1.2 to 2.0 for institutional footprint
    'require_rsi_above': 60.0, # Increased from 50 to 60 for momentum velocity
    'require_20day_breakout': False, # Disabled to allow pullbacks (Swing Strategy)
    'require_bullish_candle': True, # Close > Open
    'require_strong_close': 0.98, # Close >= High * 0.98
}

# Swing Trading Gates (Safety Filters)
SWING_TRADING_GATES = {
    'TREND_GATE': {
        'enabled': True,
        'params': {
            'adx_min': 20,
            'adx_max': 50,
            'macd_zero_buffer': 0.1,
            'sma_period': 200,
            'require_price_above_sma': True,
            'require_sma_stack': False, 
        }
    },
    'VOLATILITY_GATE': {
        'enabled': True,
        'params': {
            'min_percentile': 20, # Avoid "dead" stocks
            'max_percentile': 80, # Avoid "erratic" stocks
            'lookback_days': 100
        }
    },
    'VOLUME_GATE': {
        'enabled': True,
        'params': {
            'zscore_threshold': 0.2,
            'obv_trend_lookback': 10,
            'logic_operator': 'OR', # Either Z-Score spike OR OBV trend
        }
    },
    'MTF_GATE': {
        'enabled': True,
        'params': {
            'weekly_trend_check': True,
            'weekly_sma_fast': 10,
            'weekly_sma_slow': 30,
            'rsi_alignment_min': 60 # Increased from 50 for momentum velocity
        }
    }
}

# Exit Rules & Trade Management
SWING_PATTERNS = {
    'entry_patterns': [
        {
            'name': 'pullback_to_ema',
            'enabled': True,
            'ema_period': 20,
            'rsi_range': [40, 60],
            'bullish_candle_required': True
        },
        {
            'name': 'bollinger_squeeze_breakout',
            'enabled': True,
            'bb_period': 20,
            'bb_std': 2,
            'squeeze_threshold': 0.05,
            'retest_required': True,
            'max_squeeze_duration_days': 20 # Avoid dead stocks
        },
        {
            'name': 'macd_zero_cross',
            'enabled': True,
            'fast': 12,
            'slow': 26,
            'signal': 9,
            'above_zero_only': True
        },
        {
            'name': 'higher_low_structure',
            'enabled': True,
            'pivot_lookback': 5,
            'min_swings': 2
        },
        {
            'name': 'volatility_contraction',
            'enabled': True,
            'min_contractions': 2,
            'volume_dry_up_required': True
        }
    ],
    'exit_rules': {
        'atr_stop_multiplier': 1.5, # Reduced from 2.0 to 1.5 for tighter risk
        'target_1_atr': 3.0, # Increased from 2.0 to 3.0 to achieve 1:2 RR
        'target_2_atr': 4.5, # Scaled accordingly
        'trail_stop_atr': 3.0, # Trailing stop distance
        'time_stop_bars': 15, # Max hold time without target hit
        'breakeven_at_target_1': True, # Move SL to entry after T1
    }
}

# Portfolio & Risk Limits
RISK_MANAGEMENT = {
    'position_sizing': {
        'risk_per_trade': 0.01, # 1% risk per trade rule
        'max_position_pct': 0.10, # Max 20% of capital per stock
    },
    'portfolio_constraints': {
        'max_concurrent_positions': 5, # Max 5 active trades
        'daily_loss_limit': 0.03, # 3% portfolio loss limit per day
    },
    'pyramiding': {
        'enabled': True,
        'max_adds': 2,
        'trigger_step_atr': 1.5, # Add more every 1.5 ATR move
        'add_size_pct_steps': [0.5, 0.25], # Tapered adds: 50% then 25%
    }
}