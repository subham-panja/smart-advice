import os
import logging

# Data Science Library Threading Limits (Must be set before imports)
LIBRARY_MAX_THREADS = '1'
os.environ['OMP_NUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['OPENBLAS_NUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['MKL_NUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['VECLIB_MAXIMUM_THREADS'] = LIBRARY_MAX_THREADS
os.environ['NUMEXPR_NUM_THREADS'] = LIBRARY_MAX_THREADS

# Flask configuration
SECRET_KEY = 'your_super_secret_key_here'

# Database configuration - MongoDB
MONGODB_HOST = os.getenv('MONGODB_HOST', '127.0.0.1')
MONGODB_PORT = int(os.getenv('MONGODB_PORT', '27017'))
MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'super_advice')

MONGODB_COLLECTIONS = {
    'recommended_shares': 'recommended_shares',
    'backtest_results': 'backtest_results',
    'analysis_snapshots': 'analysis_snapshots',
    'swing_gate_results': 'swing_gate_results',
    'trade_signals': 'trade_signals',
    'scan_runs': 'scan_runs',
}

# Strategy configuration - Core Swing Signals
STRATEGY_CONFIG = {
    'RSI_Overbought_Oversold': True,       # Oversold bounce entry
    'MACD_Signal_Crossover': True,         # Momentum confirmation
    'EMA_Crossover_12_26': True,           # Fast trend confirmation
    'ADX_Trend_Strength': True,            # Trend gate
    'On_Balance_Volume': True,             # Volume confirmation

    # Disabled or redundant strategies
    'MA_Crossover_50_200': False,
    'Bollinger_Band_Breakout': False,
    'ATR_Volatility': False,
    'SMA_Crossover_20_50': False,
    'Stochastic_Overbought_Oversold': False,
    'Multi_Timeframe_RSI': False,
    'Volume_Breakout': False,
    'Support_Resistance_Breakout': False,
    'Williams_Percent_R_Overbought_Oversold': False,
    'Fibonacci_Retracement': False,
    'Chart_Patterns': False,
    'Volume_Profile': False,
    'Volume_Price_Trend': False,
    'Momentum_Oscillator': False,
    'ROC_Rate_of_Change': False,
    'Keltner_Channels_Breakout': False,
    'DEMA_Crossover': False,
    'TEMA_Crossover': False,
    'RSI_Bullish_Divergence': False,
    'MACD_Zero_Line_Crossover': False,
    'Bollinger_Band_Squeeze': False,
    'Stochastic_K_D_Crossover': False,
    'DI_Crossover': False,
    'Ichimoku_Cloud_Breakout': False,
    'Ichimoku_Kijun_Tenkan_Crossover': False,
    'OBV_Bullish_Divergence': False,
    'Accumulation_Distribution_Line': False,
    'Candlestick_Hammer': False,
    'Candlestick_Bullish_Engulfing': False,
    'Candlestick_Doji': False,
    'Parabolic_SAR_Reversal': False,
    'CCI_Crossover': False,
    'Aroon_Oscillator': False,
    'Ultimate_Oscillator_Buy': False,
    'Money_Flow_Index_Oversold': False,
    'Price_Volume_Trend': False,
    'Chaikin_Oscillator': False,
    'Pivot_Points_Bounce': False,
    'Gap_Trading': False,
    'Channel_Trading': False,
    'Triple_Moving_Average': False,
    'Vortex_Indicator': False,
    'Commodity_Channel_Index': False,
    'Linear_Regression_Channel': False,
    'Elder_Ray_Index': False,
    'Keltner_Channel_Squeeze': False,
}

# Thresholds & Parameters
MIN_RECOMMENDATION_SCORE = 0.40
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
ALT_SENTIMENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
NEWS_COUNT = 20
NEWS_MAX_RETRIES = 3
NEWS_DATE_RANGE = '10d'

# Data fetching
HISTORICAL_DATA_PERIOD = '5y'
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
NSE_CACHE_FILE = os.path.join(BACKEND_DIR, 'data', 'nse_symbols.json')
SYMBOL_GROUPS_FILE = os.path.join(BACKEND_DIR, 'data', 'symbol_groups.json')
FILTERED_SYMBOLS_CACHE_HOURS = 72
FILTER_VALIDATION_PERIOD = '1y'

# Performance & Pipeline optimization
MAX_WORKER_THREADS = 10
BATCH_SIZE = 8
REQUEST_DELAY = 0.5
MAX_RETRIES = 1
TIMEOUT_SECONDS = 10
RATE_LIMIT_DELAY = 2.0
BACKOFF_MULTIPLIER = 1.5

USE_MULTIPROCESSING_PIPELINE = True
NUM_WORKER_PROCESSES = 8
DATA_FETCH_THREADS = 16

# Maintenance
DATA_PURGE_DAYS = 7
REMOVE_OLD_DATA_ON_EACH_RUN = False
PERSIST_LOGGING = True

# Analysis Weights
ANALYSIS_WEIGHTS = {
    'technical': 1.00,
    'fundamental': 0.00,
    'sentiment': 0.00,
    'sector': 0.00
}

# Recommendation Logic Thresholds
RECOMMENDATION_THRESHOLDS = {
    'strong_buy_combined': 0.65,
    'buy_combined': 0.50,
    'technical_strong_buy': 0.40,
    'sell_combined': -0.20,
    'sentiment_positive': 0.10,
    'sentiment_negative': -0.20,
    'sentiment_cap_positive': 0.30,
    'sentiment_cap_negative': -0.60,
    'min_backtest_return': 0.0,
    'technical_minimum': 0.35,
    'fundamental_minimum': 0.10,
    'volume_confirmation_required': True,
    'volume_confidence_threshold': 0.45,
    'market_trend_weight': 0.3,
    'require_all_gates': True,
    'min_risk_reward_ratio': 1.8,
    'sector_filter_enabled': False,
    'min_sector_score': -0.5
}

# Module Toggles
ANALYSIS_CONFIG = {
    'technical_analysis': True,
    'fundamental_analysis': False,
    'sentiment_analysis': False,
    'sector_analysis': False,
    'market_regime_detection': False,
    'market_microstructure': False,
    'alternative_data': False,
    'backtesting': True,
    'risk_management': True,
    'tca_analysis': False
}

# External Screener Integration
# When enabled, these replace the legacy per-stock yfinance filtering
# with a single server-side screener API call (much faster).
# If both are False, the system uses the existing STOCK_FILTERING criteria.
# If both are True, Chartink is preferred (faster) with Screener.in as fallback.
USE_CHARTINK = True
USE_SCREENER = False

# Chartink Screener Configuration
# NOTE: 'scan_clause' must be in Chartink's internal DSL format (lowercase),
#       wrapped in ( {cash} ( ... ) ). NOT the human-readable UI format.
CHARTINK_CONFIG = {
    'scan_clause': (
        "( {cash} ( "
        "latest close > latest sma( close,50 ) and "
        "latest close > latest sma( close,200 ) and "
        "latest volume > latest sma( volume,20 ) * 2 and "
        "latest close > 1 day ago max( 20, high ) and "
        "latest rsi( 14 ) > 60 and "
        "latest close > latest open and "
        "latest close >= latest high * 0.98"
        " ) )"
    ),
    'max_retries': 3,
    'retry_delay': 2.0,
    'cache_results': True,
    'cache_ttl_minutes': 30,
}

# Screener.in Configuration
SCREENER_CONFIG = {
    'query': (
        "Current price > 20 AND "
        "Current price < 50000 AND "
        "Market Capitalization > 5000 AND "
        "Volume > 100000 AND "
        "Current price > DMA 50 AND "
        "Current price > DMA 200"
    ),
    'username': os.getenv('SCREENER_USERNAME', ''),
    'password': os.getenv('SCREENER_PASSWORD', ''),
    'max_retries': 3,
    'retry_delay': 2.0,
    'fetch_all_pages': False,
}

# Stock Screening Criteria (Legacy – used when USE_CHARTINK and USE_SCREENER are both False)
STOCK_FILTERING = {
    'min_volume': 100000,
    'min_price': 20.0,
    'max_price': 50000.0,
    'min_market_cap': 5000000000,
    'min_historical_days': 250,
    'volume_lookback_days': 50,
    'exclude_delisted': True,
    'exclude_suspended': True,
    'min_delivery_percent': 30.0,
    'max_volatility_percentile': 80
}


# Swing Trading Strategy Gates
SWING_TRADING_GATES = {
    'trend_filter': {
        'enabled': True,
        'adx_period': 14,
        'adx_threshold': 20,
        'sma_period': 200,
        'price_above_sma': True
    },
    'volatility_gate': {
        'enabled': True,
        'atr_period': 14,
        'min_percentile': 20,
        'max_percentile': 80
    },
    'volume_confirmation': {
        'enabled': True,
        'obv_trend_periods': 10,
        'volume_zscore_threshold': 1.0,
        'require_either': True
    },
    'multi_timeframe': {
        'enabled': True,
        'weekly_trend_check': True,
        'weekly_sma_fast': 20,
        'weekly_sma_slow': 50
    }
}

# Swing Trading Patterns & Exit Rules
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
            'retest_required': True
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
        }
    ],
    'exit_rules': {
        'initial_stop_type': 'atr_based',
        'atr_stop_multiplier': 1.5,
        'target_1_atr': 1.5,
        'target_2_atr': 3.0,
        'trail_stop_atr': 2.0,
        'time_stop_bars': 15,
        'breakeven_at_target_1': True
    }
}

# Risk Management & Portfolio
RISK_MANAGEMENT = {
    'position_sizing': {
        'method': 'atr_based',
        'risk_per_trade': 0.01,
        'max_position_pct': 0.20
    },
    'portfolio_constraints': {
        'max_concurrent_positions': 5,
        'max_sector_concentration': 0.40,
        'daily_loss_limit': 0.03,
        'pause_on_limit_breach': True
    },
    'risk_reward': {
        'min_ratio': 1.5,
        'optimal_ratio': 2.5,
        'adjust_targets': False
    }
}
