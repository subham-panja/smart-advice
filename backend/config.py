import os

# Flask configuration
SECRET_KEY = 'your_super_secret_key_here'  # Change this for production!

# Database configuration - MongoDB
MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
MONGODB_DATABASE = 'super_advice'
# Collections
MONGODB_COLLECTIONS = {
    'recommended_shares': 'recommended_shares',
    'backtest_results': 'backtest_results'
}

# Legacy SQLite config (for migration reference)
# DATABASE = 'data/recommendations.db'
# DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATABASE)

# Strategy configuration - Enable/disable trading strategies
# SWING TRADING OPTIMIZED: Enable key indicators for precision swing trading
STRATEGY_CONFIG = {
    # CORE HIGH-SPEED STRATEGIES (fast computation, proven reliable)
    'MA_Crossover_50_200': True,           # Golden/Death cross - FAST & RELIABLE
    'RSI_Overbought_Oversold': True,       # RSI - FAST computation
    'MACD_Signal_Crossover': True,         # MACD - FAST & RELIABLE
    'Bollinger_Band_Breakout': True,       # Bollinger bands - FAST
    'EMA_Crossover_12_26': True,           # Fast EMA crossover - VERY FAST

    # SWING TRADING ESSENTIALS - Now enabled for better precision
    'ADX_Trend_Strength': True,            # ENABLED - Essential for trend filtering
    'On_Balance_Volume': True,             # ENABLED - Volume confirmation
    'ATR_Volatility': True,                # ENABLED - Volatility gates and position sizing
    'SMA_Crossover_20_50': True,           # ENABLED - Medium-term trend confirmation
    'Stochastic_Overbought_Oversold': True, # ENABLED - With strict filters for swing entries
    'Multi_Timeframe_RSI': False,          # DISABLED - Causes import hang on some systems

    # Keep problematic strategies disabled
    'Volume_Breakout': False,              # Causes loading hang on some environments
    'Support_Resistance_Breakout': False,  # Heavy data needs

    # Additional strategies (disabled for now to avoid heavy imports/CPU)
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

# BALANCED: Minimum combined score for recommendation - balanced for quality and quantity
# Adjusted for realistic recommendations while maintaining quality
MIN_RECOMMENDATION_SCORE = 0.0  # Increased from -0.5 to filter out weak recommendations

# Sentiment analysis configuration
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
ALT_SENTIMENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

# News fetching parameters
NEWS_COUNT = 20
NEWS_MAX_RETRIES = 3
NEWS_DATE_RANGE = '10d'

# Data fetching parameters
HISTORICAL_DATA_PERIOD = '5y'  # Extended for better regime coverage in swing trading
NSE_CACHE_FILE = 'data/nse_symbols.json'

# Threading and batch processing configuration
# OPTIMIZED settings for MAXIMUM performance
MAX_WORKER_THREADS = 4  # Increased threads for parallel processing
BATCH_SIZE = 8  # Smaller batches for faster feedback
REQUEST_DELAY = 0.5  # Reduced delay for faster processing
MAX_RETRIES = 1  # Minimal retries for speed
TIMEOUT_SECONDS = 10  # Faster timeout for non-responsive calls
RATE_LIMIT_DELAY = 2.0  # Reduced rate limit delay
BACKOFF_MULTIPLIER = 1.5  # Reduced backoff multiplier

# Data purge configuration
DATA_PURGE_DAYS = 7  # Number of days to keep old data (recommendations and backtest results)
# WARNING: Setting to 0 will DELETE ALL DATA every time analysis runs!

# SWING TRADING OPTIMIZED: Technical-heavy weightage for precision entries
ANALYSIS_WEIGHTS = {
    'technical': 0.55,    # Technical analysis weight (55%) - Primary for swing trading
    'fundamental': 0.20,  # Fundamental analysis weight (20%) - Quality filter
    'sentiment': 0.10,    # Sentiment analysis weight (10%) - Risk events only
    'sector': 0.05,       # Sector analysis weight (5%) - Sector trends
    'predictive': 0.05,   # Predictive analysis weight (5%) - Supporting signal
    'rl_agent': 0.05      # RL agent weight (5%) - Supporting signal
}

# SWING TRADING PRECISION: More realistic thresholds for current market conditions
RECOMMENDATION_THRESHOLDS = {
    'strong_buy_combined': 0.50,     # Strong signals only
    'buy_combined': 0.10,            # Require positive combined score
    'technical_strong_buy': 0.50,    
    'sell_combined': -0.20,          
    'sentiment_positive': 0.10,      
    'sentiment_negative': -0.20,     
    'sentiment_cap_positive': 0.30,  
    'sentiment_cap_negative': -0.60, 
    'min_backtest_return': 3.0,      # Require minimum 3% CAGR for quality
    'technical_minimum': 0.05,       # Require slightly positive technical
    'fundamental_minimum': 0.05,     # Require slightly positive fundamental
    'volume_confirmation_required': True,  # Enable volume confirmation
    'market_trend_weight': 0.3,      
    'require_all_gates': False,      # Keep flexible for market conditions
    'min_risk_reward_ratio': 1.8,    # Good risk/reward
    'sector_filter_enabled': False,  # Keep disabled for broader coverage
    'min_sector_score': -0.5         
}

# Analysis Modules Configuration - OPTIMIZED for SPEED
# Disable heavy modules for faster analysis
ANALYSIS_CONFIG = {
    'technical_analysis': True,     # Core analysis - ESSENTIAL
    'fundamental_analysis': False,  # DISABLED - Network timeouts slowing analysis
    'sentiment_analysis': False,    # DISABLED - Heavy ML processing
    'sector_analysis': False,       # DISABLED - Additional overhead
    'market_regime_detection': False,  # DISABLED - Heavy ML processing
    'market_microstructure': False,   # DISABLED - Complex simulation
    'alternative_data': False,        # DISABLED - Additional data fetching
    'backtesting': True,             # ENABLED - Simplified for speed
    'risk_management': True,         # ENABLED - Essential for trade planning
    'predictive_analysis': False,    # DISABLED - Heavy ML processing
    'rl_trading_agent': False,       # DISABLED - Heavy ML processing
    'tca_analysis': False           # DISABLED - Complex analysis
}

# SWING TRADING QUALITY: Tighter filters for liquid, tradeable stocks only
STOCK_FILTERING = {
    'min_volume': 100000,           # Higher minimum for liquidity
    'min_price': 20.0,              # Avoid penny stocks
    'max_price': 50000.0,           # Maximum stock price
    'min_market_cap': 500000000,    # 50 crore minimum (mid-cap+)
    'min_historical_days': 250,     # 1 year minimum history
    'volume_lookback_days': 50,     # Volume lookback period
    'exclude_delisted': True,       # Exclude delisted stocks
    'exclude_suspended': True,      # Exclude suspended stocks
    'min_delivery_percent': 30.0,   # NEW: Minimum delivery percentage
    'max_volatility_percentile': 80 # NEW: Avoid extremely volatile stocks
}

# SWING TRADING GATES: Strict entry criteria for high precision
SWING_TRADING_GATES = {
    'trend_filter': {
        'enabled': True,
        'adx_period': 14,
        'adx_threshold': 20,        # Minimum ADX for trending market
        'sma_period': 200,
        'price_above_sma': True     # Price must be above 200 SMA
    },
    'volatility_gate': {
        'enabled': True,
        'atr_period': 14,
        'min_percentile': 20,        # Avoid low volatility
        'max_percentile': 80         # Avoid extreme volatility
    },
    'volume_confirmation': {
        'enabled': True,
        'obv_trend_periods': 10,     # OBV trend lookback
        'volume_zscore_threshold': 1.0,  # Volume spike threshold
        'require_either': True       # Either OBV trend OR volume spike
    },
    'multi_timeframe': {
        'enabled': True,
        'weekly_trend_check': True,  # Check weekly trend alignment
        'weekly_sma_fast': 20,
        'weekly_sma_slow': 50
    }
}

# SWING TRADING PATTERNS: Entry and exit patterns
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
        'initial_stop_type': 'swing_low',  # or 'atr_based'
        'atr_stop_multiplier': 1.5,
        'target_1_atr': 1.0,           # First target at 1x ATR
        'target_2_atr': 2.5,           # Second target at 2.5x ATR
        'trail_stop_atr': 3.0,         # Trail stop at 3x ATR
        'time_stop_bars': 15,          # Exit if no progress
        'breakeven_at_target_1': True
    }
}

# RISK MANAGEMENT: Position sizing and portfolio constraints
RISK_MANAGEMENT = {
    'position_sizing': {
        'method': 'atr_based',        # ATR-based position sizing
        'risk_per_trade': 0.01,       # 1% risk per trade
        'max_position_pct': 0.20      # Max 20% in single position
    },
    'portfolio_constraints': {
        'max_concurrent_positions': 5,
        'max_sector_concentration': 0.40,  # Max 40% in one sector
        'max_correlation': 0.70,       # Max correlation between positions
        'daily_loss_limit': 0.03,      # 3% daily loss limit
        'pause_on_limit_breach': True
    },
    'risk_reward': {
        'min_ratio': 2.5,              # Minimum 2.5:1 risk-reward
        'optimal_ratio': 3.0,          # Target 3:1 risk-reward
        'adjust_targets': True         # Adjust targets for min ratio
    }
}
