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
# OPTIMIZED CONFIGURATION: ONLY FASTEST & MOST EFFECTIVE strategies for SPEED
STRATEGY_CONFIG = {
    # CORE HIGH-SPEED STRATEGIES (fast computation, proven reliable)
    'MA_Crossover_50_200': True,           # Golden/Death cross - FAST & RELIABLE
    'RSI_Overbought_Oversold': True,       # RSI - FAST computation
    'MACD_Signal_Crossover': True,         # MACD - FAST & RELIABLE
    'Bollinger_Band_Breakout': True,       # Bollinger bands - FAST
    'EMA_Crossover_12_26': True,           # Fast EMA crossover - VERY FAST

    # Temporarily DISABLE potentially problematic/slow strategies for stability
    'Volume_Breakout': False,              # Causes loading hang on some environments
    'Support_Resistance_Breakout': False,  # Heavy data needs
    'Stochastic_Overbought_Oversold': False, # Temporarily disabled to avoid hangs

    # Additional strategies (disabled for now to avoid heavy imports/CPU)
    'ADX_Trend_Strength': False,
    'Multi_Timeframe_RSI': False,
    'Williams_Percent_R_Overbought_Oversold': False,
    'On_Balance_Volume': False,
    'SMA_Crossover_20_50': False,
    'Fibonacci_Retracement': False,
    'Chart_Patterns': False,
    'Volume_Profile': False,
    'Volume_Price_Trend': False,
    'Momentum_Oscillator': False,
    'ROC_Rate_of_Change': False,
    'ATR_Volatility': False,
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
MIN_RECOMMENDATION_SCORE = 0.03  # Very low threshold for testing backtest_metrics saving

# Sentiment analysis configuration
SENTIMENT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'
ALT_SENTIMENT_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'

# News fetching parameters
NEWS_COUNT = 20
NEWS_MAX_RETRIES = 3
NEWS_DATE_RANGE = '10d'

# Data fetching parameters
HISTORICAL_DATA_PERIOD = '2y'
NSE_CACHE_FILE = 'data/nse_symbols.json'

# Threading and batch processing configuration
# OPTIMIZED settings for MAXIMUM performance
MAX_WORKER_THREADS = 2  # Reduced threads to avoid rate limiting
BATCH_SIZE = 16  # Larger batches for better efficiency
REQUEST_DELAY = 2.0  # Increased delay to avoid rate limiting
MAX_RETRIES = 2  # Reduced retries to save time
TIMEOUT_SECONDS = 20  # Reduced timeout for faster failures
RATE_LIMIT_DELAY = 5.0  # Increased delay when rate limited
BACKOFF_MULTIPLIER = 2.0  # Increased backoff multiplier

# Data purge configuration
DATA_PURGE_DAYS = 7  # Number of days to keep old data (recommendations and backtest results)
# WARNING: Setting to 0 will DELETE ALL DATA every time analysis runs!

# BALANCED: Analysis weightage configuration - Balanced approach for realistic recommendations
ANALYSIS_WEIGHTS = {
    'technical': 0.40,    # Technical analysis weight (40%) - Strong but balanced
    'fundamental': 0.35,  # Fundamental analysis weight (35%) - Company quality important
    'sentiment': 0.15,    # Sentiment analysis weight (15%) - Market sentiment matters
    'sector': 0.05,       # Sector analysis weight (5%) - Sector trends
    'predictive': 0.03,   # Predictive analysis weight (3%) - Minimal influence
    'rl_agent': 0.02      # RL agent weight (2%) - Minimal influence
}

# BALANCED: Recommendation thresholds - More realistic thresholds for generating recommendations
RECOMMENDATION_THRESHOLDS = {
    'strong_buy_combined': 0.4,      # Reduced for more recommendations
    'buy_combined': 0.03,            # Very low for testing backtest_metrics saving
    'technical_strong_buy': 0.25,    # More achievable technical threshold
    'sell_combined': -0.2,           # Combined score threshold for sell
    'sentiment_positive': 0.05,      # Lowered sentiment threshold
    'sentiment_negative': -0.10,     # Balanced sentiment threshold for negative
    'min_backtest_return': 2.0,      # More realistic minimum CAGR
    'technical_minimum': 0.10,       # Achievable minimum technical score
    'fundamental_minimum': 0.05,     # Achievable minimum fundamental score
    'volume_confirmation_required': True,  # Enable volume confirmation for quality
    'market_trend_weight': 0.2       # Reduced weight for overall market trend
}

# Analysis Modules Configuration - BALANCED for ACCURACY and SPEED
# Enable key analysis modules for better recommendations
ANALYSIS_CONFIG = {
    'technical_analysis': True,     # Core analysis - ESSENTIAL
    'fundamental_analysis': True,   # Core analysis - ESSENTIAL
    'sentiment_analysis': True,     # ENABLED - Important for market sentiment
    'sector_analysis': False,       # DISABLED - Additional overhead (can enable later)
    'market_regime_detection': False,  # DISABLED - Heavy ML processing
    'market_microstructure': False,   # DISABLED - Complex simulation
    'alternative_data': False,        # DISABLED - Additional data fetching
    'backtesting': True,             # ENABLED - Still valuable for recommendations
    'risk_management': True,         # ENABLED - Essential for trade planning
    'predictive_analysis': False,    # DISABLED - Heavy ML processing
    'rl_trading_agent': False,       # DISABLED - Heavy ML processing
    'tca_analysis': False           # DISABLED - Complex analysis
}

# RELAXED: Stock filtering configuration - Allow more stocks to generate data
STOCK_FILTERING = {
    'min_volume': 10000,            # RELAXED minimum average daily volume
    'min_price': 10.0,              # RELAXED minimum stock price
    'max_price': 50000.0,           # Maximum stock price
    'min_market_cap': 50000000,     # RELAXED minimum market cap (5 crores)
    'min_historical_days': 100,     # RELAXED historical data required
    'volume_lookback_days': 50,     # RELAXED volume lookback period
    'exclude_delisted': True,       # Exclude delisted stocks
    'exclude_suspended': True       # Exclude suspended stocks
}
