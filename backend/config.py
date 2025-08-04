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
    'EMA_Crossover_12_26': True,          # Fast EMA crossover - VERY FAST
    
    # ESSENTIAL SWING TRADING STRATEGIES (high impact, fast computation)
    'Volume_Breakout': True,               # Volume breakouts - FAST & EFFECTIVE
    'Support_Resistance_Breakout': True,   # S/R breakouts - FAST computation
    'Stochastic_Overbought_Oversold': True, # Stochastic - FAST oscillator
    
    # DISABLED STRATEGIES FOR SPEED OPTIMIZATION
    'ADX_Trend_Strength': False,            # DISABLED - Complex computation
    'Multi_Timeframe_RSI': False,           # DISABLED - Multiple timeframe = slower
    'Williams_Percent_R_Overbought_Oversold': False, # DISABLED - Redundant with stochastic
    'On_Balance_Volume': False,             # DISABLED - Volume analysis overhead
    'SMA_Crossover_20_50': False,           # DISABLED - Redundant with MA crossover
    'Fibonacci_Retracement': False,         # DISABLED - Complex calculation
    'Chart_Patterns': False,                # DISABLED - Heavy pattern recognition
    'Volume_Profile': False,                # DISABLED - Complex volume analysis
    'Volume_Price_Trend': False,            # DISABLED - Additional volume computation
    'Momentum_Oscillator': False,           # DISABLED - Redundant
    'ROC_Rate_of_Change': False,            # DISABLED - Additional momentum calc
    'ATR_Volatility': False,                # DISABLED - Volatility computation
    'Keltner_Channels_Breakout': False,     # DISABLED - Similar to Bollinger
    'DEMA_Crossover': False,                # DISABLED - Redundant with EMA
    'TEMA_Crossover': False,                # DISABLED - Redundant with EMA
    'RSI_Bullish_Divergence': False,        # DISABLED - Complex divergence analysis
    'MACD_Zero_Line_Crossover': False,      # DISABLED - Redundant with MACD signal
    'Bollinger_Band_Squeeze': False,        # DISABLED - Additional BB computation
    'Stochastic_K_D_Crossover': False,      # DISABLED - Redundant with stochastic
    'DI_Crossover': False,                  # DISABLED - Complex directional indicators
    'Ichimoku_Cloud_Breakout': False,       # DISABLED - Very complex calculation
    'Ichimoku_Kijun_Tenkan_Crossover': False, # DISABLED - Complex Ichimoku system
    'OBV_Bullish_Divergence': False,        # DISABLED - Complex divergence analysis
    'Accumulation_Distribution_Line': False, # DISABLED - Volume-price computation
    'Candlestick_Hammer': False,            # DISABLED - Pattern recognition overhead
    'Candlestick_Bullish_Engulfing': False, # DISABLED - Pattern recognition overhead
    'Candlestick_Doji': False,              # DISABLED - Pattern recognition overhead
    'Parabolic_SAR_Reversal': False,        # DISABLED - SAR calculation overhead
    'CCI_Crossover': False,                 # DISABLED - CCI computation
    'Aroon_Oscillator': False,              # DISABLED - Aroon calculation
    'Ultimate_Oscillator_Buy': False,       # DISABLED - Multi-timeframe oscillator
    'Money_Flow_Index_Oversold': False,     # DISABLED - Volume-weighted computation
    'Price_Volume_Trend': False,            # DISABLED - Volume-price analysis
    'Chaikin_Oscillator': False,            # DISABLED - Accumulation/distribution
    'Pivot_Points_Bounce': False,           # DISABLED - Pivot calculation
    'Gap_Trading': False,                   # DISABLED - Gap analysis
    'Channel_Trading': False,               # DISABLED - Channel computation
    'Triple_Moving_Average': False,         # DISABLED - Multiple MA calculation
    'Vortex_Indicator': False,              # DISABLED - Vortex computation
    'Commodity_Channel_Index': False,       # DISABLED - CCI calculation
    'Linear_Regression_Channel': False,     # DISABLED - Regression computation
    'Elder_Ray_Index': False,               # DISABLED - Elder Ray calculation
    'Keltner_Channel_Squeeze': False,       # DISABLED - Keltner computation
}

# RELAXED: Minimum combined score for recommendation - easier to generate recommendations
# Lowered to make it easier to see data in the frontend dashboard
MIN_RECOMMENDATION_SCORE = 0.15

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
MAX_WORKER_THREADS = 12  # Increased threads for faster processing
BATCH_SIZE = 16  # Larger batches for better efficiency
REQUEST_DELAY = 0.3  # Reduced delay for faster processing
MAX_RETRIES = 2  # Reduced retries to save time
TIMEOUT_SECONDS = 20  # Reduced timeout for faster failures
RATE_LIMIT_DELAY = 1.5  # Reduced delay when rate limited
BACKOFF_MULTIPLIER = 1.5  # Reduced backoff multiplier

# Data purge configuration
DATA_PURGE_DAYS = 7  # Number of days to keep old data (recommendations and backtest results)
# WARNING: Setting to 0 will DELETE ALL DATA every time analysis runs!

# VERY STRICT: Analysis weightage configuration - Emphasis on technical and fundamental quality
ANALYSIS_WEIGHTS = {
    'technical': 0.45,    # Technical analysis weight (45%) - Increased for signal quality
    'fundamental': 0.35,  # Fundamental analysis weight (35%) - Increased for company quality
    'sentiment': 0.08,    # Sentiment analysis weight (8%) - Reduced influence
    'sector': 0.05,       # Sector analysis weight (5%) - Reduced influence
    'predictive': 0.05,   # Predictive analysis weight (5%) - Reduced influence
    'rl_agent': 0.02      # RL agent weight (2%) - Minimal influence
}

# RELAXED: Recommendation thresholds - Much easier to generate recommendations
RECOMMENDATION_THRESHOLDS = {
    'strong_buy_combined': 0.50,     # RELAXED threshold for strong buy
    'buy_combined': 0.20,            # RELAXED threshold for buy
    'technical_strong_buy': 0.25,    # RELAXED threshold for strong technical buy
    'sell_combined': -0.3,           # Combined score threshold for sell
    'sentiment_positive': 0.05,      # RELAXED sentiment threshold for positive
    'sentiment_negative': -0.15,     # Stricter sentiment threshold for negative
    'min_backtest_return': 5.0,      # RELAXED minimum 5% CAGR
    'technical_minimum': 0.10,       # RELAXED minimum technical score
    'fundamental_minimum': 0.05,     # RELAXED minimum fundamental score
    'volume_confirmation_required': False,  # RELAXED - no volume confirmation required
    'market_trend_weight': 0.3       # Lower weight for overall market trend
}

# Analysis Modules Configuration - OPTIMIZED for SPEED
# Disabled heavy ML/analysis modules for faster processing
ANALYSIS_CONFIG = {
    'technical_analysis': True,     # Core analysis - ESSENTIAL
    'fundamental_analysis': True,   # Core analysis - ESSENTIAL
    'sentiment_analysis': False,    # DISABLED - Heavy ML processing
    'sector_analysis': False,       # DISABLED - Additional overhead
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
