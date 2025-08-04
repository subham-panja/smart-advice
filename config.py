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
# REALISTIC CONFIGURATION: Only enable proven, high-quality strategies
STRATEGY_CONFIG = {
    # CORE HIGH-QUALITY STRATEGIES (proven and reliable)
    'MA_Crossover_50_200': True,           # Golden/Death cross - RELIABLE
    'RSI_Overbought_Oversold': True,       # RSI overbought/oversold - RELIABLE (now fixed)
    'MACD_Signal_Crossover': True,         # MACD crossover - RELIABLE
    'Bollinger_Band_Breakout': True,       # Bollinger bands - RELIABLE
    'EMA_Crossover_12_26': True,          # Fast EMA crossover - RELIABLE
    
    # SWING TRADING SPECIFIC STRATEGIES
    'Volume_Breakout': True,               # Volume-confirmed breakouts - GOOD FOR SWING
    'Support_Resistance_Breakout': True,   # Support/Resistance breakouts - GOOD FOR SWING
    'ADX_Trend_Strength': True,            # Trend strength confirmation - GOOD FOR SWING
    'Multi_Timeframe_RSI': True,           # Multi-timeframe RSI confluence - CRITICAL FOR SWING
    
    # MOMENTUM & OSCILLATOR STRATEGIES (selective)
    'Stochastic_Overbought_Oversold': True, # Stochastic - PROVEN
    'Williams_Percent_R_Overbought_Oversold': True, # Williams %R - PROVEN
    
    # VOLUME-BASED STRATEGIES (selective)
    'On_Balance_Volume': True,             # OBV - PROVEN
    
    # ADVANCED SWING TRADING STRATEGIES - ENABLED FOR ELITE PERFORMANCE
    'SMA_Crossover_20_50': True,           # Faster MA crossover for swing trading momentum
    'Fibonacci_Retracement': True,         # CRITICAL: Low-risk entry points at key retracement levels
    'Chart_Patterns': True,                # CRITICAL: High-probability setups (triangles, H&S, etc.)
    'Volume_Profile': True,                # CRITICAL: Precise entry/exit using POC and value areas
    'Volume_Price_Trend': True,            # Enhanced volume-price relationship analysis
    'Momentum_Oscillator': False,         # Disabled to reduce noise
    'ROC_Rate_of_Change': True,            # Rate of change momentum for swing timing
    'ATR_Volatility': True,                # Volatility context for position sizing and stops
    'Keltner_Channels_Breakout': True,     # Alternative breakout confirmation to Bollinger
    'DEMA_Crossover': True,                # Faster, smoother MA signals for swing entries
    'TEMA_Crossover': True,                # Even faster MA signals for precise timing
    'RSI_Bullish_Divergence': True,        # CRITICAL: Advanced RSI signals for reversal detection
    'MACD_Zero_Line_Crossover': True,      # Additional MACD confirmation signals
    'Bollinger_Band_Squeeze': True,        # Low volatility preceding high-volatility breakouts
    'Stochastic_K_D_Crossover': True,      # Enhanced stochastic entry/exit signals
    'DI_Crossover': False,                # Disabled for reliability
    'Ichimoku_Cloud_Breakout': True,       # CRITICAL: Comprehensive trend/momentum system
    'Ichimoku_Kijun_Tenkan_Crossover': True, # CRITICAL: Ichimoku entry/exit signals
    'OBV_Bullish_Divergence': True,        # Advanced volume divergence for reversal signals
    'Accumulation_Distribution_Line': True, # Volume-price accumulation/distribution analysis
    'Candlestick_Hammer': True,            # Bullish reversal pattern at support levels
    'Candlestick_Bullish_Engulfing': True, # Strong bullish reversal pattern
    'Candlestick_Doji': True,              # Indecision pattern signaling potential reversal
    'Parabolic_SAR_Reversal': True,        # Trend reversal and trailing stop signals
    'CCI_Crossover': True,                 # Commodity Channel Index for overbought/oversold
    'Aroon_Oscillator': True,              # Trend strength and likelihood of continuation
    'Ultimate_Oscillator_Buy': True,       # Multi-timeframe oscillator avoiding false divergences
    'Money_Flow_Index_Oversold': True,     # Volume-weighted RSI for enhanced money flow analysis
    'Price_Volume_Trend': True,            # Money flow combining price and volume
    'Chaikin_Oscillator': True,            # Accumulation/distribution volume flow
    'Pivot_Points_Bounce': True,           # Support/resistance levels for bounce trades
    'Gap_Trading': True,                   # CRITICAL: High-probability gap trading opportunities
    'Channel_Trading': True,               # CRITICAL: Channel breakouts and mean reversion
    'Triple_Moving_Average': True,         # Robust trend identification with three MAs
    'Vortex_Indicator': True,              # Trend start identification and confirmation
    'Commodity_Channel_Index': True,       # Price deviation from statistical mean
    'Linear_Regression_Channel': True,     # CRITICAL: Statistical trend channels for precise entries
    'Elder_Ray_Index': True,               # Buying and selling power measurement
    'Keltner_Channel_Squeeze': True,       # Low volatility preceding breakouts
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
# Balanced settings for good performance without overwhelming APIs
MAX_WORKER_THREADS = 6  # Optimal threads for most systems
BATCH_SIZE = 8  # Process multiple stocks per batch for efficiency
REQUEST_DELAY = 0.8  # Reasonable delay to avoid rate limits
MAX_RETRIES = 3  # Standard retry count
TIMEOUT_SECONDS = 30  # Reasonable timeout
RATE_LIMIT_DELAY = 3.0  # Delay when rate limited (seconds)
BACKOFF_MULTIPLIER = 2.0  # Standard backoff multiplier

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

# Analysis Modules Configuration - Enable/disable high-level analysis types
# Optimized ML config for memory-constrained systems
ANALYSIS_CONFIG = {
    'technical_analysis': True,
    'fundamental_analysis': True,
    'sentiment_analysis': True,  # ML-based sentiment analysis
    'sector_analysis': True,  # Sector analysis enabled
    'market_regime_detection': True,  # RE-ENABLED after OpenMP fix
    'market_microstructure': True,  # RE-ENABLED - lightweight simulation mode
    'alternative_data': True,  # ENABLED - Enhanced with real data capabilities
    'backtesting': True,  # Enabled - memory usage is acceptable
    'risk_management': True,
    'predictive_analysis': True,  # RE-ENABLED after OpenMP fix
    'rl_trading_agent': True,  # RE-ENABLED after OpenMP fix
    'tca_analysis': True  # ENABLED - Enhanced transaction cost analysis
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
