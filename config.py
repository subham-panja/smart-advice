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
    'Momentum_Oscillator': True,          # Keep disabled to reduce noise
    'ROC_Rate_of_Change': True,            # Rate of change momentum for swing timing
    'ATR_Volatility': True,                # Volatility context for position sizing and stops
    'Keltner_Channels_Breakout': True,     # Alternative breakout confirmation to Bollinger
    'DEMA_Crossover': True,                # Faster, smoother MA signals for swing entries
    'TEMA_Crossover': True,                # Even faster MA signals for precise timing
    'RSI_Bullish_Divergence': True,        # CRITICAL: Advanced RSI signals for reversal detection
    'MACD_Zero_Line_Crossover': True,      # Additional MACD confirmation signals
    'Bollinger_Band_Squeeze': True,        # Low volatility preceding high-volatility breakouts
    'Stochastic_K_D_Crossover': True,      # Enhanced stochastic entry/exit signals
    'DI_Crossover': True,                 # Keep disabled - less reliable
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

# Minimum combined score for recommendation - realistic threshold
# Lowered to account for market conditions and data limitations
MIN_RECOMMENDATION_SCORE = 0.70

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
# Ultra-conservative settings for ML analysis to prevent segmentation faults
MAX_WORKER_THREADS = 4  # Increased threads after OpenMP fix
BATCH_SIZE = 1  # Process one stock at a time for ML analysis
REQUEST_DELAY = 0.5  # Increased delay to reduce API pressure
MAX_RETRIES = 5  # Maximum retries for failed requests
TIMEOUT_SECONDS = 30  # Request timeout in seconds
RATE_LIMIT_DELAY = 2.0  # Additional delay when rate limited (seconds)
BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier for retries

# Data purge configuration
DATA_PURGE_DAYS = 0  # Number of days to keep old data (recommendations and backtest results)

# Analysis weightage configuration - Enhanced with ML components
ANALYSIS_WEIGHTS = {
    'technical': 0.4,     # Technical analysis weight (40%)
    'fundamental': 0.2,   # Fundamental analysis weight (20%)
    'sentiment': 0.15,    # Sentiment analysis weight (15%) - ML-based
    'sector': 0.1,        # Sector analysis weight (10%) - ML insights
    'predictive': 0.1,    # Predictive analysis weight (10%) - LSTM predictions
    'rl_agent': 0.05      # RL agent weight (5%) - Reinforcement learning
}

# Recommendation thresholds - Enhanced for better signal detection
RECOMMENDATION_THRESHOLDS = {
    'strong_buy_combined': 0.55,     # Combined score threshold for strong buy (optimized)
    'buy_combined': 0.30,            # Combined score threshold for buy (more sensitive)
    'technical_strong_buy': 0.25,    # Technical score threshold for strong technical buy (optimized)
    'sell_combined': -0.3,           # Combined score threshold for sell
    'sentiment_positive': 0.40,      # Sentiment score threshold for positive (more sensitive)
    'sentiment_negative': -0.02,     # Sentiment score threshold for negative
    'min_backtest_return': 2.0,      # Allow negative CAGR but not too bad (-2%)
    'technical_minimum': 0.5,       # Minimum technical score to consider
    'fundamental_minimum': 0.3,     # Minimum fundamental score to consider
    'volume_confirmation_required': True,  # Don't require volume confirmation for all signals
    'market_trend_weight': 0.4       # Weight for overall market trend consideration
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

# Stock filtering configuration
STOCK_FILTERING = {
    'min_volume': 10000,           # Minimum average daily volume
    'min_price': 5.0,               # Minimum stock price
    'max_price': 50000.0,           # Maximum stock price
    'min_market_cap': 100000000,    # Minimum market cap (10 crores)
    'min_historical_days': 200,     # Minimum days of historical data required
    'volume_lookback_days': 100,     # Days to look back for volume calculation
    'exclude_delisted': True,       # Exclude delisted stocks
    'exclude_suspended': True       # Exclude suspended stocks
}
