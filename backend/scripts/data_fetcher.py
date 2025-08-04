import pandas as pd
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.logger import setup_logging
from utils.memory_utils import optimize_dataframe_memory
import yfinance as yf
from nsetools import Nse
from config import NSE_CACHE_FILE, STOCK_FILTERING, MAX_WORKER_THREADS, MAX_RETRIES, REQUEST_DELAY, TIMEOUT_SECONDS, RATE_LIMIT_DELAY, BACKOFF_MULTIPLIER, HISTORICAL_DATA_PERIOD
import requests
from requests.exceptions import RequestException
import random
from contextlib import contextmanager

logger = setup_logging()
nse_api = None  # NSE API for stock operations - initialize when needed

def get_all_nse_symbols() -> Dict[str, str]:
    """
    Fetch and cache NSE stock symbols.
    Since nsetools can be unreliable, we'll use a predefined list of major NSE stocks.
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(NSE_CACHE_FILE), exist_ok=True)
    
    if os.path.exists(NSE_CACHE_FILE):
        try:
            with open(NSE_CACHE_FILE, 'r') as f:
                symbols = json.load(f)
                logger.info(f"Loaded {len(symbols)} NSE symbols from cache.")
                return symbols
        except Exception as e:
            logger.error(f"Error loading cached symbols: {e}")
    
    # Fallback to a predefined list of major NSE stocks
    
    try:
        # Initialize NSE API only when needed with timeout protection
        global nse_api
        if nse_api is None:
            logger.info("Initializing NSE API...")
            
            # Try to initialize NSE API with timeout protection
            import signal
            def timeout_handler(signum, frame):
                raise TimeoutError("NSE API initialization timed out")
            
            # Set timeout for NSE initialization (10 seconds)
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            
            try:
                nse_api = Nse()
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                logger.info("NSE API initialized successfully")
            except TimeoutError:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                logger.error("NSE API initialization timed out")
                raise Exception("NSE API initialization timed out")
            except Exception as e:
                signal.alarm(0)  # Cancel the alarm
                signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                logger.error(f"NSE API initialization failed: {e}")
                raise
        
        # Get stock codes from nsetools with timeout
        logger.info("Fetching stock codes from NSE...")
        
        # Set timeout for stock codes fetching (15 seconds)
        def timeout_handler(signum, frame):
            raise TimeoutError("NSE stock codes fetching timed out")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(15)
        
        try:
            stock_codes = nse_api.get_stock_codes()
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
        except TimeoutError:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
            logger.error("NSE stock codes fetching timed out")
            raise Exception("NSE stock codes fetching timed out")
        except Exception as e:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
            logger.error(f"Error fetching stock codes: {e}")
            raise
        
        # Convert list to dictionary format with symbol as key and value
        all_symbols = {symbol: symbol for symbol in stock_codes}
        
        with open(NSE_CACHE_FILE, 'w') as f:
            json.dump(all_symbols, f, indent=4)
        logger.info(f"Fetched and cached {len(all_symbols)} NSE symbols.")
        return all_symbols
    except Exception as e:
        logger.error(f"Error fetching NSE symbols: {e}")
        # Fallback to a minimal set of major stocks
        fallback_stocks = {
            'RELIANCE': 'Reliance Industries Limited',
            'TCS': 'Tata Consultancy Services Limited',
            'HDFCBANK': 'HDFC Bank Limited',
            'INFY': 'Infosys Limited',
            'HINDUNILVR': 'Hindustan Unilever Limited',
            'ICICIBANK': 'ICICI Bank Limited',
            'KOTAKBANK': 'Kotak Mahindra Bank Limited',
            'BHARTIARTL': 'Bharti Airtel Limited',
            'ITC': 'ITC Limited',
            'SBIN': 'State Bank of India'
        }
        logger.info(f"Using fallback stocks: {len(fallback_stocks)} symbols")
        return fallback_stocks

def get_historical_data_with_retry(symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical data with enhanced retry mechanism and monitoring.
    """
    yf_symbol = f"{symbol}.NS"
    
    # Track retry statistics
    retry_stats = {'http_errors': 0, 'timeout_errors': 0, 'data_quality_issues': 0}
    
    for attempt in range(MAX_RETRIES):
        try:
            # Add progressive delay with jitter to avoid overwhelming the API
            if attempt > 0:
                base_delay = REQUEST_DELAY * (BACKOFF_MULTIPLIER ** attempt)
                jitter = random.uniform(0, base_delay * 0.3)  # Add up to 30% jitter
                total_delay = base_delay + jitter
                time.sleep(total_delay)
                logger.info(f"Retry attempt {attempt + 1} for {symbol} after {total_delay:.2f}s delay")
            
            # Download data using yfinance with enhanced error handling
            data = yf.download(yf_symbol, period=period, interval=interval, progress=False, 
                             auto_adjust=True, timeout=TIMEOUT_SECONDS,
                             threads=False, group_by=None)  # Don't group by ticker to avoid MultiIndex
            
            # Enhanced data validation
            if data.empty:
                logger.warning(f"No historical data found for {symbol} (attempt {attempt + 1})")
                if attempt == MAX_RETRIES - 1:
                    return pd.DataFrame()
                continue
            
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns - take the second level (OHLCV names)
                # group_by=None prevents a MultiIndex, so just handle it if present anyway.
                data.columns = data.columns.get_level_values(-1)
            
            # Ensure column names are properly formatted
            if len(data.columns) == 5:
                # Standard OHLCV format
                data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            elif 'Adj Close' in data.columns:
                # Handle adjusted close
                expected_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                if len(data.columns) == len(expected_cols):
                    data.columns = expected_cols
                    # Use Adj Close as Close if it exists
                    data['Close'] = data['Adj Close']
                    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Data quality checks (relaxed for testing)
            min_data_points = 5 if period in ['5d', '1w'] else 10  # Lower requirement for short periods
            if len(data) < min_data_points:  # Too few data points
                retry_stats['data_quality_issues'] += 1
                logger.warning(f"Insufficient data points ({len(data)}) for {symbol} (minimum: {min_data_points})")
                if attempt == MAX_RETRIES - 1:
                    return data  # Return what we have if it's the last attempt
                continue
            
            # Check if we have the basic required columns
            if 'Close' not in data.columns:
                logger.warning(f"No 'Close' column found for {symbol}. Available columns: {list(data.columns)}")
                if attempt == MAX_RETRIES - 1:
                    return pd.DataFrame()
                continue
            
            # Check for data anomalies
            if data['Close'].isna().sum() > len(data) * 0.5:  # More than 50% missing data
                retry_stats['data_quality_issues'] += 1
                logger.warning(f"High percentage of missing data for {symbol}")
                if attempt < MAX_RETRIES - 1:
                    continue
            
            # Success - log and return
            logger.debug(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except (requests.exceptions.RequestException, 
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            retry_stats['http_errors'] += 1
            logger.warning(f"Network error for {symbol} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt == MAX_RETRIES - 1:
                logger.error(f"Failed to fetch data for {symbol} after {MAX_RETRIES} network error attempts")
                return pd.DataFrame()
            continue
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Categorize errors for better handling
            if any(keyword in error_msg for keyword in ['http2', 'curl', 'connection', '401', 'unauthorized']):
                retry_stats['http_errors'] += 1
                logger.warning(f"HTTP/Connection error for {symbol} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to fetch data for {symbol} after {MAX_RETRIES} HTTP error attempts")
                    return pd.DataFrame()
                continue
            
            elif any(keyword in error_msg for keyword in ['timeout', 'timed out']):
                retry_stats['timeout_errors'] += 1
                logger.warning(f"Timeout error for {symbol} (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Failed to fetch data for {symbol} after {MAX_RETRIES} timeout attempts")
                    return pd.DataFrame()
                continue
            
            else:
                logger.error(f"Non-retryable error for {symbol}: {e}")
                return pd.DataFrame()
    
    # Log retry statistics if there were issues
    if any(retry_stats.values()):
        logger.info(f"Retry stats for {symbol}: HTTP errors: {retry_stats['http_errors']}, "
                   f"Timeout errors: {retry_stats['timeout_errors']}, "
                   f"Data quality issues: {retry_stats['data_quality_issues']}")
    
    return pd.DataFrame()

def get_historical_data(symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance with caching.
    Supports multiple intervals ('1d', '1h', '4h').
    NSE symbols need '.NS' suffix for yfinance.
    """
    # Use absolute path for cache directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(backend_dir, "cache")
    cache_file = os.path.join(cache_dir, f"{symbol}_{period}_{interval}.csv")

    # Load from cache if available
    if os.path.exists(cache_file):
        try:
            # Try to read with different possible index column names
            data = None
            for index_col in ['Datetime', 'Date', 0]:  # Try Datetime, Date, or first column
                try:
                    data = pd.read_csv(cache_file, index_col=index_col, parse_dates=True)
                    logger.info(f"Loaded {len(data)} data points for {symbol} ({interval}) from cache using index '{index_col}'.")
                    break
                except (KeyError, ValueError):
                    continue
            
            if data is not None:
                return data
            else:
                # If all attempts fail, read without specifying index and set it manually
                data = pd.read_csv(cache_file, parse_dates=True)
                if not data.empty and len(data.columns) > 0:
                    # Set first column as index if it looks like a date
                    first_col = data.columns[0]
                    if 'date' in first_col.lower():
                        data.set_index(first_col, inplace=True)
                        logger.info(f"Loaded {len(data)} data points for {symbol} ({interval}) from cache using column '{first_col}' as index.")
                        return data
                logger.warning(f"Could not properly load cached data for {symbol}, will fetch fresh data")
        except Exception as e:
            logger.error(f"Error loading cached data for {symbol}: {e}")

    try:
        # Get data with retry mechanism
        data = get_historical_data_with_retry(symbol, period=period, interval=interval)

        if data.empty:
            logger.warning(f"No data found for {symbol} ({interval}).")
            return pd.DataFrame()

        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten MultiIndex columns - take the first level
            data.columns = data.columns.get_level_values(0)

        # Remove any duplicate columns that might exist
        if data.columns.duplicated().any():
            # Keep only unique columns, preferring the first occurrence
            data = data.loc[:, ~data.columns.duplicated()]

        # Ensure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            logger.error(f"Data missing required columns for {symbol} ({interval}). Missing: {missing_cols}. Available: {list(data.columns)}")
            return pd.DataFrame()

        # Optimize memory usage
        data = optimize_dataframe_memory(data)

        # Save to cache with better error handling
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            data.to_csv(cache_file)
            logger.info(f"Fetched and cached {len(data)} data points for {symbol} ({interval}).")
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol}: {e}")

        return data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol} ({interval}): {e}")
        return pd.DataFrame()

def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a stock symbol."""
    try:
        yf_symbol = f"{symbol}.NS"
        
        # Let yfinance handle sessions automatically to avoid curl_cffi errors
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info
        return info.get('currentPrice') or info.get('regularMarketPrice')
    except Exception as e:
        logger.error(f"Error fetching current price for {symbol}: {e}")
        return None

def get_current_price_batch(symbols: list) -> Dict[str, Optional[float]]:
    """
    Get current prices for multiple stock symbols using threading for better performance.
    
    Args:
        symbols: List of stock symbols
        
    Returns:
        Dictionary mapping symbols to their current prices
    """
    logger.info(f"Fetching current prices for {len(symbols)} symbols...")
    
    def fetch_single_price(symbol: str) -> tuple:
        """Fetch price for a single symbol."""
        price = get_current_price(symbol)
        return (symbol, price)
    
    results = {}
    max_workers = min(MAX_WORKER_THREADS, len(symbols))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(fetch_single_price, symbol): symbol
            for symbol in symbols
        }
        
        # Process completed tasks
        for future in as_completed(future_to_symbol):
            try:
                symbol, price = future.result()
                results[symbol] = price
            except Exception as e:
                symbol = future_to_symbol[future]
                logger.error(f"Error fetching price for {symbol}: {e}")
                results[symbol] = None
    
    logger.info(f"Fetched prices for {len(results)} symbols")
    return results

def get_stock_info_with_retry(symbol: str, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """
    Get comprehensive stock information with retry mechanism for rate limiting.
    
    Args:
        symbol: Stock symbol
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary containing stock information
    """
    for attempt in range(max_retries):
        try:
            yf_symbol = f"{symbol}.NS"
            
            # Let yfinance handle sessions automatically to avoid curl_cffi errors
            ticker = yf.Ticker(yf_symbol)
            
            # Add delay between attempts
            if attempt > 0:
                delay = RATE_LIMIT_DELAY * (BACKOFF_MULTIPLIER ** (attempt - 1))
                logger.info(f"Retrying {symbol} after {delay:.1f}s delay (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            
            info = ticker.info
            
            # Check if we got valid info (not None or empty)
            if info is None or not info or info.get('regularMarketPrice') is None:
                if attempt == max_retries - 1:
                    logger.warning(f"No valid ticker info for {symbol} after {max_retries} attempts. Response: {info}")
                    return {'symbol': symbol, 'valid': False, 'reason': 'No valid ticker info'}
                continue

            # Get historical data for volume calculation
            hist_data = get_historical_data(symbol, '3mo')
            
            if hist_data.empty:
                return {'symbol': symbol, 'valid': False, 'reason': 'No historical data'}
            
            # Calculate average volume
            avg_volume = hist_data['Volume'].mean()
            current_price = hist_data['Close'].iloc[-1]
            
            # Get market cap (if available)
            market_cap = info.get('marketCap', 0)
            
            return {
                'symbol': symbol,
                'valid': True,
                'current_price': current_price,
                'avg_volume': avg_volume,
                'market_cap': market_cap,
                'historical_days': len(hist_data),
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown')
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check for HTTP or network-related errors
            if any(keyword in error_msg for keyword in ['http', '401', '429', '502', 'failed to fetch']):
                if attempt == max_retries - 1:
                    logger.error(f"HTTP/network error for {symbol} after {max_retries} attempts: {e}")
                    return {'symbol': symbol, 'valid': False, 'reason': 'HTTP/network error'}
                
                # Exponential backoff for these errors
                delay = RATE_LIMIT_DELAY * (BACKOFF_MULTIPLIER ** attempt)
                logger.warning(f"HTTP/network error for {symbol}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                # Add extra random delay to prevent API overload
                time.sleep(random.uniform(0, 0.5))
                continue
            
            # For other exceptions, do not retry
            logger.error(f"Unhandled error getting stock info for {symbol}: {e}")
            return {'symbol': symbol, 'valid': False, 'reason': str(e)}
    
    # If we get here, all retries failed
    return {'symbol': symbol, 'valid': False, 'reason': 'Max retries exceeded'}

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive stock information for filtering.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        Dictionary containing stock information
    """
    return get_stock_info_with_retry(symbol)

def process_stock_for_filtering(symbol_data: tuple, filtering_criteria: dict) -> tuple:
    """
    Process a single stock for filtering (used in threading).
    
    Args:
        symbol_data: Tuple of (symbol, name)
        filtering_criteria: Dictionary of filtering criteria
        
    Returns:
        Tuple of (symbol, name, stock_info, passed_filters)
    """
    symbol, name = symbol_data
    min_volume = filtering_criteria['min_volume']
    min_price = filtering_criteria['min_price']
    max_price = filtering_criteria['max_price']
    min_market_cap = filtering_criteria['min_market_cap']
    min_historical_days = filtering_criteria['min_historical_days']
    
    try:
        # Add small random delay to prevent API overload
        delay = REQUEST_DELAY + random.uniform(0, REQUEST_DELAY)
        time.sleep(delay)
        
        # Get stock information
        stock_info = get_stock_info(symbol)
        
        if not stock_info['valid']:
            logger.debug(f"Skipping {symbol}: {stock_info['reason']}")
            return (symbol, name, stock_info, False)
        
        # Apply filters
        current_price = stock_info['current_price']
        avg_volume = stock_info['avg_volume']
        market_cap = stock_info['market_cap']
        historical_days = stock_info['historical_days']
        
        # Price filter
        if current_price < min_price or current_price > max_price:
            logger.debug(f"Skipping {symbol}: Price {current_price} not in range [{min_price}, {max_price}]")
            return (symbol, name, stock_info, False)
        
        # Volume filter
        if avg_volume < min_volume:
            logger.debug(f"Skipping {symbol}: Volume {avg_volume:,.0f} below minimum {min_volume:,.0f}")
            return (symbol, name, stock_info, False)
        
        # Market cap filter (if available)
        if market_cap > 0 and market_cap < min_market_cap:
            logger.debug(f"Skipping {symbol}: Market cap {market_cap:,.0f} below minimum {min_market_cap:,.0f}")
            return (symbol, name, stock_info, False)
        
        # Historical data filter
        if historical_days < min_historical_days:
            logger.debug(f"Skipping {symbol}: Historical days {historical_days} below minimum {min_historical_days}")
            return (symbol, name, stock_info, False)
        
        # Stock passed all filters
        logger.info(f"Added {symbol}: Price={current_price:.2f}, Volume={avg_volume:,.0f}, Days={historical_days}")
        return (symbol, name, stock_info, True)
        
    except Exception as e:
        logger.error(f"Error filtering stock {symbol}: {e}")
        return (symbol, name, None, False)

def filter_active_stocks(symbols: Dict[str, str], max_stocks: int = None) -> Dict[str, str]:
    """
    Filter stocks to get only actively traded ones with sufficient historical data.
    Uses threading for parallel processing to improve performance.
    
    Args:
        symbols: Dictionary of stock symbols
        max_stocks: Maximum number of stocks to return
        
    Returns:
        Dictionary of filtered stock symbols
    """
    logger.info(f"Filtering {len(symbols)} stocks for active trading and historical data with max_stocks={max_stocks}...")
    
    filtered_stocks = {}
    
    # Get filtering criteria from config - more lenient for test mode and large-scale analysis
    if max_stocks is not None and max_stocks <= 10:  # Test mode with small number of stocks
        filtering_criteria = {
            'min_volume': 1000,      # Very low volume requirement for testing
            'min_price': 1.0,        # Low price requirement
            'max_price': 50000.0,    # High price limit
            'min_market_cap': 0,     # No market cap requirement
            'min_historical_days': 30  # Only 30 days of historical data needed
        }
        logger.info("Using relaxed filtering criteria for test mode")
    elif max_stocks is not None and max_stocks >= 100:  # Large-scale analysis mode
        filtering_criteria = {
            'min_volume': 1000,      # Very low volume requirement for large-scale analysis
            'min_price': 1.0,        # Very low price requirement
            'max_price': 50000.0,    # High price limit
            'min_market_cap': 0,     # No market cap requirement for large-scale analysis
            'min_historical_days': 30  # Very low historical data requirement
        }
        logger.info("Using very relaxed filtering criteria for large-scale analysis")
    elif max_stocks is not None and max_stocks >= 20:  # Mid-sized analysis mode (20-99 stocks)
        filtering_criteria = {
            'min_volume': 5000,      # Relaxed volume requirement for mid-sized analysis
            'min_price': 2.0,        # Relaxed price requirement
            'max_price': 50000.0,    # High price limit
            'min_market_cap': 10000000,  # Relaxed market cap requirement (1 crore)
            'min_historical_days': 200  # Use configured requirement from config.py
        }
        logger.info("Using relaxed moderate filtering criteria for mid-sized analysis")
    else:
        filtering_criteria = {
            'min_volume': STOCK_FILTERING.get('min_volume', 100000),
            'min_price': STOCK_FILTERING.get('min_price', 5.0),
            'max_price': STOCK_FILTERING.get('max_price', 50000.0),
            'min_market_cap': STOCK_FILTERING.get('min_market_cap', 100000000),
            'min_historical_days': STOCK_FILTERING.get('min_historical_days', 200)
        }
    
    # Convert symbols dict to list of tuples for threading
    symbol_list = list(symbols.items())
    
    # Use ThreadPoolExecutor for parallel processing with adaptive concurrency
    # Reduce workers significantly to avoid rate limiting
    adaptive_workers = max(1, min(2, len(symbol_list)))  # Use only 1-2 workers
    logger.info(f"Using {adaptive_workers} worker threads for rate-limited parallel processing")
    
    with ThreadPoolExecutor(max_workers=adaptive_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_stock_for_filtering, symbol_data, filtering_criteria): symbol_data[0]
            for symbol_data in symbol_list
        }
        
        # Process completed tasks
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol, name, stock_info, passed_filters = future.result()
                
                if passed_filters:
                    filtered_stocks[symbol] = name
                    
                    # Check if we've reached the maximum
                    if max_stocks and len(filtered_stocks) >= max_stocks:
                        logger.info(f"Reached maximum stocks limit of {max_stocks}")
                        # Cancel remaining futures
                        for remaining_future in future_to_symbol:
                            remaining_future.cancel()
                        break
                        
            except Exception as e:
                logger.error(f"Error processing stock {symbol}: {e}")
                continue
    
    logger.info(f"Filtered to {len(filtered_stocks)} active stocks from {len(symbols)} total symbols")
    return filtered_stocks

def get_offline_symbols_from_cache(max_stocks: int = None) -> Dict[str, str]:
    """
    Get symbols purely from existing cache without making any API calls.
    This is a fallback mode to avoid rate limiting issues.
    
    Args:
        max_stocks: Maximum number of stocks to return
        
    Returns:
        Dictionary of cached stock symbols
    """
    logger.info(f"Using offline mode: getting symbols from existing cache files with max_stocks={max_stocks}")
    
    # Check if we have any cached CSV files that indicate stock symbols
    cached_symbols = {}
    
    try:
        # Use absolute path for cache directory
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(backend_dir, "cache")
        
        if os.path.exists(cache_dir):
            # Get all CSV files in cache directory
            csv_files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
            
            # Extract stock symbols from CSV filenames
            # Format: SYMBOL_period_interval.csv (e.g., RELIANCE_1y_1d.csv)
            for csv_file in csv_files:
                try:
                    # Extract symbol from filename
                    symbol = csv_file.split('_')[0]
                    if symbol and len(symbol) <= 20:  # Valid stock symbol length
                        cached_symbols[symbol] = symbol
                        
                        # Stop if we have enough symbols
                        if max_stocks and len(cached_symbols) >= max_stocks:
                            break
                except:
                    continue
            
            logger.info(f"Found {len(cached_symbols)} cached stock symbols")
            
            # If we found cached symbols, use them
            if cached_symbols:
                if max_stocks:
                    # Return only the requested number
                    limited_symbols = dict(list(cached_symbols.items())[:max_stocks])
                    logger.info(f"Using {len(limited_symbols)} cached symbols for offline mode")
                    return limited_symbols
                else:
                    return cached_symbols
    
    except Exception as e:
        logger.error(f"Error reading cached symbols: {e}")
    
    # Fallback to known major stocks if no cache found
    fallback_stocks = {
        'RELIANCE': 'Reliance Industries Limited',
        'TCS': 'Tata Consultancy Services Limited', 
        'HDFCBANK': 'HDFC Bank Limited',
        'INFY': 'Infosys Limited',
        'HINDUNILVR': 'Hindustan Unilever Limited',
        'ICICIBANK': 'ICICI Bank Limited',
        'SBIN': 'State Bank of India',
        'BHARTIARTL': 'Bharti Airtel Limited',
        'ITC': 'ITC Limited',
        'KOTAKBANK': 'Kotak Mahindra Bank Limited'
    }
    
    if max_stocks:
        fallback_stocks = dict(list(fallback_stocks.items())[:max_stocks])
    
    logger.info(f"Using {len(fallback_stocks)} fallback symbols for offline mode")
    return fallback_stocks

def get_filtered_nse_symbols(max_stocks: int = None) -> Dict[str, str]:
    """
    Get filtered NSE symbols that meet active trading criteria with caching.
    FAST MODE: Skip heavy filtering for better performance.
    
    Args:
        max_stocks: Maximum number of stocks to return
        
    Returns:
        Dictionary of filtered stock symbols
    """
    logger.info(f"Getting filtered NSE symbols with max_stocks={max_stocks} (FAST MODE)")
    
    # Use cache for filtered symbols with absolute path
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(backend_dir, "cache")
    cache_file = os.path.join(cache_dir, f"filtered_symbols_{max_stocks or 'all'}.json")
    
    # Load from cache if available and not older than 24 hours (extended cache time)
    if os.path.exists(cache_file):
        try:
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 86400:  # 24 hours
                with open(cache_file, 'r') as f:
                    filtered_symbols = json.load(f)
                    logger.info(f"Loaded {len(filtered_symbols)} filtered symbols from cache.")
                    if filtered_symbols:  # Only return if not empty
                        return filtered_symbols
                    else:
                        logger.info("Cache file is empty, proceeding to use known stocks.")
        except Exception as e:
            logger.error(f"Error loading cached filtered symbols: {e}")
    
    # FAST MODE: Skip expensive API filtering and use predefined liquid stocks
    logger.info("FAST MODE: Using predefined liquid stocks to avoid API bottlenecks")
    
    # Comprehensive list of liquid NSE stocks (sorted by market cap and liquidity)
    liquid_stocks = {
        'RELIANCE': 'Reliance Industries Limited',
        'TCS': 'Tata Consultancy Services Limited',
        'HDFCBANK': 'HDFC Bank Limited',
        'INFY': 'Infosys Limited',
        'HINDUNILVR': 'Hindustan Unilever Limited',
        'ICICIBANK': 'ICICI Bank Limited',
        'SBIN': 'State Bank of India',
        'BHARTIARTL': 'Bharti Airtel Limited',
        'ITC': 'ITC Limited',
        'KOTAKBANK': 'Kotak Mahindra Bank Limited',
        'LT': 'Larsen & Toubro Limited',
        'ASIANPAINT': 'Asian Paints Limited',
        'AXISBANK': 'Axis Bank Limited',
        'MARUTI': 'Maruti Suzuki India Limited',
        'SUNPHARMA': 'Sun Pharmaceutical Industries Limited',
        'ULTRACEMCO': 'UltraTech Cement Limited',
        'TITAN': 'Titan Company Limited',
        'NESTLEIND': 'Nestle India Limited',
        'POWERGRID': 'Power Grid Corporation of India Limited',
        'NTPC': 'NTPC Limited',
        'BAJFINANCE': 'Bajaj Finance Limited',
        'ONGC': 'Oil & Natural Gas Corporation Limited',
        'TECHM': 'Tech Mahindra Limited',
        'BAJAJFINSV': 'Bajaj Finserv Limited',
        'HCLTECH': 'HCL Technologies Limited',
        'WIPRO': 'Wipro Limited',
        'COALINDIA': 'Coal India Limited',
        'DRREDDY': 'Dr. Reddys Laboratories Limited',
        'JSWSTEEL': 'JSW Steel Limited',
        'TATASTEEL': 'Tata Steel Limited',
        'GRASIM': 'Grasim Industries Limited',
        'HINDALCO': 'Hindalco Industries Limited',
        'BRITANNIA': 'Britannia Industries Limited',
        'DIVISLAB': 'Divis Laboratories Limited',
        'EICHERMOT': 'Eicher Motors Limited',
        'HEROMOTOCO': 'Hero MotoCorp Limited',
        'BAJAJ-AUTO': 'Bajaj Auto Limited',
        'ADANIPORTS': 'Adani Ports and Special Economic Zone Limited',
        'BPCL': 'Bharat Petroleum Corporation Limited',
        'CIPLA': 'Cipla Limited',
        'SHREECEM': 'Shree Cement Limited',
        'INDUSINDBK': 'IndusInd Bank Limited',
        'APOLLOHOSP': 'Apollo Hospitals Enterprise Limited',
        'PIDILITIND': 'Pidilite Industries Limited',
        'GODREJCP': 'Godrej Consumer Products Limited',
        'MCDOWELL-N': 'United Spirits Limited',
        'IOC': 'Indian Oil Corporation Limited',
        'TATACONSUM': 'Tata Consumer Products Limited',
        'HDFCLIFE': 'HDFC Life Insurance Company Limited',
        'SBILIFE': 'SBI Life Insurance Company Limited',
        'ICICIPRULI': 'ICICI Prudential Life Insurance Company Limited',
        'DABUR': 'Dabur India Limited',
        'COLPAL': 'Colgate Palmolive (India) Limited',
        'MARICO': 'Marico Limited',
        'BERGEPAINT': 'Berger Paints India Limited'
    }
    
    # Apply max_stocks limit if specified
    if max_stocks is not None:
        filtered_symbols = dict(list(liquid_stocks.items())[:max_stocks])
        logger.info(f"Selected {len(filtered_symbols)} liquid stocks from predefined list")
    else:
        filtered_symbols = liquid_stocks
        logger.info(f"Using all {len(filtered_symbols)} predefined liquid stocks")
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(filtered_symbols, f, indent=4)
        logger.info(f"Cached {len(filtered_symbols)} filtered symbols.")
    except Exception as e:
        logger.error(f"Error caching filtered symbols: {e}")
    
    return filtered_symbols
