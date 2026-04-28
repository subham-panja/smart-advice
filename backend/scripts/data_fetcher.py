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
from config import NSE_CACHE_FILE, STOCK_FILTERING, MAX_WORKER_THREADS, MAX_RETRIES, REQUEST_DELAY, TIMEOUT_SECONDS, RATE_LIMIT_DELAY, BACKOFF_MULTIPLIER, HISTORICAL_DATA_PERIOD, FILTERED_SYMBOLS_CACHE_HOURS, FILTER_VALIDATION_PERIOD, DATA_FETCH_THREADS
import requests
from requests.exceptions import RequestException
import random
from contextlib import contextmanager

# Import alternative data fetcher
try:
    from scripts.alternative_data_fetcher import AlternativeDataFetcher, get_alternative_nse_symbols
    ALTERNATIVE_FETCHER_AVAILABLE = True
except ImportError:
    ALTERNATIVE_FETCHER_AVAILABLE = False
    logger.warning("Alternative data fetcher not available")

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
            
            # Set timeout for NSE initialization (5 seconds) - reduced to prevent hang
            # Note: This only works on Unix-like systems
            import platform
            if platform.system() != 'Windows':
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
            
            try:
                nse_api = Nse()
                if platform.system() != 'Windows':
                    signal.alarm(0)  # Cancel the alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                logger.info("NSE API initialized successfully")
            except TimeoutError:
                if platform.system() != 'Windows':
                    signal.alarm(0)  # Cancel the alarm
                    signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
                logger.error("NSE API initialization timed out")
                raise Exception("NSE API initialization timed out")
            except Exception as e:
                if platform.system() != 'Windows':
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
        
        # Try alternative symbol fetcher first
        if ALTERNATIVE_FETCHER_AVAILABLE:
            try:
                logger.info("Trying alternative symbol fetcher...")
                alt_symbols = get_alternative_nse_symbols()
                if alt_symbols:
                    logger.info(f"Got {len(alt_symbols)} symbols from alternative fetcher")
                    # Cache the alternative symbols
                    try:
                        with open(NSE_CACHE_FILE, 'w') as f:
                            json.dump(alt_symbols, f, indent=4)
                        logger.info(f"Cached alternative symbols")
                    except Exception as cache_e:
                        logger.warning(f"Failed to cache alternative symbols: {cache_e}")
                    return alt_symbols
            except Exception as alt_e:
                logger.error(f"Alternative symbol fetcher also failed: {alt_e}")
        
        # Final fallback to a minimal set of major stocks
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
        logger.info(f"Using final fallback stocks: {len(fallback_stocks)} symbols")
        return fallback_stocks

def get_historical_data_with_retry(symbol: str, period: str = '1y', interval: str = '1d') -> pd.DataFrame:
    """
    Fetch historical data with enhanced retry mechanism and monitoring.
    """
    yf_symbol = f"{symbol}.NS"
    
    # Track retry statistics
    retry_stats = {'http_errors': 0, 'timeout_errors': 0, 'data_quality_issues': 0}
    
    # Add initial delay to prevent rate limiting
    time.sleep(1.0)  # Add 1 second delay between API calls
    
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

            # Ensure we have a DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame(name='Close')
            elif not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            
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

def _cache_checksum_path(path: str) -> str:
    return path + '.sha256'

def _write_checksum(path: str):
    try:
        import hashlib
        with open(path, 'rb') as f:
            data = f.read()
        digest = hashlib.sha256(data).hexdigest()
        with open(_cache_checksum_path(path), 'w') as cf:
            cf.write(digest)
    except Exception as e:
        logger.warning(f"Failed to write checksum for {path}: {e}")


def _verify_checksum(path: str, strict: bool = False) -> bool:
    """Verify file integrity using checksum. If strict=False, skips slow hashing."""
    if not strict:
        return True  # Fast path: trust the file
    try:
        import hashlib
        ch_path = _cache_checksum_path(path)
        if not os.path.exists(ch_path):
            return True  # no checksum available, don't block
        with open(ch_path, 'r') as cf:
            expected = cf.read().strip()
        with open(path, 'rb') as f:
            data = f.read()
        actual = hashlib.sha256(data).hexdigest()
        return actual == expected
    except Exception as e:
        logger.warning(f"Checksum verification failed for {path}: {e}")
        return False


def get_historical_data(symbol: str, period: str = '1y', interval: str = '1d', fresh: bool = False) -> pd.DataFrame:
    """
    Fetch historical stock data using yfinance with caching.
    Supports multiple intervals ('1d', '1h', '4h').
    NSE symbols need '.NS' suffix for yfinance.
    
    Args:
        symbol: Stock symbol
        period: Data period (e.g. '1y')
        interval: Data interval (e.g. '1d')
        fresh: If True, force fresh data fetch (ignore cache unless it's from today)
    """
    # Use absolute path for cache directory
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(backend_dir, "cache")
    # Try provider-specific cache files (alt or yf)
    provider_candidates = [
        os.path.join(cache_dir, f"{symbol}_{period}_{interval}_alt.csv"),
        os.path.join(cache_dir, f"{symbol}_{period}_{interval}_yf.csv"),
    ]

    # Check if we can use cache
    use_cache = True
    if fresh:
        use_cache = False
        # If we have a cache file from TODAY, we can still use it even if fresh=True
        # because it means we already fetched it today
        for path in provider_candidates:
            if os.path.exists(path):
                try:
                    mtime = os.path.getmtime(path)
                    file_date = datetime.fromtimestamp(mtime).date()
                    today = datetime.now().date()
                    if file_date == today:
                        # Cache is from today, so it is fresh!
                        use_cache = True
                        logger.info(f"Cache for {symbol} is from today ({file_date}), using it despite fresh=True")
                        break
                except Exception:
                    pass
        
        if not use_cache:
            logger.info(f"Fresh data requested for {symbol}, bypassing cache")

    # Load from the freshest valid cache if available and allowed
    try:
        if use_cache:
            freshest_path = None
            freshest_mtime = -1
            for path in provider_candidates:
                if os.path.exists(path):
                    if not _verify_checksum(path):
                        logger.warning(f"Checksum mismatch for cache {path}; ignoring")
                        continue
                    mtime = os.path.getmtime(path)
                    if mtime > freshest_mtime:
                        freshest_mtime = mtime
                        freshest_path = path
            if freshest_path:
                # Speed Optimization: If file was modified very recently (e.g. within 12 hours), 
                # trust it completely and skip even the delta fetch.
                mtime = os.path.getmtime(freshest_path)
                is_very_fresh = (time.time() - mtime) < 43200  # 12 hours
                
                # Try to read with different possible index column names
                for index_col in ['Datetime', 'Date', 0]:
                    try:
                        # Optimization: Use engine='c' and memory_map for faster reading
                        data = pd.read_csv(freshest_path, index_col=index_col, parse_dates=True, engine='c', memory_map=True)
                        
                        if is_very_fresh:
                            logger.info(f"Loaded {len(data)} points for {symbol} (Cache is very fresh, skipping delta fetch)")
                            return data
                            
                        logger.info(f"Loaded {len(data)} data points for {symbol} ({interval}) from cache: {os.path.basename(freshest_path)}")
                        if not data.empty:
                            try:
                                last_ts = data.index[-1]
                                today = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo else pd.Timestamp.now()
                                
                                # Check if we need more data (delta)
                                days_diff = (today - last_ts).days
                                
                                if days_diff <= 0:
                                    logger.debug(f"Cache for {symbol} is up to date.")
                                    return data
                                
                                # Fetch only the delta
                                logger.info(f"Fetching delta for {symbol} ({days_diff} days missing)")
                                
                                # Use yfinance for delta (much faster than full download)
                                # Start date is one day after last cached date
                                start_date = (last_ts + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                                
                                # For simplicity, let's just fetch from the start_date to today
                                delta_data = yf.download(f"{symbol}.NS", start=start_date, interval=interval, 
                                                       progress=False, auto_adjust=True, timeout=TIMEOUT_SECONDS)
                                
                                if not delta_data.empty:
                                    # Standardize columns
                                    if isinstance(delta_data.columns, pd.MultiIndex):
                                        delta_data.columns = delta_data.columns.get_level_values(-1)
                                    
                                    # Clean up columns to match our format
                                    if 'Adj Close' in delta_data.columns:
                                        delta_data['Close'] = delta_data['Adj Close']
                                    
                                    # Keep only relevant columns if they exist
                                    cols_to_keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in delta_data.columns]
                                    delta_data = delta_data[cols_to_keep]
                                    
                                    # Deduplicate both datasets to prevent reindexing errors
                                    data = data[~data.index.duplicated(keep='last')]
                                    delta_data = delta_data[~delta_data.index.duplicated(keep='last')]
                                    
                                    # Merge and optimize
                                    combined_data = pd.concat([data, delta_data])
                                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                                    combined_data = combined_data.sort_index()
                                    
                                    # Save updated cache
                                    combined_data.to_csv(freshest_path)
                                    _write_checksum(freshest_path)
                                    logger.info(f"Updated cache for {symbol} with {len(delta_data)} new points.")
                                    return combined_data
                                else:
                                    logger.debug(f"No new data found for {symbol}, using cache.")
                                    # Even if no delta, return deduplicated cached data
                                    return data[~data.index.duplicated(keep='last')]
                            except Exception as delta_e:
                                logger.warning(f"Failed to fetch delta for {symbol}: {delta_e}. Using cached data.")
                                return data
                    except (KeyError, ValueError):
                        continue
                # Fallback generic read
                try:
                    data = pd.read_csv(freshest_path, parse_dates=True)
                    if not data.empty and len(data.columns) > 0:
                        first_col = data.columns[0]
                        if 'date' in first_col.lower():
                            data.set_index(first_col, inplace=True)
                            return data
                except Exception:
                    pass
                logger.warning(f"Could not properly load cached data for {symbol}, will fetch fresh data")
    except Exception as e:
        logger.error(f"Error loading cached data for {symbol}: {e}")

    try:
        # Try alternative data sources first for latest data
        provider_used = 'yf'
        if ALTERNATIVE_FETCHER_AVAILABLE:
            logger.info(f"Trying alternative data sources first for {symbol}...")
            try:
                alt_fetcher = AlternativeDataFetcher()
                data = alt_fetcher.get_historical_data(symbol, period=period, interval=interval)
                
                if not data.empty:
                    provider_used = 'alt'
                    logger.info(f"Successfully fetched {len(data)} data points for {symbol} from alternative sources")
                else:
                    logger.info(f"Alternative sources returned empty data for {symbol}, trying yfinance fallback")
                    # Fallback to yfinance if alternative sources fail
                    data = get_historical_data_with_retry(symbol, period=period, interval=interval)
                    provider_used = 'yf'
            except Exception as e:
                logger.error(f"Alternative data fetcher failed for {symbol}: {e}")
                # Fallback to yfinance if alternative sources fail
                data = get_historical_data_with_retry(symbol, period=period, interval=interval)
                provider_used = 'yf'
        else:
            # If alternative fetcher is not available, use yfinance
            logger.warning("Alternative data fetcher not available, using yfinance")
            data = get_historical_data_with_retry(symbol, period=period, interval=interval)
            provider_used = 'yf'

        if data.empty:
            logger.warning(f"No data found for {symbol} ({interval}) from any source.")
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
            os.makedirs(os.path.dirname(cache_dir), exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{symbol}_{period}_{interval}_{provider_used}.csv")
            data.to_csv(cache_path)
            _write_checksum(cache_path)
            logger.info(f"Fetched and cached {len(data)} data points for {symbol} ({interval}) provider={provider_used} -> {os.path.basename(cache_path)}")
        except Exception as e:
            logger.warning(f"Failed to cache data for {symbol}: {e}")

        return data

    except Exception as e:
        logger.error(f"Error fetching data for {symbol} ({interval}): {e}")
        return pd.DataFrame()

def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for a stock symbol with robust fallbacks."""
    try:
        # Try alternative data sources first for latest prices
        if ALTERNATIVE_FETCHER_AVAILABLE:
            try:
                alt_fetcher = AlternativeDataFetcher()
                alt_price = alt_fetcher.get_current_price(symbol)
                if alt_price and alt_price > 0:
                    logger.debug(f"Got current price for {symbol} from alternative sources: {alt_price}")
                    return float(alt_price)
            except Exception as e:
                logger.debug(f"Alternative price fetch failed for {symbol}: {e}")
        
        # Fallback to yfinance methods
        yf_symbol = f"{symbol}.NS"
        
        # Primary: yfinance info
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info or {}
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        if price:
            return float(price)
        
        # Fallback 1: last close from recent historical data
        hist = get_historical_data(symbol, period='5d', interval='1d')
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        
        # Fallback 2: direct download of 1d
        data = yf.download(yf_symbol, period='1d', interval='1d', progress=False, auto_adjust=True, threads=False)
        if not data.empty and 'Close' in data.columns:
            return float(data['Close'].iloc[-1])
        
        return None
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
            # Use configurable period (e.g., '1y') so we have enough data points for min_historical_days filtering
            hist_data = get_historical_data(symbol, FILTER_VALIDATION_PERIOD)
            
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
        
        # Delivery percentage filter
        delivery_pct = stock_info.get('delivery_percent', 0.0)
        min_delivery = filtering_criteria.get('min_delivery_percent', 0.0)
        if delivery_pct > 0 and delivery_pct < min_delivery:
            logger.debug(f"Skipping {symbol}: Delivery percent {delivery_pct:.1f}% below minimum {min_delivery:.1f}%")
            return (symbol, name, stock_info, False)
            
        # Volatility filter
        volatility_percentile = stock_info.get('volatility_percentile', 0.0)
        max_vol_percentile = filtering_criteria.get('max_volatility_percentile', 100)
        if volatility_percentile > max_vol_percentile:
            logger.debug(f"Skipping {symbol}: Volatility percentile {volatility_percentile:.1f} exceeds maximum {max_vol_percentile}")
            return (symbol, name, stock_info, False)
            
        # Status filters
        if filtering_criteria.get('exclude_delisted', True) and stock_info.get('is_delisted', False):
            logger.debug(f"Skipping {symbol}: Stock is delisted")
            return (symbol, name, stock_info, False)
            
        if filtering_criteria.get('exclude_suspended', True) and stock_info.get('is_suspended', False):
            logger.debug(f"Skipping {symbol}: Stock is suspended")
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
    
    # Get filtering criteria from config - enforce single source of truth
    filtering_criteria = {
        'min_volume': STOCK_FILTERING.get('min_volume', 100000),
        'min_price': STOCK_FILTERING.get('min_price', 5.0),
        'max_price': STOCK_FILTERING.get('max_price', 50000.0),
        'min_market_cap': STOCK_FILTERING.get('min_market_cap', 100000000),
        'min_historical_days': STOCK_FILTERING.get('min_historical_days', 200),
        'min_delivery_percent': STOCK_FILTERING.get('min_delivery_percent', 0.0),
        'max_volatility_percentile': STOCK_FILTERING.get('max_volatility_percentile', 80),
        'volume_lookback_days': STOCK_FILTERING.get('volume_lookback_days', 50),
        'exclude_delisted': STOCK_FILTERING.get('exclude_delisted', True),
        'exclude_suspended': STOCK_FILTERING.get('exclude_suspended', True)
    }
    
    if max_stocks is not None and max_stocks <= 20:
        logger.info(f"Applying strict config filters to {max_stocks} stocks")
    else:
        logger.info(f"Applying strict config filters to full scan of {len(symbols)} stocks")
    
    # Convert symbols to list of tuples for threading
    if isinstance(symbols, dict):
        symbol_list = list(symbols.items())
    elif isinstance(symbols, list):
        symbol_list = [(s, s) for s in symbols]
    else:
        logger.error(f"Unsupported symbols type: {type(symbols)}")
        return {}
    
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
            cache_seconds = FILTERED_SYMBOLS_CACHE_HOURS * 3600
            if file_age < cache_seconds:  # Configurable via FILTERED_SYMBOLS_CACHE_HOURS in config.py
                with open(cache_file, 'r') as f:
                    filtered_symbols = json.load(f)
                    logger.info(f"Loaded {len(filtered_symbols)} filtered symbols from cache.")
                    if filtered_symbols:  # Only return if not empty
                        return filtered_symbols
                    else:
                        logger.info("Cache file is empty, proceeding to use known stocks.")
        except Exception as e:
            logger.error(f"Error loading cached filtered symbols: {e}")
    
    # Dynamic Mode: Scan all ~2000 NSE symbols and filter based on strict live volume/price criteria
    logger.info("Executing live dynamic scanning over all NSE stocks...")
    
    # Fetch all raw symbols
    all_symbols = get_all_nse_symbols()
    
    # If the NSE API or fallback fails to give us anything, return empty
    if not all_symbols:
        logger.warning("Could not fetch base NSE symbols for filtering.")
        return {}
        
    # Process the entire list through filter_active_stocks to check live stats
    filtered_symbols = filter_active_stocks(all_symbols, max_stocks)
    
    # Save to cache
    try:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(filtered_symbols, f, indent=4)
        logger.info(f"Cached {len(filtered_symbols)} filtered symbols.")
    except Exception as e:
        logger.error(f"Error caching filtered symbols: {e}")
    
    return filtered_symbols

def apply_technical_filter(df: pd.DataFrame) -> bool:
    """
    Applies technical filtering rules identical to the Chartink scan clause.
    Uses settings from config.STOCK_FILTERING.
    """
    if df is None or len(df) < 200:
        return False
    
    import config
    rules = getattr(config, 'STOCK_FILTERING', {})
    
    try:
        import talib
        import numpy as np
        
        close = df['Close'].values
        high = df['High'].values
        open_p = df['Open'].values
        volume = df['Volume'].values
        
        # 1. SMAs
        if rules.get('require_above_sma50', True):
            sma50 = talib.SMA(close.astype(float), timeperiod=50)[-1]
            if close[-1] <= sma50: return False
            
        if rules.get('require_above_sma200', True):
            sma200 = talib.SMA(close.astype(float), timeperiod=200)[-1]
            if close[-1] <= sma200: return False
            
        # 2. Volume Spike
        spike_mult = rules.get('require_volume_spike', 2.0)
        if spike_mult > 0:
            vol_sma20 = talib.SMA(volume.astype(float), timeperiod=20)[-1]
            if volume[-1] <= (vol_sma20 * spike_mult): return False
            
        # 3. 20-day Breakout
        if rules.get('require_20day_breakout', True):
            prev_20_high = np.max(high[-21:-1])
            if close[-1] <= prev_20_high: return False
            
        # 4. RSI
        rsi_min = rules.get('require_rsi_above', 50.0)
        if rsi_min > 0:
            rsi = talib.RSI(close.astype(float), timeperiod=14)[-1]
            if rsi <= rsi_min: return False
            
        # 5. Bullish Candle
        if rules.get('require_bullish_candle', True):
            if close[-1] <= open_p[-1]: return False
            
        # 6. Strong Close
        close_threshold = rules.get('require_strong_close', 0.98)
        if close[-1] < (high[-1] * close_threshold): return False
            
        return True
    except Exception as e:
        logger.debug(f"Technical filter error: {e}")
        return False
