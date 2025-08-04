import logging
import os

def setup_logging(log_level=logging.INFO, verbose=None):
    """Set up logging configuration.
    
    Args:
        log_level: Default log level
        verbose: If True, use INFO level; if False, use ERROR level; if None, use log_level
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Override log_level based on verbose parameter
    if verbose is not None:
        log_level = logging.INFO if verbose else logging.CRITICAL  # Use CRITICAL instead of ERROR to suppress more
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up handlers
    file_handler = logging.FileHandler("logs/app.log")
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add file handler (always log to file)
    root_logger.addHandler(file_handler)
    
    # Only add stream handler in verbose mode
    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    
    # Aggressively configure all existing and future loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        # Clear handlers to prevent duplicate logging
        logger.handlers = []
        logger.propagate = True  # Ensure they propagate to root logger
    
    # Configure specific noisy modules in non-verbose mode
    if not verbose:
        # Suppress strategy-related logging
        logging.getLogger('utils.logger').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.technical_analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.strategy_evaluator').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.fundamental_analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.sentiment_analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.sector_analyzer').setLevel(logging.CRITICAL)
        
        # Suppress third-party logging
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('requests').setLevel(logging.CRITICAL)
    
    return logging.getLogger(__name__)
