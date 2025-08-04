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
        log_level = logging.INFO if verbose else logging.ERROR
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up handlers
    file_handler = logging.FileHandler("logs/app.log")
    stream_handler = logging.StreamHandler()
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    
    # Also configure all existing loggers to inherit from root
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.NOTSET)  # Inherit from parent/root
    
    return logging.getLogger(__name__)
