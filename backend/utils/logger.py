import logging
import os
import queue

# Global thread-safe queue for SSE logs
log_queue = queue.Queue(maxsize=1000)

class SSEHandler(logging.Handler):
    """Custom logging handler that pushes logs to a queue for SSE streaming."""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Use non-blocking put with try-except to avoid stalling if queue is full
            try:
                log_queue.put_nowait(msg)
            except queue.Full:
                # Remove oldest message and try again if full
                try:
                    log_queue.get_nowait()
                    log_queue.put_nowait(msg)
                except (queue.Empty, queue.Full):
                    pass
        except Exception:
            self.handleError(record)

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
        log_level = logging.INFO if verbose else logging.CRITICAL
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up handlers
    from config import PERSIST_LOGGING
    log_mode = 'a' if PERSIST_LOGGING else 'w'
    file_handler = logging.FileHandler("logs/app.log", mode=log_mode)
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add file handler (always log to file)
    root_logger.addHandler(file_handler)
    
    # Add SSE handler (always available for streaming)
    sse_handler = SSEHandler()
    sse_handler.setFormatter(formatter)
    root_logger.addHandler(sse_handler)
    
    # Only add stream handler in verbose mode
    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    
    # Aggressively configure all existing and future loggers
    for logger_name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        logger.handlers = []
        logger.propagate = True
    
    # Configure specific noisy modules in non-verbose mode
    if not verbose:
        logging.getLogger('utils.logger').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.technical_analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.strategy_evaluator').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.fundamental_analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.sentiment_analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('scripts.sector_analyzer').setLevel(logging.CRITICAL)
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)
        logging.getLogger('urllib3').setLevel(logging.CRITICAL)
        logging.getLogger('requests').setLevel(logging.CRITICAL)
    
    return logging.getLogger(__name__)
