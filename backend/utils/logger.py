import logging
import os
import queue

log_queue = queue.Queue(maxsize=1000)

class SSEHandler(logging.Handler):
    """Pushes logs to a queue for SSE streaming."""
    def emit(self, record):
        try:
            msg = self.format(record)
            try: log_queue.put_nowait(msg)
            except queue.Full:
                try: log_queue.get_nowait(); log_queue.put_nowait(msg)
                except: pass
        except: self.handleError(record)

def setup_logging(level=logging.INFO, verbose=None):
    os.makedirs('logs', exist_ok=True)
    if verbose is not None: level = logging.INFO if verbose else logging.CRITICAL
    
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []
    
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    h1 = logging.FileHandler("logs/app.log")
    h1.setFormatter(fmt)
    root.addHandler(h1)
    
    h2 = SSEHandler()
    h2.setFormatter(fmt)
    root.addHandler(h2)
    
    if verbose:
        h3 = logging.StreamHandler()
        h3.setFormatter(fmt)
        root.addHandler(h3)
    
    for name in ['yfinance', 'urllib3', 'requests']:
        logging.getLogger(name).setLevel(logging.CRITICAL)
        
    return logging.getLogger(__name__)
