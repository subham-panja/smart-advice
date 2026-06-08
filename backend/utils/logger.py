import logging
import os
import queue

log_queue = queue.Queue(maxsize=1000)

_verbose_mode = False


def is_verbose():
    return _verbose_mode


class SSEHandler(logging.Handler):
    """Pushes logs to a queue for SSE streaming."""

    def emit(self, record):
        try:
            msg = self.format(record)
            try:
                log_queue.put_nowait(msg)
            except queue.Full:
                try:
                    log_queue.get_nowait()
                    log_queue.put_nowait(msg)
                except Exception:
                    pass
        except Exception:
            self.handleError(record)


def setup_logging(level=logging.INFO, verbose=None):
    global _verbose_mode

    os.makedirs("logs", exist_ok=True)

    if verbose is not None:
        _verbose_mode = verbose
        level = logging.DEBUG if verbose else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = []

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    h1 = logging.FileHandler("logs/app.log", mode="a")
    h1.setFormatter(fmt)
    root.addHandler(h1)

    sse_handler = SSEHandler()
    sse_fmt = logging.Formatter("%(levelname)s | %(message)s")
    sse_handler.setFormatter(sse_fmt)
    if not _verbose_mode:
        sse_handler.setLevel(logging.WARNING)
    else:
        sse_handler.setLevel(logging.INFO)
    root.addHandler(sse_handler)

    if verbose:
        h3 = logging.StreamHandler()
        h3.setFormatter(fmt)
        root.addHandler(h3)

    for name in ["yfinance", "urllib3", "requests"]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    return logging.getLogger(__name__)


def set_verbose(verbose: bool):
    """Dynamically toggle verbose mode for SSE streaming."""
    global _verbose_mode
    _verbose_mode = verbose

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    for handler in root.handlers:
        if isinstance(handler, SSEHandler):
            handler.setLevel(logging.INFO if verbose else logging.WARNING)
        elif isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG if verbose else logging.INFO)
