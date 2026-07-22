import logging
import os
import queue
from typing import Optional

# Register custom log level IMPORTANT (Level 25: between INFO 20 and WARNING 30)
IMPORTANT_LEVEL_NUM = 25
logging.addLevelName(IMPORTANT_LEVEL_NUM, "IMPORTANT")


def _important_logger_method(self, message, *args, **kws):
    if self.isEnabledFor(IMPORTANT_LEVEL_NUM):
        self._log(IMPORTANT_LEVEL_NUM, message, args, **kws)


# Patch Logger class so any logger.important(...) call works natively
logging.Logger.important = _important_logger_method  # type: ignore

log_queue = queue.Queue(maxsize=1000)
_verbose_mode = False


def is_verbose() -> bool:
    return _verbose_mode


def log_important(msg: str):
    """Standalone helper to log an important message on root logger."""
    logging.getLogger().important(msg)


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


def setup_logging(level: Optional[int] = None, verbose: Optional[bool] = None) -> logging.Logger:
    """Configures system-wide logging driven by config.VERBOSE."""
    global _verbose_mode

    os.makedirs("logs", exist_ok=True)

    # Automatically read config.VERBOSE if verbose is not explicitly passed
    if verbose is None:
        try:
            import config

            verbose = getattr(config, "VERBOSE", False)
        except Exception:
            verbose = False

    _verbose_mode = bool(verbose)

    if level is None:
        level = logging.DEBUG if _verbose_mode else logging.INFO

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # Root accepts all, handlers filter
    root.handlers = []

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler (logs all details to logs/app.log)
    h1 = logging.FileHandler("logs/app.log", mode="a")
    h1.setFormatter(fmt)
    h1.setLevel(logging.DEBUG if _verbose_mode else logging.INFO)
    root.addHandler(h1)

    # SSE handler for frontend dashboard
    sse_handler = SSEHandler()
    sse_fmt = logging.Formatter("%(levelname)s | %(message)s")
    sse_handler.setFormatter(sse_fmt)
    sse_handler.setLevel(logging.INFO if _verbose_mode else IMPORTANT_LEVEL_NUM)
    root.addHandler(sse_handler)

    # Console Stream Handler (Terminal stdout output)
    console_handler = logging.StreamHandler()
    if _verbose_mode:
        console_handler.setFormatter(fmt)
        console_handler.setLevel(logging.DEBUG)
    else:
        # Clean output: message only, no timestamp/level prefix
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.setLevel(IMPORTANT_LEVEL_NUM)
    root.addHandler(console_handler)

    # Silence noisy third-party loggers
    for name in [
        "yfinance",
        "urllib3",
        "requests",
        "pymongo",
        "asyncio",
        "multiprocessing",
        "peewee",
        "tvDatafeed",
        "chardet",
        "charset_normalizer",
    ]:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    return logging.getLogger(__name__)


def set_verbose(verbose: bool):
    """Dynamically toggle verbose mode."""
    global _verbose_mode
    _verbose_mode = verbose

    try:
        import config

        config.VERBOSE = verbose
    except Exception:
        pass

    root = logging.getLogger()

    full_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    clean_fmt = logging.Formatter("%(message)s")

    for handler in root.handlers:
        if isinstance(handler, SSEHandler):
            handler.setLevel(logging.INFO if verbose else IMPORTANT_LEVEL_NUM)
        elif isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG if verbose else IMPORTANT_LEVEL_NUM)
            handler.setFormatter(full_fmt if verbose else clean_fmt)
        elif isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG if verbose else logging.INFO)
