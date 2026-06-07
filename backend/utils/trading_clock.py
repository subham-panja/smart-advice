from datetime import datetime

_simulated_date = None


def set_simulated_date(dt):
    global _simulated_date
    _simulated_date = dt


def get_simulated_date():
    return _simulated_date


def trading_now(tz=None):
    if _simulated_date is not None:
        return _simulated_date
    return datetime.now(tz)


def is_replay():
    return _simulated_date is not None
