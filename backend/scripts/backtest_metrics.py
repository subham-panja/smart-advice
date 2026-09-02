"""
Portfolio Backtest Metrics & Helper Functions
==============================================

Provides performance calculation routines (CAGR, Sharpe, Drawdown, Profit Factor, Expectancy)
and snapshot recording for the portfolio backtest engine.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def calculate_portfolio_metrics(
    daily_snapshots: List[dict],
    trades: List[Any],
    initial_capital: float,
    cash: float,
    common_dates: pd.DatetimeIndex,
) -> Dict[str, Any]:
    """Calculate final portfolio performance metrics."""
    final_value = daily_snapshots[-1]["portfolio_value"] if daily_snapshots else cash
    total_return_pct = ((final_value - initial_capital) / initial_capital) * 100

    days = (common_dates[-1] - common_dates[0]).days if len(common_dates) > 1 else 1
    years = days / 365.25
    if years >= 1.0:
        cagr = ((final_value / initial_capital) ** (1 / years) - 1) * 100
    else:
        cagr = total_return_pct / years if years > 0.01 else 0.0

    max_dd_pct = min((s["drawdown_from_peak_pct"] for s in daily_snapshots), default=0)

    completed_trades = [t for t in trades if getattr(t, "trade_type", "") in ("SELL", "PARTIAL_SELL")]
    winning_trades = [t for t in completed_trades if getattr(t, "pnl", 0) > 0]
    total_trades = len(completed_trades)
    win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = abs(sum(t.pnl for t in completed_trades if t.pnl <= 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    avg_win = gross_profit / len(winning_trades) if winning_trades else 0
    avg_loss = gross_loss / (total_trades - len(winning_trades)) if total_trades > len(winning_trades) else 1
    expectancy = ((win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)) if total_trades > 0 else 0.0

    sharpe = 0.0
    if len(daily_snapshots) > 1:
        pv = np.array([s["portfolio_value"] for s in daily_snapshots], dtype=np.float64)
        mask = pv[:-1] > 0
        daily_returns = np.where(mask, (pv[1:] - pv[:-1]) / np.where(mask, pv[:-1], 1.0), 0.0)
        std = np.std(daily_returns)
        if std > 0:
            sharpe = float(np.mean(daily_returns) / std * np.sqrt(252))

    avg_positions = (
        sum(s["open_positions_count"] for s in daily_snapshots) / len(daily_snapshots) if daily_snapshots else 0
    )

    return {
        "final_portfolio_value": round(final_value, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr": round(cagr, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "sharpe_ratio": round(sharpe, 2),
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(expectancy, 2),
        "avg_positions_held": round(avg_positions, 2),
    }


def check_market_breadth(
    strategy_config: dict,
    indicator_store: Any,
    date: pd.Timestamp,
    symbols_data: Dict[str, pd.DataFrame],
    date_idx_map: dict,
) -> bool:
    """Calculate market breadth across stock universe."""
    bread_cfg = strategy_config.get("market_breadth_filter", {})
    if not bread_cfg.get("enabled", True):
        return True

    min_advance_pct = bread_cfg.get("min_advance_pct", 35)

    if indicator_store is not None:
        above_count = 0
        total = 0
        for symbol in symbols_data.keys():
            if date not in symbols_data[symbol].index:
                continue
            close_val = indicator_store.get(symbol, "close", date)
            sma_val = indicator_store.get(symbol, "sma_20", date)
            if np.isnan(close_val) or np.isnan(sma_val):
                continue
            total += 1
            if close_val > sma_val:
                above_count += 1
    else:
        sma_period = 20
        above_count = 0
        total = 0
        for symbol, df in symbols_data.items():
            if date not in df.index:
                continue
            idx = date_idx_map.get(symbol, {}).get(date)
            if idx is None:
                idx = df.index.searchsorted(date)
            hist = df.iloc[: idx + 1]
            if len(hist) < sma_period:
                continue
            total += 1
            sma = hist["Close"].tail(sma_period).mean()
            if hist["Close"].iloc[-1] > sma:
                above_count += 1

    if total < 5:
        return True

    advance_pct = (above_count / total) * 100
    return advance_pct >= min_advance_pct


def _calculate_series_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Calculate ADX on price series."""
    try:
        import talib

        adx = talib.ADX(
            high.astype(float).values, low.astype(float).values, close.astype(float).values, timeperiod=period
        )
        val = adx[-1]
        if not np.isnan(val):
            return float(val)
    except Exception:
        pass

    if len(close) < period * 2:
        return 25.0

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (
        pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    )
    minus_di = 100 * (
        pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    )

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / di_sum) * 100
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    val = adx.iloc[-1]
    return float(val) if not np.isnan(val) else 25.0


def check_market_regime(
    regime_enabled: bool,
    regime_config: dict,
    date: pd.Timestamp,
    index_data_override: Optional[pd.DataFrame],
    symbols_data: Dict[str, pd.DataFrame],
    date_idx_map: dict,
    logger_obj: Any,
) -> str:
    """Check market regime (BULL/BEAR) using index data."""
    if not regime_enabled or not regime_config:
        raise RuntimeError(
            "Market regime detection is DISABLED but required by this strategy. "
            "Set 'market_regime_detection': true in analysis_config of your strategy JSON."
        )

    index_symbol = regime_config.get("index", "^NSEI")
    if index_data_override is not None:
        index_df = index_data_override
        idx = index_df.index.searchsorted(date)
        index_hist = index_df.iloc[: idx + 1]
    elif index_symbol in symbols_data:
        index_df = symbols_data[index_symbol]
        idx = date_idx_map.get(index_symbol, {}).get(date)
        if idx is None:
            idx = index_df.index.searchsorted(date)
        index_hist = index_df.iloc[: idx + 1]
    else:
        from scripts.data_fetcher import get_historical_data

        full_data = get_historical_data(index_symbol, period="10y")
        if date.tzinfo is not None and full_data.index.tzinfo is None:
            full_data.index = full_data.index.tz_localize(date.tzinfo)
        index_hist = full_data.loc[:date]

    rule = regime_config.get("bull_market_rule", "latest close > sma(200)")
    import re

    sma_match = re.search(r"sma\((\d+)\)", rule)
    sma_period = int(sma_match.group(1)) if sma_match else 200

    min_required = min(250, sma_period)
    if len(index_hist) < min_required:
        if len(index_hist) < 30:
            return "BULL"
        logger_obj.warning(
            f"Index history {len(index_hist)} days < {min_required} for regime check; using available {len(index_hist)} bars."
        )

    current_price = index_hist["Close"].iloc[-1]
    effective_window = min(sma_period, len(index_hist))
    sma_series = index_hist["Close"].rolling(effective_window, min_periods=30).mean()
    sma_value = sma_series.iloc[-1]

    is_bull = current_price > sma_value

    # Check SMA slope if required
    require_slope = regime_config.get("require_sma_slope_up", False)
    slope_bars = int(regime_config.get("sma_slope_lookback_bars", 5))
    if is_bull and require_slope and len(sma_series) > slope_bars:
        past_sma = sma_series.iloc[-1 - slope_bars]
        if not np.isnan(past_sma) and past_sma > 0:
            if sma_value < past_sma:
                is_bull = False

    # Check Index ADX trend strength if configured
    min_adx = float(regime_config.get("min_adx", 0))
    if is_bull and min_adx > 0 and "High" in index_hist and "Low" in index_hist:
        idx_adx = _calculate_series_adx(index_hist["High"], index_hist["Low"], index_hist["Close"], 14)
        if idx_adx < min_adx:
            is_bull = False

    regime_status = "BULL" if is_bull else "BEAR"

    if not is_bull:
        logger_obj.info(f"🔴 MACRO REGIME: BEARISH on {date.date()} - {index_symbol} condition not met")

    return regime_status


def record_daily_snapshot(
    snapshots_list: List[dict],
    date: pd.Timestamp,
    portfolio_value: float,
    cash: float,
    peak_value: float,
    positions: Dict[str, Any],
) -> float:
    """Record daily portfolio state snapshot and return updated peak_value."""
    market_value = portfolio_value - cash

    new_peak = max(peak_value, portfolio_value)
    drawdown = portfolio_value - new_peak
    drawdown_pct = (drawdown / new_peak) * 100 if new_peak > 0 else 0

    snapshots_list.append(
        {
            "date": str(date.date()),
            "portfolio_value": round(portfolio_value, 2),
            "cash_balance": round(cash, 2),
            "market_value": round(market_value, 2),
            "open_positions_count": len(positions),
            "open_positions": list(positions.keys()),
            "drawdown_from_peak": round(drawdown, 2),
            "drawdown_from_peak_pct": round(drawdown_pct, 2),
        }
    )
    return new_peak
