use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Position {
    symbol_idx: usize,
    entry_bar: usize,
    entry_price: f64,
    quantity: i64,
    stop_loss: f64,
    current_stop: f64,
    targets_hit: i32,
    is_scaled_out: bool,
}

#[derive(Clone, Debug)]
struct Trade {
    symbol_idx: usize,
    entry_bar: usize,
    entry_price: f64,
    exit_bar: usize,
    exit_price: f64,
    quantity: i64,
    pnl: f64,
    pnl_pct: f64,
    exit_reason: String,
}

#[derive(Clone)]
struct StrategyConfig {
    risk_per_trade: f64,
    max_position_pct: f64,
    max_positions: usize,
    brokerage: f64,
    slippage: f64,
    stop_loss_pct: f64,
    t1_sell_pct: f64,
    t1_target_pct: f64,
    time_stop_bars: usize,
    atr_stop_multiplier: f64,
    trailing_stop_enabled: bool,
}

fn parse_config(py_config: &Bound<'_, PyDict>) -> StrategyConfig {
    let get_f64 = |key: &str, default: f64| -> f64 {
        py_config.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<f64>().ok())
            .unwrap_or(default)
    };
    let get_usize = |key: &str, default: usize| -> usize {
        py_config.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<usize>().ok())
            .unwrap_or(default)
    };
    let get_bool = |key: &str, default: bool| -> bool {
        py_config.get_item(key).ok().flatten()
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(default)
    };

    StrategyConfig {
        risk_per_trade: get_f64("risk_per_trade", 0.02),
        max_position_pct: get_f64("max_position_pct", 0.10),
        max_positions: get_usize("max_positions", 15),
        brokerage: get_f64("brokerage", 0.002),
        slippage: get_f64("slippage", 0.0005),
        stop_loss_pct: get_f64("stop_loss_pct", 0.08),
        t1_sell_pct: get_f64("t1_sell_pct", 0.50),
        t1_target_pct: get_f64("t1_target_pct", 0.20),
        time_stop_bars: get_usize("time_stop_bars", 20),
        atr_stop_multiplier: get_f64("atr_stop_multiplier", 3.0),
        trailing_stop_enabled: get_bool("trailing_stop_enabled", true),
    }
}

/// Core simulation: runs a portfolio backtest with pre-computed signals.
///
/// prices: list of (N, 5) numpy arrays [Open, High, Low, Close, Volume] per symbol
/// signals: dict mapping symbol_idx -> list of (bar_idx, score) tuples
/// config: strategy configuration dict
/// initial_capital: starting capital
///
/// Returns: dict with trades, cagr, sharpe, max_drawdown, total_return, etc.
#[pyfunction]
fn run_backtest(
    py: Python<'_>,
    prices: Vec<numpy::PyReadonlyArray2<f64>>,
    signals: HashMap<usize, Vec<(usize, f64)>>,
    config: &Bound<'_, PyDict>,
    initial_capital: f64,
) -> PyResult<PyObject> {
    let cfg = parse_config(config);

    let num_symbols = prices.len();
    let num_bars = if num_symbols > 0 { prices[0].as_array().nrows() } else { 0 };
    if num_bars < 60 {
        let result = PyDict::new_bound(py);
        result.set_item("status", "failed")?;
        result.set_item("cagr", 0.0)?;
        return Ok(result.into());
    }

    let mut cash = initial_capital;
    let mut peak_value = initial_capital;
    let mut positions: Vec<Position> = Vec::new();
    let mut trades: Vec<Trade> = Vec::new();
    let mut equity_curve: Vec<f64> = Vec::with_capacity(num_bars);

    // Build signal lookup: bar -> list of (symbol_idx, score)
    let mut bar_signals: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for (sym_idx, sig_list) in &signals {
        for (bar, score) in sig_list {
            bar_signals.entry(*bar).or_default().push((*sym_idx, *score));
        }
    }

    for bar in 0..num_bars {
        // --- Phase 1: Process exits ---
        let mut to_close: Vec<(usize, f64, String)> = Vec::new();
        let mut stop_updates: Vec<(usize, f64)> = Vec::new();
        let mut partial_sells: Vec<(usize, i64, f64)> = Vec::new();

        for (i, pos) in positions.iter().enumerate() {
            let sym = pos.symbol_idx;
            if sym >= num_symbols { continue; }
            let arr = prices[sym].as_array();
            if bar >= arr.nrows() { continue; }

            let close = arr[[bar, 3]];
            let bars_held = bar - pos.entry_bar;

            // O'Neil fixed stop loss
            if close <= pos.current_stop {
                to_close.push((i, pos.current_stop, "STOP_LOSS".to_string()));
                continue;
            }

            // Trailing stop update (collect for later application)
            if cfg.trailing_stop_enabled && close > pos.entry_price {
                let gain_pct = (close - pos.entry_price) / pos.entry_price;
                if gain_pct > 0.10 {
                    let new_stop = close * (1.0 - cfg.stop_loss_pct * 0.5);
                    if new_stop > pos.current_stop {
                        stop_updates.push((i, new_stop));
                    }
                }
            }

            // O'Neil target (T1) — collect partial sell
            let gain_pct = (close - pos.entry_price) / pos.entry_price * 100.0;
            if gain_pct >= cfg.t1_target_pct && pos.targets_hit == 0 {
                let sell_qty = (pos.quantity as f64 * cfg.t1_sell_pct) as i64;
                if sell_qty > 0 {
                    partial_sells.push((i, sell_qty, close));
                }
            }

            // Time stop
            if bars_held >= cfg.time_stop_bars && pos.targets_hit == 0 {
                let gain_pct = (close - pos.entry_price) / pos.entry_price * 100.0;
                if gain_pct < 5.0 {
                    to_close.push((i, close, "TIME_STOP".to_string()));
                }
            }
        }

        // Apply trailing stop updates
        for (idx, new_stop) in stop_updates {
            if idx < positions.len() {
                positions[idx].current_stop = new_stop;
            }
        }

        // Apply partial sells
        for (idx, sell_qty, close) in partial_sells {
            if idx >= positions.len() { continue; }
            let sell_price = close * (1.0 - cfg.slippage);
            let cost = sell_price * sell_qty as f64 * cfg.brokerage;
            cash += sell_price * sell_qty as f64 - cost;
            positions[idx].quantity -= sell_qty;
            positions[idx].targets_hit = 1;

            if positions[idx].quantity <= 0 {
                to_close.push((idx, sell_price, format!("ONEIL_TARGET_{:.0}%", cfg.t1_target_pct)));
            }
        }

        // Execute closes (reverse order to preserve indices)
        to_close.sort_by(|a, b| b.0.cmp(&a.0));
        for (idx, exit_price, reason) in to_close {
            if idx >= positions.len() { continue; }
            let pos = &positions[idx];
            let sell_price = exit_price * (1.0 - cfg.slippage);
            let revenue = sell_price * pos.quantity as f64;
            let cost = revenue * cfg.brokerage;
            cash += revenue - cost;

            let pnl = (sell_price - pos.entry_price) * pos.quantity as f64 - cost;
            let pnl_pct = (sell_price - pos.entry_price) / pos.entry_price * 100.0;

            trades.push(Trade {
                symbol_idx: pos.symbol_idx,
                entry_bar: pos.entry_bar,
                entry_price: pos.entry_price,
                exit_bar: bar,
                exit_price: sell_price,
                quantity: pos.quantity,
                pnl,
                pnl_pct,
                exit_reason: reason,
            });
            positions.remove(idx);
        }

        // --- Phase 2: Process entries from pre-computed signals ---
        if let Some(candidates) = bar_signals.get(&bar) {
            let mut sorted_candidates = candidates.clone();
            sorted_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            for (sym_idx, score) in sorted_candidates {
                if positions.len() >= cfg.max_positions { break; }
                if positions.iter().any(|p| p.symbol_idx == sym_idx) { continue; }
                if sym_idx >= num_symbols { continue; }

                let arr = prices[sym_idx].as_array();
                if bar >= arr.nrows() { continue; }

                let close = arr[[bar, 3]];
                let exec_price = close * (1.0 + cfg.slippage);

                // Position sizing: risk-based
                let portfolio_value = calc_portfolio_value(cash, &positions, &prices, bar);
                let max_alloc = portfolio_value * cfg.max_position_pct;
                let risk_amount = portfolio_value * cfg.risk_per_trade;
                let stop_distance = exec_price * cfg.stop_loss_pct;
                if stop_distance <= 0.0 { continue; }

                let quantity = ((risk_amount / stop_distance).min(max_alloc / exec_price)) as i64;
                if quantity <= 0 { continue; }

                let cost = exec_price * quantity as f64 * cfg.brokerage;
                let total_cost = exec_price * quantity as f64 + cost;
                if total_cost > cash { continue; }

                cash -= total_cost;
                let stop_loss = exec_price * (1.0 - cfg.stop_loss_pct);

                positions.push(Position {
                    symbol_idx: sym_idx,
                    entry_bar: bar,
                    entry_price: exec_price,
                    quantity,
                    stop_loss,
                    current_stop: stop_loss,
                    targets_hit: 0,
                    is_scaled_out: false,
                });
            }
        }

        // --- Phase 3: Record equity ---
        let portfolio_value = calc_portfolio_value(cash, &positions, &prices, bar);
        equity_curve.push(portfolio_value);
        if portfolio_value > peak_value {
            peak_value = portfolio_value;
        }
    }

    // Force-close remaining positions
    for pos in &positions {
        if pos.symbol_idx >= num_symbols { continue; }
        let arr = prices[pos.symbol_idx].as_array();
        let last_bar = arr.nrows() - 1;
        let close = arr[[last_bar, 3]];
        let sell_price = close * (1.0 - cfg.slippage);
        let revenue = sell_price * pos.quantity as f64;
        let cost = revenue * cfg.brokerage;
        cash += revenue - cost;

        let pnl = (sell_price - pos.entry_price) * pos.quantity as f64;
        let pnl_pct = (sell_price - pos.entry_price) / pos.entry_price * 100.0;
        trades.push(Trade {
            symbol_idx: pos.symbol_idx,
            entry_bar: pos.entry_bar,
            entry_price: pos.entry_price,
            exit_bar: last_bar,
            exit_price: sell_price,
            quantity: pos.quantity,
            pnl,
            pnl_pct,
            exit_reason: "END_OF_DATA".to_string(),
        });
    }

    // --- Calculate metrics ---
    let final_value = equity_curve.last().copied().unwrap_or(initial_capital);
    let total_return_pct = (final_value - initial_capital) / initial_capital * 100.0;
    let years = num_bars as f64 / 252.0;
    let cagr = if years > 0.0 && final_value > 0.0 {
        ((final_value / initial_capital).powf(1.0 / years) - 1.0) * 100.0
    } else {
        0.0
    };

    // Max drawdown
    let mut max_dd = 0.0f64;
    let mut running_peak = initial_capital;
    for &val in &equity_curve {
        if val > running_peak { running_peak = val; }
        let dd = (val - running_peak) / running_peak * 100.0;
        if dd < max_dd { max_dd = dd; }
    }

    // Sharpe ratio
    let mut daily_returns: Vec<f64> = Vec::new();
    for i in 1..equity_curve.len() {
        if equity_curve[i - 1] > 0.0 {
            daily_returns.push((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]);
        }
    }
    let sharpe = if daily_returns.len() > 1 {
        let mean = daily_returns.iter().sum::<f64>() / daily_returns.len() as f64;
        let variance = daily_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / daily_returns.len() as f64;
        let std = variance.sqrt();
        if std > 0.0 { (mean / std) * (252.0f64).sqrt() } else { 0.0 }
    } else {
        0.0
    };

    // Win rate and profit factor
    let winning_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl > 0.0).collect();
    let losing_trades: Vec<&Trade> = trades.iter().filter(|t| t.pnl <= 0.0).collect();
    let win_rate = if !trades.is_empty() {
        winning_trades.len() as f64 / trades.len() as f64 * 100.0
    } else {
        0.0
    };
    let gross_profit: f64 = winning_trades.iter().map(|t| t.pnl).sum();
    let gross_loss: f64 = losing_trades.iter().map(|t| t.pnl.abs()).sum();
    let profit_factor = if gross_loss > 0.0 { gross_profit / gross_loss } else { 0.0 };
    let expectancy = if !trades.is_empty() {
        trades.iter().map(|t| t.pnl).sum::<f64>() / trades.len() as f64
    } else {
        0.0
    };

    // Build result dict
    let result = PyDict::new_bound(py);
    result.set_item("status", "completed")?;
    result.set_item("initial_capital", initial_capital)?;
    result.set_item("final_portfolio_value", final_value)?;
    result.set_item("cash_remaining", cash)?;
    result.set_item("total_return_pct", total_return_pct)?;
    result.set_item("cagr", cagr)?;
    result.set_item("max_drawdown_pct", max_dd)?;
    result.set_item("sharpe_ratio", sharpe)?;
    result.set_item("total_trades", trades.len())?;
    result.set_item("win_rate", win_rate)?;
    result.set_item("profit_factor", profit_factor)?;
    result.set_item("expectancy", expectancy)?;

    let py_trades = PyList::empty_bound(py);
    for t in &trades {
        let td = PyDict::new_bound(py);
        td.set_item("symbol_idx", t.symbol_idx)?;
        td.set_item("entry_bar", t.entry_bar)?;
        td.set_item("entry_price", t.entry_price)?;
        td.set_item("exit_bar", t.exit_bar)?;
        td.set_item("exit_price", t.exit_price)?;
        td.set_item("quantity", t.quantity)?;
        td.set_item("pnl", t.pnl)?;
        td.set_item("pnl_pct", t.pnl_pct)?;
        td.set_item("exit_reason", &t.exit_reason)?;
        py_trades.append(td)?;
    }
    result.set_item("trades", py_trades)?;

    Ok(result.into())
}

fn calc_portfolio_value(
    cash: f64,
    positions: &[Position],
    prices: &[numpy::PyReadonlyArray2<f64>],
    bar: usize,
) -> f64 {
    let mut value = cash;
    for pos in positions {
        if pos.symbol_idx < prices.len() {
            let arr = prices[pos.symbol_idx].as_array();
            if bar < arr.nrows() {
                value += arr[[bar, 3]] * pos.quantity as f64; // Close price
            } else {
                value += pos.entry_price * pos.quantity as f64;
            }
        }
    }
    value
}

#[pymodule]
fn portfolio_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_backtest, m)?)?;
    Ok(())
}
