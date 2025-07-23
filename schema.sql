DROP TABLE IF EXISTS recommended_shares;
CREATE TABLE recommended_shares (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL UNIQUE,
    company_name TEXT,
    technical_score REAL,
    fundamental_score REAL,
    sentiment_score REAL,
    recommendation_date TEXT DEFAULT CURRENT_TIMESTAMP,
    reason TEXT,
    buy_price REAL,
    sell_price REAL,
    est_time_to_target TEXT,
    -- Backtesting metrics as JSON object
    backtest_metrics TEXT
);

DROP TABLE IF EXISTS backtest_results;
CREATE TABLE backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    period TEXT NOT NULL,
    CAGR REAL,
    win_rate REAL,
    max_drawdown REAL,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    avg_trade_duration REAL,
    avg_profit_per_trade REAL,
    avg_loss_per_trade REAL,
    largest_win REAL,
    largest_loss REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    calmar_ratio REAL,
    volatility REAL,
    start_date TEXT,
    end_date TEXT,
    initial_capital REAL,
    final_capital REAL,
    total_return REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
