# Smart Advice - AI-Powered Stock Analysis Platform

Smart Advice is a comprehensive stock market analysis application that provides intelligent recommendations and detailed analytics to help traders and investors make informed decisions. The platform combines advanced machine learning algorithms with traditional financial analysis across multiple dimensions including technical patterns, fundamental metrics, and market sentiment.

![Smart Advice Dashboard](https://img.shields.io/badge/Status-Active-green)
![Frontend](https://img.shields.io/badge/Frontend-Next.js_15-blue)
![Backend](https://img.shields.io/badge/Backend-Flask-lightgrey)
![Database](https://img.shields.io/badge/Database-MongoDB-green)

## 🌟 Features

### Core Analysis Capabilities
- **Technical Analysis**: Advanced technical indicators and chart pattern recognition using TA-Lib
- **Fundamental Analysis**: Financial metrics and company performance evaluation
- **Sentiment Analysis**: Market sentiment and news analysis using NLP models
- **Multi-Strategy Backtesting**: Comprehensive backtesting with CAGR, win rate, and expectancy calculations
- **Risk Management**: Advanced position sizing, risk-reward evaluation, and ATR-based stop losses

### Trading & Execution
- **Paper Trading Engine**: Full paper trading simulation with portfolio tracking and PnL monitoring
- **Unified Trading Cycle**: Automated end-to-end pipeline from analysis to execution
- **Pyramiding Support**: ATR-triggered position scaling with configurable add steps
- **Circuit Breaker**: Global trading halt flag for risk control
- **Telegram Bot**: Remote control bot for running analysis, viewing recommendations, positions, and broker balance

### Strategy System
- **JSON-Based Strategies**: Self-contained strategy configs in `backend/strategies/` (e.g., `Hybrid_Trading`, `Momentum_Trading`, `Swing_Trading`)
- **Swing Trading Gates**: 
  - **TREND_GATE**: ADX strength + DI alignment + price above SMA 50/150/200 stack
  - **VOLATILITY_GATE**: ATR must be in bottom 30% of 100-day lookback (volatility contraction)
  - **VOLUME_GATE**: Volume >= 80% of 20-day average + positive OBV trend slope (accumulation)
  - **MTF_GATE**: Multi-timeframe weekly trend confirmation
- **Entry Patterns**: Pullback to EMA, Bollinger Squeeze Breakout, MACD Zero Cross, Higher Low Structure, Volatility Contraction, NR7 Volatility Squeeze, 20-Day High Breakout
- **Multi-Target Exits**: ATR-based partial targets (T1: 3x, T2: 5x), breakeven at T1, trailing stop loss, and time-stop exits

### Advanced Analytics
- **Options OI Analysis**: NSE Option Chain analysis with PCR and unwinding detection
- **Smart Money Tracker**: FII/DII flow tracking and delivery volume analysis
- **Confluence Engine**: Multi-timeframe signal confluence analysis
- **Market Regime Detection**: HMM-based automatic detection of bullish/bearish market conditions
- **Screener.in Integration**: Web-scraping based fundamental stock screening
- **Machine Learning Models**: PyTorch LSTMs and custom classifiers for price prediction
- **Reinforcement Learning**: Trading agents for decision optimization (stable-baselines3)
- **Sector Analysis**: Industry-specific insights and comparisons
- **Market Microstructure**: Order flow and liquidity analysis
- **Sentiment Analysis**: NLP-based news and market sentiment (HuggingFace Transformers)

### Modern Web Interface
- **Real-time Dashboard**: Interactive charts and visualizations using Chart.js
- **Dark Mode Support**: Toggle between light and dark themes
- **Responsive Design**: Optimized for desktop and mobile devices
- **Progress Tracking**: Real-time analysis progress monitoring

## 🏗️ Architecture

### Frontend (Next.js 15 + React 19 + TypeScript)
- **Framework**: Next.js 15.5 with App Router
- **Styling**: Tailwind CSS v4 with custom design system
- **Charts**: Chart.js 4 with `react-chartjs-2`
- **UI Libraries**: Headless UI, Heroicons, TanStack Table
- **State Management**: React hooks and context
- **Testing**: Jest + Playwright for unit and E2E testing

### Backend (Python Flask)
- **API Framework**: Flask with CORS support
- **Data Processing**: Pandas, NumPy, SciPy for numerical analysis
- **Machine Learning**: PyTorch, scikit-learn, stable-baselines3, HMM
- **Technical Analysis**: TA-Lib for indicators
- **Market Data**: Yahoo Finance (yfinance) integration
- **Trading Engine**: Paper trading execution with pyramiding and position monitoring
- **Telegram Integration**: Remote bot for analysis and trading control
- **Strategy System**: JSON-based dynamic strategy loader (`backend/strategies/`) with swing gates and entry patterns

### Database
- **Primary**: MongoDB for document storage
- **Caching**: Redis for performance optimization

## 📦 Installation

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.9+ and pip
- **MongoDB** database instance
- **Redis** (optional, for caching)
- **Git** for version control

### Backend Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd smart_advice/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   # Set up MongoDB connection and other config in config.py
   # Ensure MongoDB is running on your system
   ```

5. **Run the backend server**:
   ```bash
   python app.py
   ```
   Backend will be available at `http://localhost:5001`

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd ../frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Configure environment variables**:
   ```bash
   # Create .env.local file with:
   NEXT_PUBLIC_API_URL=http://127.0.0.1:5001
   ```

4. **Run the development server**:
   ```bash
   npm run dev
   ```
   Frontend will be available at `http://localhost:3000`

## 🚀 Usage

### Getting Started

1. **Access the application** at `http://localhost:3000`
2. **Check system status** on the main dashboard
3. **View current recommendations** in the recommendations section
4. **Trigger new analysis** using the analysis configuration panel

### API Endpoints

#### Core Endpoints
- `GET /` - Health check
- `GET /recommendations` - Fetch all stock recommendations
- `POST /trigger-analysis` - Start stock analysis
- `GET /analysis-progress` - Check analysis progress
- `GET /symbols` - Get available NSE symbols
- `GET /test_db` - Test database connection

#### Analysis Configuration
```json
{
  "max_stocks": 100,
  "test": false,
  "all": true,
  "offline": false,
  "verbose": true,
  "purge_days": 30,
  "disable_volume_filter": false
}
```

### Scripts and Commands

#### Frontend Scripts
```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run test         # Run Jest unit tests
npm run test:e2e     # Run Playwright E2E tests
npm run test:all     # Run all tests
```

#### Backend Scripts
```bash
python run_analysis.py                              # Run complete stock analysis (JSON strategy-driven)
python main_orchestrator.py                         # Run unified trading cycle (analysis + execution + portfolio backtest)
python telegram_bot.py                              # Start Telegram control bot
python scripts/portfolio_monitor_paper.py           # Monitor open positions and exits
python scripts/run_portfolio_backtest.py            # Run portfolio-level backtest simulation
python scripts/run_portfolio_backtest.py --strategy Hybrid_Trading --walk-forward --mc-iterations 10  # Walk-forward robustness test
python scripts/backtesting.py                       # Run per-strategy backtesting
python tests/test_complete_system.py               # Test system integration
```

### Walk-Forward Backtesting

Validate your strategy across different time periods and stock universes to ensure robustness:

```bash
cd backend
python scripts/run_portfolio_backtest.py --strategy Hybrid_Trading --walk-forward --mc-iterations 10 --period 5y
```

**How it works:**
1. Splits 5-year period into rolling 6-month windows with 3-month steps
2. For each window: runs Monte Carlo sampling (70% of stocks, 10 iterations)
3. Aggregates results: mean CAGR, std dev, min/max, robustness score, positive CAGR %

**Output metrics:**
- **CAGR**: Mean ± std, range (min → max), median
- **Risk & Return**: Avg win rate, Sharpe, max drawdown, profit factor
- **Robustness Score**: 0-100 (higher = more consistent across periods/universes)
- **Positive CAGR %**: % of runs with positive returns

A robust strategy shows: robustness_score > 60 AND positive_cagr_pct > 70%

## 📊 Analysis Strategies

The platform includes 55+ technical indicator modules and JSON-configured trading strategies:

### Technical Indicators
- Moving Average Crossovers (SMA, EMA, DEMA, TEMA)
- Momentum Oscillators (RSI, Stochastic, Williams %R)
- Volume Indicators (OBV, Chaikin Oscillator, MFI)
- Volatility Indicators (Bollinger Bands, ATR, Keltner Channels)

### Chart Patterns
- Candlestick Patterns (Doji, Hammer, Engulfing)
- Support/Resistance Breakouts
- Channel Trading
- Fibonacci Retracements

### Advanced Analysis
- Ichimoku Cloud Analysis
- MACD Signal and Zero Line Crossovers
- Parabolic SAR Reversals
- Elder Ray Index
- Multi-Timeframe Confluence Engine
- Market Regime Detection
- Options Chain OI Analysis (PCR, Unwinding)
- Smart Money Tracking (FII/DII Flows)
- Screener.in Fundamental Screening

### Portfolio Backtesting & Validation
- **Portfolio-Level Simulation**: Multi-stock backtest with shared capital pool (₹10L default)
- **Parallel Signal Generation**: 8-worker multiprocessing for pre-computing daily signals across all stocks
- **Single Simulation Engine**: One pass with shared capital gives correct CAGR (not averaged mini-portfolios)
- **Walk-Forward + Monte Carlo**: Rolling 6-month windows with random stock universe sampling for robustness validation
- **Identical Strategy Logic**: Entry gates, exits, trailing stops, pyramiding — same across individual backtest, portfolio backtest, and live trading

## 🔧 Configuration

### Environment Variables

**Frontend (.env.local)**:
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:5001
```

**Backend (config.py)**:
```python
# Database Configuration
MONGODB_HOST = "127.0.0.1"
MONGODB_PORT = 27017
MONGODB_DATABASE = "super_advice"

# Analysis Parameters
HISTORICAL_DATA_PERIOD = "5y"
ENABLE_VOLUME_FILTER = True
VOLUME_SPIKE_THRESHOLD = 1.5

# API Configuration
FLASK_PORT = 5001

# Trading Options
TRADING_OPTIONS = {
    "is_paper_trading": True,
    "initial_capital": 100000.0,
    "brokerage_charges": 0.0020,
    "auto_execute": True,
    "circuit_breaker": False,
}

# Risk Management
RISK_MANAGEMENT = {
    "position_sizing": {"risk_per_trade": 0.01, "max_position_pct": 0.10},
    "portfolio_constraints": {"max_concurrent_positions": 10, "daily_loss_limit": 0.03},
    "pyramiding": {"enabled": True, "max_adds": 2, "steps": [...]},
}
```

**Strategy Configs (backend/strategies/*.json)**:
Each strategy is a self-contained JSON file with analysis weights, stock filters, swing gates, entry patterns, and exit rules.

## 🧪 Testing

### Frontend Testing
```bash
# Unit tests with Jest
npm run test

# E2E tests with Playwright
npm run test:e2e

# Run with UI
npm run test:e2e:ui

# Test coverage
npm run test:coverage
```

### Backend Testing
```bash
cd backend

# Run complete system integration test
python tests/test_complete_system.py

# Run swing trading gate tests
python tests/test_swing_trading_gates.py

# Run activated strategy tests
python tests/test_activated_strategies.py

# Run new strategy tests
python tests/test_new_strategies.py

# Run all tests (if using pytest)
python -m pytest tests/
```

## 📈 Performance

### Backend Optimizations
- **Caching**: Redis integration for frequently accessed data
- **Parallel Processing**: Multi-threaded analysis execution
- **Memory Management**: Optimized data structures and garbage collection
- **Database Indexing**: MongoDB indexes for faster queries

### Frontend Optimizations
- **Code Splitting**: Automatic code splitting with Next.js
- **Image Optimization**: Next.js Image component with WebP support
- **Bundle Analysis**: Webpack bundle analyzer integration
- **Lazy Loading**: Component-level lazy loading

## 🛡️ Security

### Backend Security
- **CORS Configuration**: Properly configured cross-origin requests
- **Input Validation**: Pydantic models for request validation
- **Error Handling**: Comprehensive error handling and logging
- **Rate Limiting**: API rate limiting (configurable)

### Frontend Security
- **Environment Variables**: Secure handling of sensitive data
- **CSP Headers**: Content Security Policy implementation
- **XSS Protection**: Built-in Next.js XSS protection
- **HTTPS**: SSL/TLS in production environments

## 🚧 Roadmap

### Phase 1: Current Features ✅
- [x] Core technical analysis
- [x] Fundamental analysis integration
- [x] Sentiment analysis
- [x] Web-based dashboard
- [x] Real-time progress tracking

### Phase 2: Advanced Features 🚧
- [x] **F&O Analysis**: Options chain analysis and volatility insights (backend ready)
- [x] **Portfolio Management**: Paper trading portfolio tracking with position monitoring
- [x] **Alert System**: Telegram bot for remote notifications and control
- [ ] **Mobile App**: React Native mobile application
- [x] **Multi-Strategy System**: JSON-based dynamic strategy loading and execution

### Phase 3: Enterprise Features 📋
- [ ] **Multi-user Support**: User authentication and management
- [ ] **Custom Strategies**: User-defined trading strategies
- [ ] **API Integration**: Third-party broker API integration
- [ ] **Advanced Reporting**: PDF reports and analytics

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow
1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper testing
4. **Run tests**: `npm test` (frontend) and `python -m pytest` (backend)
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Code Standards
- **Frontend**: ESLint + Prettier configuration
- **Backend**: PEP 8 Python style guide
- **Testing**: Minimum 80% test coverage
- **Documentation**: JSDoc for TypeScript, docstrings for Python

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

### Documentation
- **API Documentation**: Available at `/docs` endpoint
- **Component Storybook**: `npm run storybook`
- **Technical Guides**: See `/docs` directory

### Community
- **Issues**: GitHub Issues for bug reports
- **Discussions**: GitHub Discussions for questions
- **Updates**: Follow releases for latest features

### Professional Support
For enterprise support and custom implementations, please contact the development team.

---

## 🙏 Acknowledgments

- **TA-Lib**: Technical Analysis Library
- **Yahoo Finance**: Market data provider
- **Chart.js**: Charting library
- **Next.js**: React framework
- **Tailwind CSS**: Utility-first CSS framework
- **Flask**: Python web framework

---

**Made with ❤️ for the trading community**

*Disclaimer: This software is for educational and research purposes. Always consult with financial advisors before making investment decisions.*
