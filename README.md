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
- **Multi-Strategy Backtesting**: Comprehensive backtesting with CAGR calculations
- **Risk Management**: Advanced position sizing and risk assessment

### Advanced Analytics
- **Machine Learning Models**: Deep learning models for price prediction
- **Reinforcement Learning**: Trading agents for decision optimization
- **Market Regime Detection**: Automatic detection of market conditions
- **Sector Analysis**: Industry-specific insights and comparisons
- **Market Microstructure**: Order flow and liquidity analysis

### Modern Web Interface
- **Real-time Dashboard**: Interactive charts and visualizations using Chart.js
- **Dark Mode Support**: Toggle between light and dark themes
- **Responsive Design**: Optimized for desktop and mobile devices
- **Progress Tracking**: Real-time analysis progress monitoring

## 🏗️ Architecture

### Frontend (Next.js 15 + TypeScript)
- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS v4 with custom design system
- **Charts**: Chart.js with React integration
- **State Management**: React hooks and context
- **Testing**: Jest + Playwright for unit and E2E testing

### Backend (Python Flask)
- **API Framework**: Flask with CORS support
- **Data Processing**: Pandas, NumPy, SciPy for numerical analysis
- **Machine Learning**: PyTorch, scikit-learn, stable-baselines3
- **Technical Analysis**: TA-Lib for indicators
- **Market Data**: Yahoo Finance (yfinance) integration

### Database
- **Primary**: MongoDB for document storage
- **Caching**: Redis for performance optimization

## 📦 Installation

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.8+ and pip
- **MongoDB** database instance
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
python run_analysis.py          # Run complete stock analysis
python test_complete_system.py  # Test system integration
python scripts/backtesting.py   # Run backtesting analysis
```

## 📊 Analysis Strategies

The platform includes 70+ built-in trading strategies:

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

## 🔧 Configuration

### Environment Variables

**Frontend (.env.local)**:
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:5001
```

**Backend (config.py)**:
```python
# Database Configuration
MONGODB_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "smart_advice"

# Analysis Parameters
DATA_PURGE_DAYS = 30
MAX_ANALYSIS_STOCKS = 100
ENABLE_VOLUME_FILTER = True

# API Configuration
FLASK_PORT = 5001
CORS_ORIGINS = ["http://localhost:3000"]
```

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
# Run basic system tests
python test_basic.py

# Test data fetching
python test_data_fetch.py

# Test complete system integration
python test_complete_system.py

# Test specific strategies
python test_new_strategies.py
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
- [ ] **F&O Analysis**: Options chain analysis and volatility insights
- [ ] **Portfolio Management**: Multi-asset portfolio tracking
- [ ] **Alert System**: Email/SMS notifications for triggers
- [ ] **Mobile App**: React Native mobile application

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
