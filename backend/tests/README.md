# Test Files

This directory contains all test files for the Smart Advice trading system backend.

## Test Files Overview

- `test_analysis_simple.py` - Simple analysis tests
- `test_analyzer_components.py` - Tests for analyzer components
- `test_analyzer_init.py` - Tests for analyzer initialization
- `test_backtesting_integration.py` - Backtesting integration tests
- `test_basic.py` - Basic functionality tests
- `test_complete_system.py` - Complete system integration tests
- `test_data_fetch.py` - Data fetching tests
- `test_fixed_analysis.py` - Fixed analysis tests
- `test_full_init_sequence.py` - Full initialization sequence tests
- `test_mongo_simple.py` - Simple MongoDB tests
- `test_new_strategies.py` - New trading strategies tests
- `test_openmp_fix.py` - OpenMP configuration fix tests
- `test_progressive_run.py` - Progressive run tests
- `test_strategy_init.py` - Strategy initialization tests
- `test_activated_strategies.py` - Activated strategies tests

## Running Tests

To run tests, navigate to the parent directory and run:

```bash
# Run a specific test file
python tests/test_basic.py

# Or run all tests (if using pytest)
pytest tests/
```

## Note

These test files are designed to validate various components of the trading system including data fetching, analysis algorithms, backtesting, and database operations.
