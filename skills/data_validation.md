# Skill: Data Validation

Guidelines for ensuring high-quality stock data for analysis.

## Overview
Stock data from free APIs like `yfinance` can be noisy. This skill ensures we filter out the garbage before it hits our models.

## Checkpoints
1.  **Volume Check**: 
    *   Verify `Volume > 100,000` (Average 50-day).
    *   Ignore stocks with 0 volume bars in the last 5 sessions.
2.  **Price Sanity**:
    *   Check for "Penny Stocks" (Price < 20).
    *   Check for "Dead Stocks" (No price change for > 3 days).
3.  **Dividend Logic**:
    *   Ensure dividend yield is <= 20% (anything higher is often a data error or special dividend).
4.  **Market Cap**:
    *   Prioritize Mid-cap and Large-cap for technical signals (> 500 Cr).

## Resolution Skills
- If data is missing for 1-2 days: Use linear interpolation for signals.
- If data is missing for > 10% of the period: **Discard the stock from analysis.**
