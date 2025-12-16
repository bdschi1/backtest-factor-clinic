# Backtest Diagnostician

## Description
A specialized skill for diagnosing quantitative trading strategies. It detects look-ahead bias, survivorship bias, overfitting, and data leakage using the "Backtest Factor Clinic" toolkit.

## Capabilities
1. **Check Momentum (Look-Ahead Bias):**
   - Verifies if the strategy correctly uses Point-in-Time (PIT) data.
   - Run: `python skills/backtest_diagnostician/wrapper.py --check momentum --data among_synth.csv`

2. **Check Value (Survivorship Bias):**
   - Simulates performance with and without delisted companies.
   - Run: `python skills/backtest_diagnostician/wrapper.py --check value --data among_synth.csv`

3. **Check Overfitting (Deflated Sharpe):**
   - Calculates the probability of a False Positive Sharpe Ratio.
   - Run: `python skills/backtest_diagnostician/wrapper.py --check multifactor --data among_synth.csv`

4. **Check ML Leakage (Purged CV):**
   - Tests for training data leakage using Purged K-Fold Cross-Validation.
   - Run: `python skills/backtest_diagnostician/wrapper.py --check ml --data among_synth.csv`

## Rules
- If the user does not specify a data file, default to `among_synth.csv`.
- If `among_synth.csv` is missing, generate it first using `python generate_synth_csv.py`.