# AI Evaluation Guide: Grading Quantitative Code

## Overview
This document outlines the **RLHF (Reinforcement Learning from Human Feedback)** rubric used to evaluate AI-generated financial code in this repository. 

As Large Language Models (LLMs) become proficient at syntax, their failure mode shifts to **semantic hallucinations**—subtle logic errors that compile correctly but lead to financial ruin. This guide serves as the standard for detecting those hallucinations.

---

## 🛑 The "Hallucination" Checklist
When reviewing AI-generated backtests, I audit for these four specific "silent killers."

### 1. Look-Ahead Bias (The Time Travel Fallacy)
* **The Error:** The AI inadvertently uses future data to make past decisions.
* **Code Red Flags:**
    * Using `df.shift(-1)` or `next_day_return` inside the signal generation block.
    * Calculating aggregations (e.g., `mean()`, `std()`) on the entire dataset *before* splitting dates.
    * Accessing "Close" prices for trade execution at the *open* of the same bar.
* **The Fix:** Strict point-in-time partitioning. Signals must be calculated using only data available at $t-1$.

### 2. Survivorship Bias (The "Winner's" Fallacy)
* **The Error:** The AI selects a universe of stocks that exist *today* and backtests them starting in 2010.
* **Code Red Flags:**
    * `universe = ['AAPL', 'NVDA', 'AMZN']` (Hardcoded modern winners).
    * Downloading "current S&P 500 members" for historical tests.
    * Lack of handling for `NaN` or delisted tickers in the price matrix.
* **The Fix:** Universe selection must be dynamic per timestamp, or use an "Index Historical Constituents" dataset.

### 3. Overfitting (The P-Hacking Fallacy)
* **The Error:** The AI iterates through parameters until it finds a curve that fits the noise.
* **Code Red Flags:**
    * Complex logic trees: `if RSI > 30 and MA < 50 and VIX < 20...`
    * Lack of a "Deflated Sharpe Ratio" (DSR) adjustment for multiple trials.
    * "Magic Numbers" optimized to the 4th decimal place.
* **The Fix:** Penalize complexity. Require Out-of-Sample (OOS) testing or Walk-Forward Analysis.

### 4. Data Leakage (The ML Fallacy)
* **The Error:** Information from the test set "leaks" into the training set during normalization.
* **Code Red Flags:**
    * `scaler.fit_transform(X)` called on the whole dataframe *before* `train_test_split`.
    * Random shuffling of time-series data (destroys temporal correlation).
* **The Fix:** Fit scalers *only* on the training fold. Use `PurgedKFold` CV to enforce embargo periods between train/test sets.

---

## 📝 Grading Rubric (RLHF Score)

I assign a 1-5 score to AI outputs based on financial validity, not just Python syntax.

| Score | Rating | Criteria |
| :--- | :--- | :--- |
| **1** | **Dangerous** | Contains Look-Ahead Bias. Will lose money in production despite a beautiful backtest curve. |
| **2** | **Flawed** | Mathematically unsound (e.g., averaging arithmetic returns incorrectly). Survivorship bias present. |
| **3** | **Functional** | Python runs without errors, but lacks transaction costs or slippage logic. Academic only. |
| **4** | **Robust** | Includes costs, lag, and proper date alignment. Code is modular and clean. |
| **5** | **Production** | Institutional grade. Handles edge cases (zeros, NaNs), implements DSR/Walk-Forward, and separates Signal from Execution. |

---

## 🧠 Chain-of-Thought (CoT) Verification
Beyond the code, I evaluate the AI's *reasoning path*. A correct answer with wrong reasoning is a "False Positive."

**Good CoT Pattern:**
> "I need to calculate momentum. I must ensure I lag the signal by one day to simulate trade execution at the next open, preventing look-ahead bias..."

**Bad CoT Pattern:**
> "I will calculate returns using today's close and trade on them immediately to maximize alpha..." (Violates causality).

---

*This guide guides the "Agent Skills" found in the `/skills` directory.*