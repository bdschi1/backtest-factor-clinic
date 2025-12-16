# Backtest Factor Clinic 🏥

**A professional-grade diagnostic toolkit for identifying, quantifying, and fixing common quantitative finance backtest errors.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![AI Agent: Ready](https://img.shields.io/badge/AI_Agent-Skill_Enabled-purple)](./skills)

---

## 🎯 Mission

Most backtests are broken. They suffer from subtle bugs like **look-ahead bias**, **survivorship bias**, **overfitting**, or **data leakage**.

This repository is a **diagnostic suite** designed to:
1.  **Demonstrate** these errors in realistic scenarios.
2.  **Quantify** their impact (e.g., "Look-ahead bias inflated Sharpe from 0.5 to 3.0").
3.  **Correct** them using industry-standard engineering patterns (Point-in-Time data, Purging/Embargoing, Deflated Sharpe Ratio).

---

## 🚀 Quick Start (One-Click Demo)

The fastest way to see the diagnostics in action is the automated demo script. This generates synthetic data, runs all CLI checks, and launches the visual dashboard.

```bash
# 1. Clone the repo
git clone [https://github.com/bdschi1/backtest-factor-clinic.git](https://github.com/bdschi1/backtest-factor-clinic.git)
cd backtest-factor-clinic

# 2. Run the full suite
chmod +x run_demo.sh
./run_demo.sh

## 📦 Manual Installation

If you prefer to run components individually or are on Windows (where .sh scripts don't work), use these commands:

```bash
# 1. Install dependencies
pip install pandas numpy scipy scikit-learn streamlit

# 2. Run the Dashboard
python3 -m streamlit run dashboard.py

# 3. Run a specific diagnostic check (CLI)
python3 notebooks/corrected/run_diagnosis.py --check momentum