# Backtest Factor Clinic 🏥

**A professional-grade diagnostic toolkit for identifying, quantifying, and fixing common quantitative finance backtest errors.**

[![Quant CI/CD Pipeline](https://github.com/brads-repo/backtest-factor-clinic/actions/workflows/test.yml/badge.svg)](https://github.com/brads-repo/backtest-factor-clinic/actions)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![AI Agent: Ready](https://img.shields.io/badge/AI_Agent-Skill_Enabled-purple)](https://openai.com/)

---

## 🎯 Mission

Most backtests are broken. They suffer from subtle bugs like **look-ahead bias**, **survivorship bias**, **overfitting**, or **data leakage**.

This repository is a **diagnostic suite** designed to:
1.  **Demonstrate** these errors in realistic scenarios.
2.  **Quantify** their impact (e.g., "Look-ahead bias inflated Sharpe by 0.5").
3.  **Correct** them using industry-standard engineering patterns (Point-in-Time data, Purging/Embargoing).

## 🚀 Three Ways to Run

### 1. The Web Dashboard (Recommended)
Visual, interactive diagnosis running locally in your browser.
```bash
pip install streamlit
streamlit run dashboard.py