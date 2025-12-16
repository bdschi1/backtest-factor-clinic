# Value Strategy Backtest with Survivorship Bias
# ==============================================
#
# This notebook implements a value (low P/E) strategy.
#
# ⚠️ WARNING: This backtest contains SURVIVORSHIP BIAS and other errors.
# See /notebooks/corrected/02_value_survivorship_fixed.py for the corrected version.

import pandas as pd
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

INITIAL_CAPITAL = 1_000_000
START_DATE = "2005-01-01"
END_DATE = "2023-12-31"
N_STOCKS = 30  # Number of stocks in portfolio

# =============================================================================
# DATA LOADING - WITH SURVIVORSHIP BIAS
# =============================================================================


def load_current_sp500():
    """
    Load current S&P 500 constituents.

    BUG #1: SURVIVORSHIP BIAS
    Using TODAY'S S&P 500 constituents for a backtest starting in 2005.
    Companies that went bankrupt, were acquired, or dropped from the index
    are excluded, biasing results upward.

    Examples of what we're missing:
    - Lehman Brothers (bankrupt 2008)
    - Bear Stearns (acquired 2008)
    - Enron (bankrupt 2001)
    - WorldCom (bankrupt 2002)
    - Washington Mutual (failed 2008)
    - General Motors (bankrupt 2009)
    - Kodak (bankrupt 2012)
    - RadioShack (bankrupt 2015)
    """
    # Simulating loading "current" S&P 500
    # In reality, about 40% of S&P 500 constituents change over a 20-year period
    np.random.seed(42)

    # Generate 500 "current" stocks
    current_tickers = [f"SURV_{i:03d}" for i in range(500)]

    return current_tickers


def generate_price_data(tickers, start_date, end_date):
    """
    Generate synthetic price data.

    BUG #2: SELECTION BIAS IN DATA GENERATION
    We're generating data only for stocks that "exist" today.
    Failed companies would have had declining prices before delisting.
    """
    np.random.seed(42)
    dates = pd.date_range(start_date, end_date, freq="B")

    prices = {}
    for ticker in tickers:
        # Generate random walk with slight upward drift
        # BUG #3: This drift is too optimistic - doesn't include failed companies
        returns = np.random.normal(0.0004, 0.02, len(dates))  # ~10% annual
        prices[ticker] = 100 * np.cumprod(1 + returns)

    return pd.DataFrame(prices, index=dates)


def generate_fundamental_data(tickers, dates):
    """
    Generate P/E ratio data for stocks.

    BUG #4: HINDSIGHT IN FUNDAMENTAL DATA
    Generating "clean" fundamental data without:
    - Restatements
    - Reporting lags (10-K filed 60-90 days after fiscal year end)
    - Data revisions
    """
    np.random.seed(123)

    fundamentals = {}
    for date in dates[::21]:  # Monthly data
        pe_ratios = {}
        for ticker in tickers:
            # Random P/E between 5 and 50
            pe = np.random.uniform(5, 50)
            pe_ratios[ticker] = pe
        fundamentals[date] = pe_ratios

    return pd.DataFrame(fundamentals).T


def generate_earnings_data(tickers, dates, prices):
    """
    Generate earnings data.

    BUG #5: NO REPORTING LAG
    In reality, Q4 earnings aren't available until February-March.
    We're assuming instant availability.
    """
    np.random.seed(456)

    earnings = {}
    for date in dates[::63]:  # Quarterly
        eps = {}
        for ticker in tickers:
            # EPS based on current price (THIS IS CIRCULAR!)
            # BUG #6: Using price to generate earnings creates look-ahead
            current_price = prices.loc[date, ticker] if date in prices.index else 100
            pe_ratio = np.random.uniform(10, 30)
            eps[ticker] = current_price / pe_ratio
        earnings[date] = eps

    return pd.DataFrame(earnings).T


# =============================================================================
# SIGNAL GENERATION
# =============================================================================


def calculate_pe_ratios(prices, earnings, date):
    """
    Calculate P/E ratios for stock selection.
    """
    # Get most recent earnings
    # BUG #7: Not accounting for fiscal year end differences
    # Some companies are calendar year, some are June FYE, etc.
    available_earnings = earnings.loc[:date]
    if len(available_earnings) == 0:
        return None

    latest_earnings = available_earnings.iloc[-1]
    current_prices = prices.loc[date]

    # Calculate P/E
    pe_ratios = current_prices / (latest_earnings * 4)  # Annualize quarterly

    # BUG #8: Not handling negative earnings
    # Negative P/E is meaningless, but we're including them

    return pe_ratios


def select_value_stocks(pe_ratios, n_stocks):
    """
    Select lowest P/E stocks.

    BUG #9: VALUE TRAP - Low P/E doesn't mean undervalued
    Many low P/E stocks are cheap for a reason (declining business, fraud, etc.)
    No quality screen applied.
    """
    # Remove infinite and NaN
    pe_ratios = pe_ratios.replace([np.inf, -np.inf], np.nan).dropna()

    # BUG #10: Including negative P/E stocks
    # These could be the worst performers

    # Select lowest P/E
    lowest_pe = pe_ratios.nsmallest(n_stocks)

    return lowest_pe.index.tolist()


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================


def construct_portfolio(selected_stocks):
    """
    Equal weight portfolio construction.

    BUG #11: No consideration of:
    - Liquidity constraints
    - Sector concentration
    - Position size limits
    - Trading costs
    """
    n = len(selected_stocks)
    weights = pd.Series(1.0 / n, index=selected_stocks)
    return weights


# =============================================================================
# BACKTEST ENGINE
# =============================================================================


def run_backtest(prices, earnings):
    """
    Run value strategy backtest.
    """
    # Monthly rebalancing
    rebalance_dates = pd.date_range(START_DATE, END_DATE, freq="MS")[12:]

    portfolio_values = [INITIAL_CAPITAL]
    portfolio_dates = [pd.Timestamp(START_DATE)]

    trade_count = 0

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # Ensure dates are in price data
        if date not in prices.index:
            date = prices.index[prices.index.get_indexer([date], method="ffill")[0]]
        if next_date not in prices.index:
            next_date = prices.index[
                prices.index.get_indexer([next_date], method="ffill")[0]
            ]

        # Calculate P/E ratios
        pe_ratios = calculate_pe_ratios(prices, earnings, date)
        if pe_ratios is None:
            continue

        # Select stocks
        selected = select_value_stocks(pe_ratios, N_STOCKS)

        # Construct portfolio
        weights = construct_portfolio(selected)

        # Calculate returns
        entry_prices = prices.loc[date, selected]
        exit_prices = prices.loc[next_date, selected]

        # BUG #12: Ignoring stocks that get delisted during holding period
        # If a stock goes to zero, we lose that portion of the portfolio
        # But with survivorship bias, we don't see these cases

        stock_returns = (exit_prices / entry_prices) - 1

        # BUG #13: No transaction costs
        portfolio_return = (weights * stock_returns).sum()

        # BUG #14: No slippage modeling
        # Real trades don't execute at the price you see

        new_value = portfolio_values[-1] * (1 + portfolio_return)
        portfolio_values.append(new_value)
        portfolio_dates.append(next_date)

        trade_count += len(selected)

    return pd.Series(portfolio_values, index=portfolio_dates), trade_count


# =============================================================================
# BENCHMARK COMPARISON
# =============================================================================


def calculate_benchmark_returns(prices):
    """
    Calculate equal-weight benchmark returns.

    BUG #15: BENCHMARK ALSO HAS SURVIVORSHIP BIAS
    We're comparing our biased strategy to a biased benchmark.
    Both are inflated, so relative performance looks reasonable.
    """
    # Equal weight all stocks
    returns = prices.pct_change()
    benchmark_returns = returns.mean(axis=1)

    benchmark_value = INITIAL_CAPITAL * (1 + benchmark_returns).cumprod()
    return benchmark_value


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================


def calculate_metrics(portfolio_values, benchmark_values):
    """
    Calculate performance metrics.
    """
    # Align dates
    common_dates = portfolio_values.index.intersection(benchmark_values.index)
    portfolio_values = portfolio_values.loc[common_dates]
    benchmark_values = benchmark_values.loc[common_dates]

    # Returns
    port_returns = portfolio_values.pct_change().dropna()
    bench_returns = benchmark_values.pct_change().dropna()

    # Total return
    port_total = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    bench_total = benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1

    years = len(port_returns) / 12

    # Annualized metrics
    port_ann = (1 + port_total) ** (1 / years) - 1
    bench_ann = (1 + bench_total) ** (1 / years) - 1

    port_vol = port_returns.std() * np.sqrt(12)
    bench_vol = bench_returns.std() * np.sqrt(12)

    # Sharpe (assuming 2% risk-free rate for this period)
    rf = 0.02
    port_sharpe = (port_ann - rf) / port_vol
    bench_sharpe = (bench_ann - rf) / bench_vol

    # Information ratio
    active_returns = port_returns - bench_returns
    tracking_error = active_returns.std() * np.sqrt(12)
    info_ratio = (port_ann - bench_ann) / tracking_error

    # Max drawdown
    port_cummax = portfolio_values.cummax()
    port_dd = ((portfolio_values - port_cummax) / port_cummax).min()

    return {
        "Strategy Total Return": f"{port_total:.2%}",
        "Strategy Ann. Return": f"{port_ann:.2%}",
        "Strategy Ann. Vol": f"{port_vol:.2%}",
        "Strategy Sharpe": f"{port_sharpe:.2f}",
        "Strategy Max DD": f"{port_dd:.2%}",
        "Benchmark Total Return": f"{bench_total:.2%}",
        "Benchmark Ann. Return": f"{bench_ann:.2%}",
        "Benchmark Sharpe": f"{bench_sharpe:.2f}",
        "Information Ratio": f"{info_ratio:.2f}",
        "Tracking Error": f"{tracking_error:.2%}",
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("VALUE STRATEGY BACKTEST")
    print("=" * 60)

    # Load current constituents (SURVIVORSHIP BIAS!)
    print("\nLoading current S&P 500 constituents...")
    print("⚠️  WARNING: This introduces survivorship bias!")
    tickers = load_current_sp500()
    print(f"  Loaded {len(tickers)} stocks")

    # Generate data
    print("\nGenerating price data...")
    dates = pd.date_range(START_DATE, END_DATE, freq="B")
    prices = generate_price_data(tickers, START_DATE, END_DATE)

    print("Generating earnings data...")
    earnings = generate_earnings_data(tickers, dates, prices)

    # Run backtest
    print("\nRunning backtest...")
    portfolio_values, trade_count = run_backtest(prices, earnings)
    print(f"  Total trades: {trade_count}")

    # Calculate benchmark
    print("Calculating benchmark...")
    benchmark_values = calculate_benchmark_returns(prices)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("RESULTS (INFLATED DUE TO SURVIVORSHIP BIAS)")
    print("=" * 60)

    metrics = calculate_metrics(portfolio_values, benchmark_values)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("⚠️  BUGS IN THIS BACKTEST:")
    print("=" * 60)
    print(
        """
    1. Survivorship bias - using current constituents for historical backtest
    2. No delisted stocks - missing bankruptcies, acquisitions, delistings
    3. No reporting lag - using earnings before they were available
    4. No restatement handling - using "final" numbers
    5. No negative earnings handling - including value traps
    6. No transaction costs
    7. No liquidity constraints
    8. Benchmark also biased - relative comparison meaningless
    
    ESTIMATED IMPACT:
    - Survivorship bias alone can add 1-2% annually to returns
    - Combined biases could inflate Sharpe by 0.3-0.5
    """
    )

    return portfolio_values, benchmark_values, metrics


if __name__ == "__main__":
    portfolio, benchmark, metrics = main()
