# Momentum Strategy Backtest
# ==========================
#
# This notebook implements a 12-month momentum strategy.
#
# ⚠️ WARNING: This backtest contains INTENTIONAL ERRORS for educational purposes.
# See /notebooks/corrected/01_momentum_lookahead_fixed.py for the corrected version.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION
# =============================================================================

LOOKBACK_MONTHS = 12
HOLDING_PERIOD_MONTHS = 1
TOP_DECILE = 0.10
INITIAL_CAPITAL = 1_000_000
START_DATE = "2010-01-01"
END_DATE = "2023-12-31"

# =============================================================================
# DATA LOADING
# =============================================================================


def load_price_data():
    """
    Load historical price data.

    In production, this would load from a database.
    For this example, we'll generate synthetic data.
    """
    np.random.seed(42)

    # Generate 500 stocks over ~14 years
    dates = pd.date_range(START_DATE, END_DATE, freq="B")
    n_stocks = 500
    tickers = [f"STOCK_{i:03d}" for i in range(n_stocks)]

    # Generate correlated returns
    base_return = np.random.normal(0.0003, 0.015, len(dates))

    prices = {}
    for ticker in tickers:
        idio_returns = np.random.normal(0, 0.02, len(dates))
        stock_returns = 0.5 * base_return + 0.5 * idio_returns
        prices[ticker] = 100 * np.cumprod(1 + stock_returns)

    df = pd.DataFrame(prices, index=dates)
    return df


def load_fundamental_data(price_data):
    """
    Load fundamental data (market cap, sector, etc.)
    """
    # Generate market caps (will be used for universe filtering)
    np.random.seed(123)
    tickers = price_data.columns

    # BUG #1: Using FINAL prices to determine market cap for universe selection
    # This is look-ahead bias - we're using end-of-period data to make
    # beginning-of-period decisions
    final_prices = price_data.iloc[-1]
    shares_outstanding = np.random.uniform(10e6, 500e6, len(tickers))
    market_caps = final_prices * shares_outstanding

    return pd.DataFrame(
        {"ticker": tickers, "market_cap": market_caps, "shares": shares_outstanding}
    ).set_index("ticker")


# =============================================================================
# SIGNAL GENERATION
# =============================================================================


def calculate_momentum_signal(prices, date):
    """
    Calculate 12-month momentum signal for a given date.

    Momentum = (P_t / P_{t-12m}) - 1
    """
    # Get lookback start date
    lookback_start = date - pd.DateOffset(months=LOOKBACK_MONTHS)

    # BUG #2: Using .loc with exact date matching fails if date isn't in index
    # This can accidentally grab future data or fail silently
    try:
        current_prices = prices.loc[date]
    except KeyError:
        # BUG #3: When date not found, we use forward-fill which can peek ahead
        current_prices = prices.loc[:date].iloc[-1]

    try:
        past_prices = prices.loc[lookback_start]
    except KeyError:
        past_prices = prices.loc[:lookback_start].iloc[-1]

    # Calculate momentum
    momentum = (current_prices / past_prices) - 1

    return momentum


def calculate_all_signals(prices):
    """
    Calculate momentum signals for all rebalance dates.
    """
    # Generate monthly rebalance dates
    rebalance_dates = pd.date_range(START_DATE, END_DATE, freq="MS")[
        LOOKBACK_MONTHS:
    ]  # Skip first year for lookback

    signals = {}

    for date in rebalance_dates:
        # BUG #4: Not accounting for reporting lag
        # In reality, you can't use month-end prices on the first business day
        # of the next month - there's settlement, data availability, etc.
        momentum = calculate_momentum_signal(prices, date)
        signals[date] = momentum

    return pd.DataFrame(signals).T


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================


def select_universe(prices, fundamentals, date):
    """
    Select tradeable universe based on market cap and liquidity.
    """
    # BUG #5: Using static market cap calculated from end-of-period prices
    # Should use point-in-time market cap
    large_caps = fundamentals[
        fundamentals["market_cap"] > fundamentals["market_cap"].quantile(0.5)
    ]

    return large_caps.index.tolist()


def construct_portfolio(signals, date, universe):
    """
    Construct long-only momentum portfolio.
    Select top decile of momentum stocks.
    """
    # Get signals for this date
    date_signals = signals.loc[date]

    # Filter to universe
    date_signals = date_signals[date_signals.index.isin(universe)]

    # BUG #6: Including stocks that will be delisted before end of holding period
    # This is survivorship bias - we're only seeing stocks that "survived"

    # Select top decile
    n_stocks = int(len(date_signals) * TOP_DECILE)
    top_momentum = date_signals.nlargest(n_stocks)

    # Equal weight
    weights = pd.Series(1.0 / n_stocks, index=top_momentum.index)

    return weights


# =============================================================================
# BACKTEST ENGINE
# =============================================================================


def run_backtest(prices, signals, fundamentals):
    """
    Run the momentum strategy backtest.
    """
    rebalance_dates = signals.index

    portfolio_values = [INITIAL_CAPITAL]
    portfolio_dates = [rebalance_dates[0] - pd.DateOffset(days=1)]

    holdings = None

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # Select universe
        universe = select_universe(prices, fundamentals, date)

        # Construct portfolio
        weights = construct_portfolio(signals, date, universe)

        # BUG #7: Not accounting for transaction costs
        # Real trading has costs that eat into returns

        # Calculate returns
        # Get prices at rebalance and next rebalance
        try:
            entry_prices = prices.loc[date, weights.index]
        except:
            entry_prices = prices.loc[:date, weights.index].iloc[-1]

        try:
            exit_prices = prices.loc[next_date, weights.index]
        except:
            exit_prices = prices.loc[:next_date, weights.index].iloc[-1]

        # BUG #8: Using close prices for both entry and exit
        # Can't actually trade at close - need to use next open or VWAP

        # Calculate portfolio return
        stock_returns = (exit_prices / entry_prices) - 1
        portfolio_return = (weights * stock_returns).sum()

        # Update portfolio value
        new_value = portfolio_values[-1] * (1 + portfolio_return)
        portfolio_values.append(new_value)
        portfolio_dates.append(next_date)

        holdings = weights

    return pd.Series(portfolio_values, index=portfolio_dates)


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================


def calculate_metrics(portfolio_values):
    """
    Calculate performance metrics.
    """
    returns = portfolio_values.pct_change().dropna()

    # Annualized return
    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    years = (portfolio_values.index[-1] - portfolio_values.index[0]).days / 365.25
    ann_return = (1 + total_return) ** (1 / years) - 1

    # Annualized volatility
    ann_vol = returns.std() * np.sqrt(12)  # Monthly returns

    # Sharpe ratio
    # BUG #9: Using 0 risk-free rate and not accounting for the period
    sharpe = ann_return / ann_vol

    # Max drawdown
    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    max_dd = drawdown.min()

    # BUG #10: Reporting in-sample Sharpe without out-of-sample validation
    # This Sharpe is likely overstated due to all the bugs above

    return {
        "Total Return": f"{total_return:.2%}",
        "Annualized Return": f"{ann_return:.2%}",
        "Annualized Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("Loading data...")
    prices = load_price_data()
    fundamentals = load_fundamental_data(prices)

    print("Calculating signals...")
    signals = calculate_all_signals(prices)

    print("Running backtest...")
    portfolio_values = run_backtest(prices, signals, fundamentals)

    print("\n" + "=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)

    metrics = calculate_metrics(portfolio_values)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\n⚠️  WARNING: This backtest contains multiple bugs!")
    print("    See the corrected version for proper implementation.")

    return portfolio_values, metrics


if __name__ == "__main__":
    portfolio_values, metrics = main()
