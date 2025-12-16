# "Optimized" Multi-Factor Strategy
# ==================================
#
# This notebook demonstrates a "highly optimized" multi-factor strategy.
#
# ⚠️ WARNING: This backtest is severely OVERFITTED and will not work out-of-sample.
# See /notebooks/corrected/03_multifactor_overfitting_fixed.py for proper methodology.

import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION - SUSPICIOUSLY SPECIFIC PARAMETERS
# =============================================================================

# BUG #1: These parameters were clearly optimized on the test data
# Note the bizarre specificity: why 11 days? why 47? why 0.73?
MOMENTUM_LOOKBACK = 247  # Not 252 (1 year) but 247 days... suspicious
MOMENTUM_SKIP = 11  # Skip last 11 days... oddly specific
REVERSION_LOOKBACK = 47  # 47 days... why not 50 or 60?
VALUE_WEIGHT = 0.314  # Three decimal places of "optimization"
MOMENTUM_WEIGHT = 0.427  # Extremely precise
REVERSION_WEIGHT = 0.259  # Adds to exactly 1.000

VOLATILITY_LOOKBACK = 37  # Another arbitrary number
VOLATILITY_CAP = 0.73  # Cap at 73rd percentile... why?
SECTOR_MAX_WEIGHT = 0.187  # 18.7%... curiously specific

REBALANCE_DAY = 3  # 3rd trading day of month (optimized?)
TOP_PERCENTILE = 0.127  # Top 12.7%... clearly fitted

# =============================================================================
# DATA GENERATION
# =============================================================================


def generate_data(n_stocks=300, n_years=15):
    """Generate synthetic price and factor data."""
    np.random.seed(42)

    start = "2008-01-01"
    end = "2022-12-31"
    dates = pd.date_range(start, end, freq="B")

    tickers = [f"STK_{i:03d}" for i in range(n_stocks)]

    # Generate prices
    prices = {}
    for ticker in tickers:
        returns = np.random.normal(0.0003, 0.02, len(dates))
        prices[ticker] = 100 * np.cumprod(1 + returns)

    prices_df = pd.DataFrame(prices, index=dates)

    # Generate sectors
    sectors = {
        t: np.random.choice(["Tech", "Finance", "Healthcare", "Consumer", "Industrial"])
        for t in tickers
    }

    return prices_df, sectors


# =============================================================================
# FACTOR CALCULATIONS - WITH SNOOPING
# =============================================================================


def calculate_momentum(prices, date):
    """
    Calculate momentum factor.

    BUG #2: Lookback period was optimized on in-sample data
    The 247-day lookback with 11-day skip was found by testing
    hundreds of combinations.
    """
    end_idx = prices.index.get_indexer([date], method="ffill")[0]
    start_idx = end_idx - MOMENTUM_LOOKBACK
    skip_start = end_idx - MOMENTUM_SKIP

    if start_idx < 0:
        return None

    # Price at start of lookback
    start_prices = prices.iloc[start_idx]
    # Price at skip point (not current)
    end_prices = prices.iloc[skip_start]

    momentum = (end_prices / start_prices) - 1
    return momentum


def calculate_reversion(prices, date):
    """
    Calculate mean reversion factor (short-term).

    BUG #3: The 47-day lookback is suspiciously specific
    Classic sign of parameter mining.
    """
    end_idx = prices.index.get_indexer([date], method="ffill")[0]
    start_idx = end_idx - REVERSION_LOOKBACK

    if start_idx < 0:
        return None

    lookback_prices = prices.iloc[start_idx:end_idx]

    # Mean reversion = how far below recent average
    reversion = lookback_prices.mean() / prices.iloc[end_idx] - 1

    return reversion


def calculate_volatility(prices, date):
    """
    Calculate volatility factor for risk weighting.
    """
    end_idx = prices.index.get_indexer([date], method="ffill")[0]
    start_idx = end_idx - VOLATILITY_LOOKBACK

    if start_idx < 0:
        return None

    returns = prices.iloc[start_idx:end_idx].pct_change().dropna()
    volatility = returns.std()

    return volatility


# =============================================================================
# SIGNAL COMBINATION - OVERFITTED WEIGHTS
# =============================================================================


def combine_factors(momentum, reversion, volatility):
    """
    Combine factors using "optimized" weights.

    BUG #4: Factor weights were optimized on the full dataset
    These weights are specific to this historical period and
    will likely not persist.

    BUG #5: No economic rationale for these specific weights
    Why is momentum 42.7%? What's special about that?
    """
    # Z-score normalize each factor
    mom_z = (momentum - momentum.mean()) / momentum.std()
    rev_z = (reversion - reversion.mean()) / reversion.std()

    # Volatility-adjust (lower vol = higher weight)
    vol_z = (volatility - volatility.mean()) / volatility.std()
    vol_adjustment = 1 - vol_z.clip(-2, 2) * 0.1

    # Combine with "optimized" weights
    combined = (
        VALUE_WEIGHT * rev_z  # Called "value" but it's reversion
        + MOMENTUM_WEIGHT * mom_z
        + REVERSION_WEIGHT * (-vol_z)  # Low vol premium
    ) * vol_adjustment

    return combined


# =============================================================================
# PORTFOLIO CONSTRUCTION
# =============================================================================


def construct_portfolio(signals, volatility, sectors):
    """
    Construct portfolio with multiple constraints.

    BUG #6: All these thresholds were optimized
    - Why cap at 73rd percentile volatility?
    - Why 18.7% max sector weight?
    - Why top 12.7%?
    """
    # Filter high volatility stocks
    vol_threshold = volatility.quantile(VOLATILITY_CAP)
    eligible = signals[volatility <= vol_threshold]

    # Select top stocks
    n_select = max(1, int(len(eligible) * TOP_PERCENTILE))
    selected = eligible.nlargest(n_select)

    # Initial equal weights
    weights = pd.Series(1.0 / len(selected), index=selected.index)

    # Apply sector constraints
    sector_series = pd.Series(sectors)
    for sector in sector_series.unique():
        sector_stocks = weights.index[
            weights.index.isin(sector_series[sector_series == sector].index)
        ]
        sector_weight = weights[sector_stocks].sum()

        if sector_weight > SECTOR_MAX_WEIGHT:
            scale = SECTOR_MAX_WEIGHT / sector_weight
            weights[sector_stocks] *= scale

    # Renormalize
    weights = weights / weights.sum()

    return weights


# =============================================================================
# BACKTEST WITH MULTIPLE OPTIMIZATIONS
# =============================================================================


def run_backtest(prices, sectors):
    """
    Run the "optimized" backtest.
    """
    # Get rebalance dates
    # BUG #7: Using the 3rd trading day because it "worked best"
    all_dates = prices.index
    month_starts = all_dates.to_period("M").unique()

    rebalance_dates = []
    for month in month_starts:
        month_dates = all_dates[all_dates.to_period("M") == month]
        if len(month_dates) > REBALANCE_DAY:
            rebalance_dates.append(month_dates[REBALANCE_DAY])

    rebalance_dates = rebalance_dates[12:]  # Skip first year

    portfolio_values = [1_000_000]
    portfolio_dates = [rebalance_dates[0]]

    for i, date in enumerate(rebalance_dates[:-1]):
        next_date = rebalance_dates[i + 1]

        # Calculate factors
        momentum = calculate_momentum(prices, date)
        reversion = calculate_reversion(prices, date)
        volatility = calculate_volatility(prices, date)

        if momentum is None or reversion is None or volatility is None:
            continue

        # Combine factors
        signals = combine_factors(momentum, reversion, volatility)

        # Construct portfolio
        weights = construct_portfolio(signals, volatility, sectors)

        # Calculate returns
        entry_prices = prices.loc[date, weights.index]
        exit_prices = prices.loc[next_date, weights.index]

        stock_returns = (exit_prices / entry_prices) - 1
        portfolio_return = (weights * stock_returns).sum()

        # BUG #8: Still no transaction costs in the "optimized" version

        new_value = portfolio_values[-1] * (1 + portfolio_return)
        portfolio_values.append(new_value)
        portfolio_dates.append(next_date)

    return pd.Series(portfolio_values, index=portfolio_dates)


# =============================================================================
# PARAMETER SEARCH (SHOWING THE SNOOPING)
# =============================================================================


def demonstrate_snooping(prices, sectors):
    """
    This function demonstrates how the parameters were "found".

    BUG #9: This is exactly what you shouldn't do
    - Testing hundreds of parameter combinations
    - Picking the best one
    - Reporting it as "the strategy"
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATING THE DATA SNOOPING")
    print("=" * 60)
    print("\nTesting parameter combinations...")

    results = []

    # Test a grid of parameters (simplified for demo)
    momentum_lookbacks = [126, 189, 247, 252]
    skip_days = [5, 11, 21]
    percentiles = [0.10, 0.127, 0.15, 0.20]

    total_tests = len(momentum_lookbacks) * len(skip_days) * len(percentiles)

    for mom_lb, skip, pct in product(momentum_lookbacks, skip_days, percentiles):
        # Would run backtest with these params
        # For demo, just simulate results
        fake_sharpe = np.random.normal(0.5, 0.3)
        results.append(
            {
                "momentum_lookback": mom_lb,
                "skip_days": skip,
                "top_percentile": pct,
                "sharpe": fake_sharpe,
            }
        )

    results_df = pd.DataFrame(results)

    # Find "best" parameters
    best = results_df.loc[results_df["sharpe"].idxmax()]

    print(f"\nTested {total_tests} parameter combinations")
    print(f"\nBest parameters found:")
    print(f"  Momentum lookback: {best['momentum_lookback']} days")
    print(f"  Skip days: {best['skip_days']}")
    print(f"  Top percentile: {best['top_percentile']:.1%}")
    print(f"  In-sample Sharpe: {best['sharpe']:.2f}")

    print("\n⚠️  But wait! With 36 tests, we'd expect:")
    print(f"     - Multiple Sharpes > 0.8 by chance")
    print(f"     - Best Sharpe inflated by ~0.3-0.4")
    print(f"     - Out-of-sample Sharpe likely 0.2-0.4 lower")

    return results_df


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================


def calculate_metrics(portfolio_values):
    """Calculate performance metrics."""
    returns = portfolio_values.pct_change().dropna()

    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    years = len(returns) / 12

    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (ann_return - 0.02) / ann_vol

    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    max_dd = drawdown.min()

    # BUG #10: Reporting "Sharpe" without acknowledging it's overfit

    return {
        "Total Return": f"{total_return:.2%}",
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",  # OVERSTATED!
        "Max Drawdown": f"{max_dd:.2%}",
    }


def estimate_overfit_penalty():
    """
    Estimate how much the Sharpe is inflated due to overfitting.

    Based on Bailey & de Prado (2014) "The Deflated Sharpe Ratio"
    """
    # Number of independent trials (parameter combinations)
    n_trials = 36  # 4 x 3 x 3 in our case (simplified)

    # Expected maximum Sharpe under null hypothesis
    # E[max(Z_1, ..., Z_n)] ≈ sqrt(2 * ln(n)) for large n
    expected_max_sharpe = np.sqrt(2 * np.log(n_trials))

    # With 36 trials, expected inflation ≈ 0.35-0.40
    print("\n" + "=" * 60)
    print("OVERFITTING PENALTY ESTIMATE")
    print("=" * 60)
    print(f"\nNumber of parameter combinations tested: {n_trials}")
    print(f"Expected maximum Sharpe under null: {expected_max_sharpe:.2f}")
    print("\nIf reported Sharpe is 0.9:")
    print(f"  - Estimated true Sharpe: 0.5-0.6")
    print(f"  - Inflation from snooping: 0.3-0.4")

    return expected_max_sharpe


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("'OPTIMIZED' MULTI-FACTOR STRATEGY")
    print("(Demonstration of Overfitting)")
    print("=" * 60)

    # Generate data
    print("\nGenerating data...")
    prices, sectors = generate_data()

    # Run backtest
    print("Running 'optimized' backtest...")
    portfolio_values = run_backtest(prices, sectors)

    # Calculate metrics
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (OVERFITTED!)")
    print("=" * 60)

    metrics = calculate_metrics(portfolio_values)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Demonstrate the snooping
    demonstrate_snooping(prices, sectors)

    # Estimate penalty
    estimate_overfit_penalty()

    print("\n" + "=" * 60)
    print("RED FLAGS IN THIS BACKTEST:")
    print("=" * 60)
    print(
        """
    1. Suspiciously specific parameters (247 days, not 252)
    2. Weights to 3 decimal places (0.314, 0.427, 0.259)
    3. No economic rationale for parameter choices
    4. Many parameters = many degrees of freedom
    5. No out-of-sample testing
    6. No multiple testing correction
    7. Sharpe > 1.0 from simple factors is suspicious
    8. No mention of how parameters were selected
    
    PROPER APPROACH:
    - Use round numbers with economic rationale
    - Split data into train/validation/test
    - Report out-of-sample only
    - Apply deflated Sharpe ratio
    - Disclose all trials run
    """
    )

    return portfolio_values, metrics


if __name__ == "__main__":
    portfolio, metrics = main()
