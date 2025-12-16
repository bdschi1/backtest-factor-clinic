# ML-Based Return Prediction with Label Leakage
# ==============================================
#
# This notebook demonstrates a "sophisticated" ML approach to return prediction.
#
# ⚠️ WARNING: This model has SEVERE LABEL LEAKAGE and other ML anti-patterns.
# See /notebooks/corrected/04_ml_leakage_fixed.py for proper methodology.

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

PREDICTION_HORIZON = 21  # Predict 21-day forward returns
TRAIN_SIZE = 0.7
RANDOM_STATE = 42

# =============================================================================
# DATA GENERATION
# =============================================================================


def generate_market_data(n_stocks=200, n_years=10):
    """Generate synthetic market data."""
    np.random.seed(42)

    start = "2013-01-01"
    end = "2022-12-31"
    dates = pd.date_range(start, end, freq="B")

    tickers = [f"STK_{i:03d}" for i in range(n_stocks)]

    # Generate prices
    prices = {}
    for ticker in tickers:
        returns = np.random.normal(0.0003, 0.02, len(dates))
        prices[ticker] = 100 * np.cumprod(1 + returns)

    prices_df = pd.DataFrame(prices, index=dates)

    return prices_df


# =============================================================================
# FEATURE ENGINEERING - WITH LEAKAGE
# =============================================================================


def create_features_with_leakage(prices, date_idx):
    """
    Create features for ML model.

    ⚠️ THIS FUNCTION CONTAINS MULTIPLE FORMS OF LEAKAGE!
    """
    features = {}

    # Technical features (these are OK)
    returns_5d = prices.pct_change(5).iloc[date_idx]
    returns_21d = prices.pct_change(21).iloc[date_idx]
    returns_63d = prices.pct_change(63).iloc[date_idx]

    features["momentum_5d"] = returns_5d
    features["momentum_21d"] = returns_21d
    features["momentum_63d"] = returns_63d

    # Volatility features (these are OK)
    vol_21d = prices.pct_change().iloc[date_idx - 21 : date_idx].std()
    features["volatility_21d"] = vol_21d

    # BUG #1: LEAKAGE - Using future data in "moving average crossover"
    # The 5-day MA includes today's price, which we're trying to predict returns from
    ma_5 = prices.iloc[date_idx - 4 : date_idx + 1].mean()  # Includes today
    ma_20 = prices.iloc[date_idx - 19 : date_idx + 1].mean()  # Includes today
    features["ma_crossover"] = (ma_5 / ma_20) - 1

    # BUG #2: SEVERE LEAKAGE - Using future returns as a "sentiment" proxy
    # This is computing the actual target and calling it a feature!
    future_returns = (
        prices.iloc[date_idx : date_idx + PREDICTION_HORIZON].pct_change().mean()
    )
    features["market_sentiment"] = future_returns  # THIS IS THE TARGET!

    # BUG #3: LEAKAGE - Using same-day volume-weighted information
    # In reality, you can't know full-day volume until market close
    # And you can't trade on close prices

    # BUG #4: LEAKAGE - Using next-day open
    if date_idx + 1 < len(prices):
        next_day_gap = (prices.iloc[date_idx + 1] / prices.iloc[date_idx]) - 1
        features["overnight_gap"] = next_day_gap  # LEAKAGE: Future data!
    else:
        features["overnight_gap"] = 0

    # BUG #5: INFORMATION LEAKAGE through standardization
    # We'll scale using ALL data (including future) later

    return features


def create_labels(prices, date_idx):
    """
    Create target labels (future returns).
    """
    current_price = prices.iloc[date_idx]

    if date_idx + PREDICTION_HORIZON >= len(prices):
        return None

    future_price = prices.iloc[date_idx + PREDICTION_HORIZON]
    forward_return = (future_price / current_price) - 1

    # Convert to binary: positive vs negative return
    labels = (forward_return > 0).astype(int)

    return labels


def build_dataset_with_leakage(prices):
    """
    Build feature matrix and labels.

    BUG #6: TEMPORAL LEAKAGE in train/test split
    We'll mix data from different time periods randomly.
    """
    all_features = []
    all_labels = []
    all_dates = []

    # Start after enough history for features
    for date_idx in range(63, len(prices) - PREDICTION_HORIZON - 1):
        features = create_features_with_leakage(prices, date_idx)
        labels = create_labels(prices, date_idx)

        if labels is None:
            continue

        # Convert features to DataFrame row
        feature_df = pd.DataFrame(features)

        all_features.append(feature_df)
        all_labels.append(labels)
        all_dates.append(prices.index[date_idx])

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)

    return X, y, all_dates


# =============================================================================
# MODEL TRAINING - WITH MORE ISSUES
# =============================================================================


def train_model_with_issues(X, y):
    """
    Train ML model with problematic methodology.

    BUG #7: RANDOM SPLIT instead of temporal split
    This leaks future information into training.
    """
    # Scale features
    # BUG #8: Fitting scaler on ALL data (including test)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Fits on everything!

    # BUG #9: Random train/test split (not temporal)
    # This means training data contains future observations
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        # shuffle=True is default - mixing time periods!
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Train model
    # BUG #10: No hyperparameter tuning on validation set
    # Using test set for both tuning and evaluation
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    return model, scaler, train_acc, test_acc


def demonstrate_leakage_impact(X, y, dates):
    """
    Show the difference between leaked and proper methodology.
    """
    print("\n" + "=" * 60)
    print("DEMONSTRATING LEAKAGE IMPACT")
    print("=" * 60)

    # Method 1: With leakage (random split)
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=RANDOM_STATE
    )

    model_leaked = RandomForestClassifier(n_estimators=100, random_state=42)
    model_leaked.fit(X_train, y_train)

    leaked_train_acc = accuracy_score(y_train, model_leaked.predict(X_train))
    leaked_test_acc = accuracy_score(y_test, model_leaked.predict(X_test))

    print(f"\n📊 With Leakage (Random Split):")
    print(f"   Train Accuracy: {leaked_train_acc:.1%}")
    print(f"   Test Accuracy:  {leaked_test_acc:.1%}")
    print(f"   ⚠️  Test accuracy is inflated due to temporal leakage!")

    # Method 2: Proper temporal split
    split_idx = int(len(X) * 0.7)

    X_train_temp = X.iloc[:split_idx]
    X_test_temp = X.iloc[split_idx:]
    y_train_temp = y.iloc[:split_idx]
    y_test_temp = y.iloc[split_idx:]

    # Scale properly (fit on train only)
    scaler_proper = StandardScaler()
    X_train_scaled = scaler_proper.fit_transform(X_train_temp)
    X_test_scaled = scaler_proper.transform(X_test_temp)

    model_proper = RandomForestClassifier(n_estimators=100, random_state=42)
    model_proper.fit(X_train_scaled, y_train_temp)

    proper_train_acc = accuracy_score(
        y_train_temp, model_proper.predict(X_train_scaled)
    )
    proper_test_acc = accuracy_score(y_test_temp, model_proper.predict(X_test_scaled))

    print(f"\n📊 Without Leakage (Temporal Split):")
    print(f"   Train Accuracy: {proper_train_acc:.1%}")
    print(f"   Test Accuracy:  {proper_test_acc:.1%}")
    print(f"   ℹ️  More realistic, but still inflated due to feature leakage")

    # Feature importance analysis
    print("\n📊 Feature Importances (Leaked Model):")
    importances = pd.Series(
        model_leaked.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    for feat, imp in importances.items():
        flag = "⚠️ LEAKED!" if feat in ["market_sentiment", "overnight_gap"] else ""
        print(f"   {feat}: {imp:.3f} {flag}")

    return {
        "leaked_test_acc": leaked_test_acc,
        "proper_test_acc": proper_test_acc,
        "accuracy_inflation": leaked_test_acc - proper_test_acc,
    }


# =============================================================================
# BACKTEST WITH LEAKED MODEL
# =============================================================================


def backtest_strategy(prices, model, scaler, X, y, dates):
    """
    Backtest the ML strategy.

    BUG #11: Using same data for training and backtest
    """
    # Get predictions for all dates
    X_scaled = scaler.transform(X)
    predictions = model.predict_proba(X_scaled)[:, 1]  # Probability of positive return

    # Create trading signals
    signals = pd.Series(predictions, index=dates[: len(predictions)])

    portfolio_values = [1_000_000]
    portfolio_dates = [dates[0]]

    # Trade top quartile
    for i, date in enumerate(dates[:-PREDICTION_HORIZON]):
        if date not in prices.index:
            continue

        # Get signal
        signal = signals.iloc[i] if i < len(signals) else 0.5

        # Simple strategy: go long if signal > 0.5
        if signal > 0.6:  # "Confident" long
            position = 1.0
        elif signal < 0.4:  # "Confident" short
            position = -0.5
        else:
            position = 0.0

        # Calculate return
        if i + PREDICTION_HORIZON < len(dates):
            entry_date = date
            exit_date = dates[i + PREDICTION_HORIZON]

            if exit_date in prices.index and entry_date in prices.index:
                market_return = (
                    prices.loc[exit_date].mean() / prices.loc[entry_date].mean()
                ) - 1
                portfolio_return = position * market_return

                new_value = portfolio_values[-1] * (1 + portfolio_return)
                portfolio_values.append(new_value)
                portfolio_dates.append(exit_date)

    return pd.Series(portfolio_values, index=portfolio_dates[: len(portfolio_values)])


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================


def calculate_metrics(portfolio_values):
    """Calculate performance metrics."""
    returns = portfolio_values.pct_change().dropna()

    total_return = portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1
    years = len(returns) / 252

    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
    sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0

    cummax = portfolio_values.cummax()
    drawdown = (portfolio_values - cummax) / cummax
    max_dd = drawdown.min()

    return {
        "Total Return": f"{total_return:.2%}",
        "Ann. Return": f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}",
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 60)
    print("ML-BASED RETURN PREDICTION")
    print("(Demonstration of Label Leakage)")
    print("=" * 60)

    # Generate data
    print("\nGenerating market data...")
    prices = generate_market_data()

    # Build dataset with leakage
    print("\nBuilding feature dataset (with leakage)...")
    X, y, dates = build_dataset_with_leakage(prices)
    print(f"  Samples: {len(X)}")
    print(f"  Features: {list(X.columns)}")

    # Train model
    print("\nTraining model (with methodological issues)...")
    model, scaler, train_acc, test_acc = train_model_with_issues(X, y)

    print(f"\n📊 Model Performance (Inflated!):")
    print(f"   Train Accuracy: {train_acc:.1%}")
    print(f"   Test Accuracy:  {test_acc:.1%}")

    # Demonstrate leakage impact
    impact = demonstrate_leakage_impact(X, y, dates)

    # Run backtest
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (UNRELIABLE)")
    print("=" * 60)

    portfolio_values = backtest_strategy(prices, model, scaler, X, y, dates)
    metrics = calculate_metrics(portfolio_values)

    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Print warnings
    print("\n" + "=" * 60)
    print("⚠️  LEAKAGE AND ISSUES IN THIS MODEL:")
    print("=" * 60)
    print(
        """
    FEATURE LEAKAGE:
    1. 'market_sentiment' IS the target (forward returns)!
    2. 'overnight_gap' uses next-day prices
    3. Moving averages include current-day close
    
    METHODOLOGICAL ISSUES:
    4. Random train/test split (not temporal)
    5. Scaler fit on all data (including test)
    6. No separate validation set
    7. Hyperparameters not tuned properly
    8. Same data for training and backtest
    
    IMPACT:
    - Accuracy inflation: ~{:.0%}
    - This model would FAIL in live trading
    - The features with highest importance are the leaked ones!
    
    PROPER APPROACH:
    - Use temporal split (train on past, test on future)
    - Fit scaler on training data only
    - Create features using only past data
    - Use walk-forward validation
    - Apply purging and embargo periods
    """.format(
            impact["accuracy_inflation"]
        )
    )

    return model, portfolio_values, metrics


if __name__ == "__main__":
    model, portfolio, metrics = main()
