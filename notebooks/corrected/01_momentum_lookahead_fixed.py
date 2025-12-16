"""
Momentum Strategy Backtest - Look-Ahead Bias Free
=================================================

A professional implementation of a cross-sectional momentum strategy ensuring 
Point-in-Time (PIT) data usage and realistic execution delays.

Usage:
    # Run with synthetic data
    python3 notebooks/corrected/01_momentum_lookahead_fixed.py
    
    # Run with real data
    python3 notebooks/corrected/01_momentum_lookahead_fixed.py --csv among_synth.csv
"""

import argparse
import logging
import sys
import os
from typing import List, Optional

import numpy as np
import pandas as pd

# --- ROBUST IMPORT FIX ---
# Calculate the project root relative to THIS file's location
# (Go up 2 levels: notebooks/corrected/ -> notebooks/ -> root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

# Add root to Python path if not already there
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from data_adapter import DataAdapter
except ImportError:
    # Fallback if standard import fails
    sys.path.append('../../')
    from data_adapter import DataAdapter
# -------------------------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class MomentumEngine:
    def __init__(self, lookback_months: int, top_percentile: float):
        self.lookback_months = lookback_months
        self.top_percentile = top_percentile

    def get_available_date(self, dates: pd.Index, target_date: pd.Timestamp, lag_days: int = 1) -> Optional[pd.Timestamp]:
        """Find the most recent valid data point accounting for reporting lag."""
        cutoff = target_date - pd.Timedelta(days=lag_days)
        valid_dates = dates[dates <= cutoff]
        return valid_dates[-1] if not valid_dates.empty else None

    def calculate_signal(self, prices: pd.DataFrame, decision_date: pd.Timestamp) -> Optional[pd.Series]:
        """Calculate momentum using only data available at decision time."""
        # 1. Available 'Now'
        t_now = self.get_available_date(prices.index, decision_date, lag_days=1)
        if t_now is None: 
            return None
        
        # 2. Available 'Lookback Start'
        # Note: We look back from the available date, not the decision date
        t_start = t_now - pd.DateOffset(months=self.lookback_months)
        t_start_valid = self.get_available_date(prices.index, t_start, lag_days=0)
        
        if t_start_valid is None:
            return None

        # 3. Calculate Returns
        p_now = prices.loc[t_now]
        p_start = prices.loc[t_start_valid]
        
        return (p_now / p_start) - 1

    def select_universe(self, market_caps: pd.DataFrame, decision_date: pd.Timestamp) -> List[str]:
        """Filter universe using only market caps known at decision time."""
        t_valid = self.get_available_date(market_caps.index, decision_date, lag_days=1)
        if t_valid is None:
            return []
            
        caps = market_caps.loc[t_valid]
        median_cap = caps.quantile(0.5)
        return caps[caps > median_cap].index.tolist()


class Backtester:
    def __init__(self, cost_bps: float, slippage_bps: float):
        self.total_cost_bps = cost_bps + slippage_bps

    def run(self, prices: pd.DataFrame, market_caps: pd.DataFrame, 
            engine: MomentumEngine, capital: float) -> pd.Series:
        
        logger.info("Starting backtest execution...")
        
        # Rebalance Monthly
        rebalance_dates = pd.date_range(prices.index[0], prices.index[-1], freq="BMS")
        
        # Skip first year to allow lookback (buffer)
        start_idx = 13
        if len(rebalance_dates) < start_idx:
            # Not a critical error if dataset is short, but warn user
            if len(rebalance_dates) > 2:
                logger.warning("Short dataset detected. Using minimal lookback buffer.")
                start_idx = 2
            else:
                raise ValueError("Dataset too short for backtest.")
            
        rebalance_dates = rebalance_dates[start_idx:]
        
        portfolio = [capital]
        dates = [rebalance_dates[0]]
        current_weights = pd.Series(dtype=float)
        
        for i, date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i+1]
            
            # 1. Define Universe
            universe = engine.select_universe(market_caps, date)
            
            # 2. Calculate Signal
            raw_momentum = engine.calculate_signal(prices, date)
            
            if raw_momentum is None or not universe:
                portfolio.append(portfolio[-1])
                dates.append(next_date)
                continue
                
            # 3. Select Top Decile within Universe
            valid_mom = raw_momentum[raw_momentum.index.isin(universe)].dropna()
            
            if valid_mom.empty:
                portfolio.append(portfolio[-1])
                dates.append(next_date)
                continue

            n_select = max(1, int(len(valid_mom) * engine.top_percentile))
            selected = valid_mom.nlargest(n_select).index
            
            target_weights = pd.Series(1.0 / len(selected), index=selected)
            
            # 4. Transaction Costs
            # Turnover calculation
            combined_idx = current_weights.index.union(target_weights.index)
            turnover = (target_weights.reindex(combined_idx, fill_value=0) - 
                        current_weights.reindex(combined_idx, fill_value=0)).abs().sum()
            
            cost = turnover * (self.total_cost_bps / 10000) * portfolio[-1]
            
            # 5. Execution (Next Day Returns)
            # Find next trading day for entry
            try:
                entry_idx = prices.index.get_indexer([date], method='bfill')[0] + 1
                exit_idx = prices.index.get_indexer([next_date], method='ffill')[0]
                
                if entry_idx >= len(prices) or exit_idx >= len(prices):
                    break

                entry_prices = prices.iloc[entry_idx][selected]
                exit_prices = prices.iloc[exit_idx][selected]
                
                period_return = (exit_prices - entry_prices) / entry_prices
                avg_return = period_return.mean()
                
                # Check for NaN (if no stocks selected or data missing)
                if np.isnan(avg_return):
                    avg_return = 0.0

                new_equity = (portfolio[-1] - cost) * (1 + avg_return)
                
                portfolio.append(new_equity)
                dates.append(next_date)
                current_weights = target_weights
                
            except (IndexError, KeyError):
                logger.warning(f"Data missing for execution at {date}. Skipping.")
                portfolio.append(portfolio[-1])
                dates.append(next_date)
                
        return pd.Series(portfolio, index=dates)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Momentum Strategy")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--lookback", type=int, default=12, help="Momentum lookback months")
    parser.add_argument("--top-decile", type=float, default=0.10, help="Top percentile to select")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--csv", type=str, help="Path to CSV file (optional)")
    args = parser.parse_args()

    # 1. Initialize Adapter
    source = 'csv' if args.csv else 'synthetic'
    adapter = DataAdapter(source_type=source, csv_path=args.csv)
    
    # 2. Get Data
    try:
        prices, meta = adapter.get_data(args.start, args.end)
        # If market_caps missing (e.g. simple CSV), use prices as proxy (imperfect but runnable)
        market_caps = meta.get('market_caps', prices) 
        
        logger.info(f"Loaded {len(prices)} rows of price data.")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        sys.exit(1)
    
    # 3. Run Strategy
    engine = MomentumEngine(args.lookback, args.top_decile)
    tester = Backtester(cost_bps=args.cost_bps, slippage_bps=5.0)
    
    results = tester.run(prices, market_caps, engine, args.capital)
    
    # 4. Stats
    if len(results) > 1:
        total_ret = (results.iloc[-1] / args.capital) - 1
        ann_vol = results.pct_change().std() * np.sqrt(12)
        sharpe = (results.pct_change().mean() * 12 - 0.02) / ann_vol if ann_vol > 0 else 0
        
        print(f"\nRESULTS: Momentum ({args.lookback}m) [{source.upper()}]")
        print(f"Final Value:  ${results.iloc[-1]:,.2f}")
        print(f"Total Return: {total_ret:.2%}")
        print(f"Sharpe Ratio: {sharpe:.2f}")
    else:
        print("Not enough data to generate results.")

if __name__ == "__main__":
    main()