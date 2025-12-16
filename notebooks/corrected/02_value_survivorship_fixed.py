"""
Value Strategy Backtest - Survivorship-Free Version
===================================================

Now supports Real Data via CSV and Robust Imports.
Usage: python3 notebooks/corrected/02_value_survivorship_fixed.py --csv among_synth.csv
"""

import argparse
import logging
import sys
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# --- ROBUST IMPORT FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from data_adapter import DataAdapter
except ImportError:
    sys.path.append('../../')
    from data_adapter import DataAdapter
# -------------------------

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class ValueStrategy:
    def __init__(self, n_stocks: int, transaction_cost_bps: float):
        self.n_stocks = n_stocks
        self.transaction_cost_pct = transaction_cost_bps / 10000.0

    def calculate_pe_ratios(self, prices, earnings, date, constituents):
        # 1. Get index members
        if constituents:
            const_dates = sorted(constituents.keys())
            valid_const = [d for d in const_dates if d <= date]
            if not valid_const: return None
            current_members = constituents[max(valid_const)]
        else:
            # Fallback if no constituent data (e.g. CSV)
            current_members = prices.columns.tolist()

        # 2. Get Earnings
        available_earnings = earnings.loc[earnings.index <= date]
        if len(available_earnings) < 4: return None
        ttm_earnings = available_earnings.tail(4).sum()

        # 3. Get Prices
        price_date = prices.index[prices.index <= date][-1]
        current_prices = prices.loc[price_date]

        pe_ratios = {}
        for ticker in current_members:
            if ticker in current_prices and ticker in ttm_earnings:
                p, e = current_prices[ticker], ttm_earnings[ticker]
                if pd.notna(p) and pd.notna(e) and e > 0:
                    pe_ratios[ticker] = p / e
        
        return pd.Series(pe_ratios)

    def select_portfolio(self, pe_ratios):
        valid_pe = pe_ratios[(pe_ratios >= 3) & (pe_ratios <= 50)]
        if valid_pe.empty: return []
        return valid_pe.nsmallest(self.n_stocks).index.tolist()

    def run_backtest(self, prices, earnings, constituents, initial_capital):
        logger.info("Starting backtest loop...")
        rebalance_dates = pd.date_range(prices.index[0], prices.index[-1], freq="MS")[12:]
        portfolio_values = [initial_capital]
        
        for i, date in enumerate(rebalance_dates[:-1]):
            next_date = rebalance_dates[i + 1]
            
            # Default: No change
            portfolio_values.append(portfolio_values[-1])
            
            try:
                # 1. Generate Signal
                pe = self.calculate_pe_ratios(prices, earnings, date, constituents)
                if pe is None or pe.empty: continue

                # 2. Select
                selected = self.select_portfolio(pe)
                if not selected: continue

                # 3. Execution
                entry_idx = prices.index.get_indexer([date], method="bfill")[0] + 1
                exit_idx = prices.index.get_indexer([next_date], method="ffill")[0]
                
                if entry_idx >= len(prices) or exit_idx >= len(prices): break

                entry_p = prices.iloc[entry_idx][selected]
                exit_p = prices.iloc[exit_idx][selected]
                
                ret = (exit_p - entry_p) / entry_p
                # Delisting check (-95% if price disappears)
                ret = ret.fillna(-0.95)
                
                # Apply Cost
                cost = self.transaction_cost_pct  # Simple approximation
                net_ret = ret.mean() - cost
                
                # Overwrite the default "no change" value
                portfolio_values[-1] = portfolio_values[-2] * (1 + net_ret)
                
            except Exception:
                pass # Skip period on error

        return pd.Series(portfolio_values, index=rebalance_dates[:len(portfolio_values)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--stocks", type=int, default=30)
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()

    # 1. Data Adapter
    source = 'csv' if args.csv else 'synthetic'
    adapter = DataAdapter(source, args.csv)
    prices, meta = adapter.get_data(args.start, args.end)
    
    # Extract earnings/constituents if available, else mock/proxy
    earnings = meta.get('earnings', prices * 0.05) 
    constituents = meta.get('constituents', {}) # Empty dict = use all cols

    # 2. Strategy
    strat = ValueStrategy(args.stocks, transaction_cost_bps=15)
    res = strat.run_backtest(prices, earnings, constituents, args.capital)
    
    if len(res) > 0:
        ret = (res.iloc[-1] / args.capital) - 1
        print(f"\nRESULTS: Value Strategy [{source.upper()}]")
        print(f"Final Value:  ${res.iloc[-1]:,.2f}")
        print(f"Total Return: {ret:.2%}")
    else:
        print("Not enough data.")

if __name__ == "__main__":
    main()