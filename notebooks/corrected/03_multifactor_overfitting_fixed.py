"""
Multi-Factor Strategy - Overfitting Corrected
=============================================

Now supports Real Data via CSV and Robust Imports.
Usage: python3 notebooks/corrected/03_multifactor_overfitting_fixed.py --csv among_synth.csv
"""

import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

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

class FactorLibrary:
    @staticmethod
    def momentum(p, d): 
        try: return (p.loc[d]/p.shift(252).loc[d]) - 1
        except: return None
    @staticmethod
    def reversion(p, d):
        try: return (p.shift(21).loc[d].mean()/p.loc[d]) - 1
        except: return None
    @staticmethod
    def volatility(p, d):
        try: return p.pct_change().rolling(21).std().loc[d]
        except: return None

class PortfolioOptimizer:
    def get_weights(self, mom, rev, vol):
        # Normalize
        z = lambda x: (x - x.mean())/x.std()
        score = z(mom) + z(rev) - z(vol)
        
        # Select Top 20%
        n_sel = max(1, int(len(score)*0.2))
        sel = score.nlargest(n_sel).index
        return pd.Series(1.0/len(sel), index=sel)

class Statistics:
    @staticmethod
    def deflated_sharpe(sharpe, n_trials, n_obs):
        e_max = np.sqrt(2 * np.log(n_trials)) if n_trials > 1 else 0
        var = (1 + 0.5 * sharpe**2)/n_obs
        if var <= 0: return 0.0
        return stats.norm.cdf((sharpe - e_max) / np.sqrt(var))

def run_strategy(prices, start, end):
    dates = pd.date_range(start, end, freq='MS')
    port = [1e6]
    lib, opt = FactorLibrary(), PortfolioOptimizer()
    
    # Filter dates to data range
    dates = dates[(dates >= prices.index[0]) & (dates <= prices.index[-1])]
    
    for i, date in enumerate(dates[:-1]):
        try:
            mom = lib.momentum(prices, date)
            rev = lib.reversion(prices, date)
            vol = lib.volatility(prices, date)
            
            if any(x is None for x in [mom, rev, vol]): 
                port.append(port[-1])
                continue

            w = opt.get_weights(mom, rev, vol)
            
            entry = prices.index.get_indexer([date], method='ffill')[0]
            exit_i = prices.index.get_indexer([dates[i+1]], method='ffill')[0]
            
            p_entry = prices.iloc[entry][w.index]
            p_exit = prices.iloc[exit_i][w.index]
            
            ret = (p_exit / p_entry) - 1
            port.append(port[-1] * (1 + ret.mean() - 0.0020)) # 20bps cost
        except:
            port.append(port[-1])
            
    return pd.Series(port)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()
    
    source = 'csv' if args.csv else 'synthetic'
    adapter = DataAdapter(source, args.csv)
    
    try:
        # Load wider range to ensure lookbacks work
        prices, _ = adapter.get_data('2004-01-01', '2023-12-31')
        
        # Run Test
        res = run_strategy(prices, '2010-01-01', '2023-12-31')
        
        if len(res) > 1:
            ret = res.pct_change().dropna()
            sharpe = (ret.mean()*12) / (ret.std()*np.sqrt(12))
            dsr = Statistics.deflated_sharpe(sharpe, 10, len(ret))
            
            print(f"\nRESULTS: Multi-Factor [{source.upper()}]")
            print(f"Final Value:     ${res.iloc[-1]:,.2f}")
            print(f"Sharpe Ratio:    {sharpe:.2f}")
            print(f"Deflated Sharpe: {dsr:.1%}")
        else:
            print("Not enough data.")
            
    except Exception as e:
        logger.error(f"Strategy failed: {e}")

if __name__ == "__main__":
    main()