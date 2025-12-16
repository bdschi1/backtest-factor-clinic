"""
ML Return Prediction - Leakage Free
===================================

Now supports Real Data via CSV and Robust Imports.
Usage: python3 notebooks/corrected/04_ml_leakage_fixed.py --csv among_synth.csv
"""

import argparse
import logging
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

class FeatureEngineer:
    @staticmethod
    def create_dataset(prices, horizon=21):
        df = pd.DataFrame(index=prices.index)
        
        # Features (Using Shift to prevent leakage)
        df['ret_5d'] = prices.pct_change(5).shift(1)
        df['ret_21d'] = prices.pct_change(21).shift(1)
        df['vol_21d'] = prices.pct_change().rolling(21).std().shift(1)
        
        # Target (Future return)
        future_ret = prices.pct_change(horizon).shift(-horizon)
        df['target'] = (future_ret > 0).astype(int)
        
        return df.dropna()

class MLEngine:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=3)
        self.scaler = StandardScaler()

    def run_walk_forward(self, X, y, n_splits=5):
        fold_size = len(X) // (n_splits + 1)
        preds, truth = [], []
        
        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            test_start = train_end + 21 # Embargo
            test_end = test_start + fold_size
            
            if test_end >= len(X): break
            
            # Train (Purge last 5 days)
            X_train = X.iloc[:train_end-5]
            y_train = y.iloc[:train_end-5]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            if len(X_test) == 0: continue
            
            X_train_s = self.scaler.fit_transform(X_train)
            X_test_s = self.scaler.transform(X_test)
            
            self.model.fit(X_train_s, y_train)
            preds.extend(self.model.predict(X_test_s))
            truth.extend(y_test)
            
        return accuracy_score(truth, preds) if preds else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()
    
    source = 'csv' if args.csv else 'synthetic'
    adapter = DataAdapter(source, args.csv)
    
    # For ML, we process one stock (e.g. Market Proxy or first stock)
    prices, _ = adapter.get_data('2010-01-01', '2023-12-31')
    stock_price = prices.iloc[:, 0] # Take first column
    
    logger.info(f"Processing {stock_price.name} for ML...")
    
    df = FeatureEngineer.create_dataset(stock_price)
    engine = MLEngine()
    acc = engine.run_walk_forward(df.drop(columns=['target']), df['target'])
    
    print(f"\nRESULTS: ML Prediction [{source.upper()}]")
    print(f"Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()