import sys
import os
import argparse
import importlib
import pandas as pd
import numpy as np

# --- 1. SETUP: Add Project Root to Path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import DataAdapter now that path is fixed
try:
    from data_adapter import DataAdapter
except ImportError:
    print("❌ Critical Error: Could not import DataAdapter. Check sys.path.")
    sys.exit(1)

# --- 2. Dynamic Import Helper ---
def load_module(module_name):
    return importlib.import_module(f"notebooks.corrected.{module_name}")

try:
    mom = load_module("01_momentum_lookahead_fixed")
    val = load_module("02_value_survivorship_fixed")
    multi = load_module("03_multifactor_overfitting_fixed")
    ml = load_module("04_ml_leakage_fixed")
except ImportError as e:
    print(f"⚠️ Warning: Could not load strategy modules. Error: {e}")
    sys.exit(1)

# --- 3. Main CLI Logic ---
def main():
    parser = argparse.ArgumentParser(description="Quant Clinic CLI")
    parser.add_argument("--check", choices=["momentum", "value", "multifactor", "ml"], required=True)
    parser.add_argument("--data", default="among_synth.csv", help="Path to market data CSV")
    args = parser.parse_args()

    # --- Data Loading ---
    print(f"📊 Loading data from {args.data}...")
    
    # Handle path check
    data_path = args.data
    if not os.path.exists(data_path):
        potential_path = os.path.join(project_root, args.data)
        if os.path.exists(potential_path):
            data_path = potential_path
        else:
            print(f"❌ Error: Data file '{args.data}' not found.")
            sys.exit(1)

    adapter = DataAdapter(source_type='csv', csv_path=data_path)
    prices, meta = adapter.get_data('2000-01-01', '2024-01-01')
    
    # Extract helpers
    earnings = meta.get('earnings', prices * 0.05)
    constituents = meta.get('constituents', {})
    market_caps = meta.get('market_caps', prices)

    # --- Strategy Router ---
    
    # === 1. MOMENTUM ===
    if args.check == "momentum":
        print("🏥 Running Momentum Diagnosis (Look-Ahead Bias Check)...")
        # FIX: Changed 'top_pct' to 'top_percentile' to match your class definition
        engine = mom.MomentumEngine(lookback_months=12, top_percentile=0.1)
        tester = mom.Backtester(cost_bps=10, slippage_bps=5)
        
        res = tester.run(prices, market_caps, engine, capital=1_000_000)
        
        ret = (res.iloc[-1] / 1_000_000) - 1
        print(f"   ✅ Result: {ret:.2%} Total Return")

    # === 2. VALUE ===
    elif args.check == "value":
        print("🏥 Running Value Diagnosis (Survivorship Bias Check)...")
        strat = val.ValueStrategy(n_stocks=30, transaction_cost_bps=15)
        
        res = strat.run_backtest(prices, earnings, constituents, initial_capital=1_000_000)
        
        ret = (res.iloc[-1] / 1_000_000) - 1
        print(f"   ✅ Result: {ret:.2%} Total Return")

    # === 3. MULTI-FACTOR ===
    elif args.check == "multifactor":
        print("🏥 Running Multi-Factor Diagnosis (Overfitting Check)...")
        # This module uses a direct function call
        res = multi.run_strategy(prices, '2010-01-01', '2023-12-31')
        
        if len(res) > 1:
            ret_series = res.pct_change().dropna()
            sharpe = (ret_series.mean() * 12) / (ret_series.std() * np.sqrt(12))
            dsr = multi.Statistics.deflated_sharpe(sharpe, n_trials=10, n_obs=len(ret_series))
            print(f"   ✅ Result: Sharpe={sharpe:.2f} | Deflated Sharpe Prob={dsr:.1%}")

    # === 4. ML ===
    elif args.check == "ml":
        print("🏥 Running ML Diagnosis (Leakage Check)...")
        # ML checks one asset at a time
        target_asset = prices.iloc[:, 0] 
        print(f"   >> Training on {target_asset.name}...")
        
        df = ml.FeatureEngineer.create_dataset(target_asset)
        engine = ml.MLEngine()
        acc = engine.run_walk_forward(df.drop(columns=['target']), df['target'])
        
        print(f"   ✅ Result: Out-of-Sample Accuracy = {acc:.2%}")

if __name__ == "__main__":
    main()