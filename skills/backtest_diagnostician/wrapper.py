import argparse
import sys
import os
import subprocess

# Define paths to the actual logic
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks/corrected")

def run_script(script_name, args):
    script_path = os.path.join(NOTEBOOKS_DIR, script_name)
    cmd = [sys.executable, script_path] + args
    print(f"🏥 AGENT: Executing {script_name}...")
    subprocess.run(cmd, check=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=["momentum", "value", "multifactor", "ml"], required=True)
    parser.add_argument("--data", default="among_synth.csv", help="Path to market data CSV")
    args = parser.parse_args()

    # Map commands to scripts
    script_map = {
        "momentum": "01_momentum_lookahead_fixed.py",
        "value": "02_value_survivorship_fixed.py",
        "multifactor": "03_multifactor_overfitting_fixed.py",
        "ml": "04_ml_leakage_fixed.py"
    }

    # Construct arguments for the target script
    script_args = ["--csv", args.data]
    
    # Run
    if args.check in script_map:
        run_script(script_map[args.check], script_args)
    else:
        print("Unknown check type.")

if __name__ == "__main__":
    main()