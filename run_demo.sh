echo '#!/bin/bash
python3 generate_synth_csv.py
echo "--- 🏥 STARTING CLI DIAGNOSTICS ---"
python3 notebooks/corrected/run_diagnosis.py --check momentum
python3 notebooks/corrected/run_diagnosis.py --check value
python3 notebooks/corrected/run_diagnosis.py --check multifactor
python3 notebooks/corrected/run_diagnosis.py --check ml
echo "--- 🚀 DIAGNOSTICS COMPLETE. LAUNCHING DASHBOARD ---"
python3 -m streamlit run dashboard.py' > run_demo.sh

chmod +x run_demo.sh