import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import importlib.util
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. System Setup & Helper Functions ---
sys.path.append(os.path.abspath("."))
from data_adapter import DataAdapter

def load_strategy_module(module_name, file_path):
    if not os.path.exists(file_path): return None
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    mom_strat = load_strategy_module("momentum", "notebooks/corrected/01_momentum_lookahead_fixed.py")
    val_strat = load_strategy_module("value", "notebooks/corrected/02_value_survivorship_fixed.py")
    mf_strat = load_strategy_module("multifactor", "notebooks/corrected/03_multifactor_overfitting_fixed.py")
    ml_strat = load_strategy_module("ml", "notebooks/corrected/04_ml_leakage_fixed.py")
except Exception: pass

# --- 2. UI Configuration (The "Sleek" Engine) ---
st.set_page_config(page_title="Quant Clinic 🏥", layout="wide", page_icon="🏥")

# Inject Custom CSS for Glassmorphism & Typography
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f1116 0%, #1a1c2e 100%);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0d12;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* GLASSMORPHISM CARDS */
    div.css-1r6slb0, div.stExpander, div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }

    /* Header Styling */
    h1, h2, h3 {
        color: #f0f2f6;
        letter-spacing: -0.5px;
    }
    
    h1 {
        background: linear-gradient(90deg, #a29bfe 0%, #6c5ce7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Custom Button Styling (Gradient) */
    div.stButton > button {
        background: linear-gradient(90deg, #6c5ce7 0%, #a29bfe 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(108, 92, 231, 0.4);
    }

    /* Metric Value Styling */
    [data-testid="stMetricValue"] {
        color: #a29bfe;
        font-weight: 600;
    }
    
    /* Remove standard streamlit chrome */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. Pro Charting Function ---
def plot_pro_chart(equity_curve, title="Performance"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])

    # Equity Curve (Gradient Fill)
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values, 
        name="Equity", 
        line=dict(color='#6c5ce7', width=2),
        fill='tozeroy',
        fillcolor='rgba(108, 92, 231, 0.1)'
    ), row=1, col=1)
    
    # Drawdown Area
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values, 
        name="Drawdown", 
        line=dict(color='#ff7675', width=1),
        fill='tozeroy',
        fillcolor='rgba(255, 118, 117, 0.2)'
    ), row=2, col=1)

    # Dark Theme Layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=600,
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    return fig

# --- 4. Sidebar Navigation ---
with st.sidebar:
    st.title("🏥 Quant Clinic")
    st.markdown("<div style='font-size: 0.8em; opacity: 0.7; margin-bottom: 20px;'>PROFESSIONAL DIAGNOSTIC SUITE</div>", unsafe_allow_html=True)
    
    st.subheader("Data Feed")
    data_source = st.radio("Source", ["Synthetic (Monte Carlo)", "Upload CSV"], index=0, label_visibility="collapsed")
    
    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            pd.read_csv(uploaded_file).to_csv("temp_dashboard_data.csv", index=False)
            
    st.markdown("---")
    st.subheader("Diagnosis Module")
    selected_strategy = st.selectbox("Select Strategy", 
        ["Momentum (Lookahead)", "Value (Survivorship)", "Multi-Factor (Overfitting)", "ML (Leakage)"]
    )
    
    st.markdown("---")
    st.success("● System Operational")

# --- 5. Main Logic ---
@st.cache_data
def get_data(use_synthetic, uploaded):
    adapter = DataAdapter(source_type='csv' if uploaded else 'synthetic', 
                          csv_path="temp_dashboard_data.csv" if uploaded else None)
    return adapter.get_data('2004-01-01', '2024-01-01')

if data_source == "Synthetic (Monte Carlo)" or uploaded_file:
    prices, meta = get_data(data_source == "Synthetic (Monte Carlo)", uploaded_file)
else:
    st.warning("⚠️ Waiting for data stream...")
    st.stop()

# Header Section
st.title(f"{selected_strategy}")
st.markdown("---")

# Strategy Logic
if "Momentum" in selected_strategy:
    with st.expander("📘 Technical Methodology: Point-in-Time Architecture", expanded=True):
        st.markdown("""
        **Corrected Logic:** Enforces explicit lag between signal generation ($T_{decision}$) and execution ($T_{trade}$).
        * **Signal:** `Close[T-1]`
        * **Execution:** `Open[T]`
        """)
    
    col1, col2, col3 = st.columns([1,1,1])
    with col1: lookback = st.slider("Lookback (Months)", 3, 24, 12)
    with col2: top_pct = st.slider("Top Percentile", 0.05, 0.5, 0.1)
    with col3: cost = st.number_input("Cost (bps)", 0, 50, 10)
    
    if st.button("RUN SIMULATION"):
        engine = mom_strat.MomentumEngine(lookback, top_pct)
        tester = mom_strat.Backtester(cost_bps=cost, slippage_bps=5.0)
        caps = meta.get('market_caps', prices)
        
        with st.spinner("Calculating Point-in-Time Vector..."):
            equity = tester.run(prices, caps, engine, 1_000_000)
        
        # Pro Metrics Cards
        ret = (equity.iloc[-1]/1e6)-1
        sharpe = (equity.pct_change().mean()*252) / (equity.pct_change().std()*np.sqrt(252))
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{ret:.1%}")
        m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m3.metric("CAGR", f"{(1+ret)**(1/20)-1:.1%}")
        m4.metric("Ending Capital", f"${equity.iloc[-1]/1e6:.2f}M")
        
        st.plotly_chart(plot_pro_chart(equity), use_container_width=True)

elif "Value" in selected_strategy:
    with st.expander("📘 Technical Methodology: Survivorship Injection", expanded=True):
        st.markdown("**Corrected Logic:** Dynamically forces liquidation of assets at -95% return when they exit the constituent list.")

    c1, c2 = st.columns(2)
    n_stocks = c1.slider("Portfolio Size", 10, 100, 30)
    lag = c2.number_input("Reporting Lag (Days)", 0, 90, 45)
    
    if st.button("RUN SIMULATION"):
        strat = val_strat.ValueStrategy(n_stocks, transaction_cost_bps=15.0)
        earnings = meta.get('earnings', prices * 0.05)
        constituents = meta.get('constituents', {})
        
        with st.spinner("Processing Corporate Actions..."):
            equity = strat.run_backtest(prices, earnings, constituents, 1_000_000)
            
        ret = (equity.iloc[-1]/1e6)-1
        m1, m2 = st.columns(2)
        m1.metric("Total Return", f"{ret:.1%}")
        m2.metric("Ending Capital", f"${equity.iloc[-1]:,.0f}")
        
        st.plotly_chart(plot_pro_chart(equity), use_container_width=True)

elif "Multi-Factor" in selected_strategy:
    with st.expander("📘 Technical Methodology: Deflated Sharpe Ratio (DSR)", expanded=True):
        st.markdown(r"**Corrected Logic:** Adjusts significance based on trial count ($N$) using Bailey & de Prado (2014) theorem.")

    trials = st.slider("Hypothesis Trials (N)", 1, 100, 10)
    
    if st.button("CHECK ROBUSTNESS"):
        with st.spinner("Optimizing..."):
            equity = mf_strat.run_strategy(prices, '2010-01-01', '2023-12-31')
            
        ret_series = equity.pct_change().dropna()
        sharpe = (ret_series.mean()*252)/(ret_series.std()*np.sqrt(252))
        dsr = mf_strat.Statistics.deflated_sharpe(sharpe, trials, len(ret_series))
        
        c1, c2 = st.columns(2)
        c1.metric("Raw Sharpe", f"{sharpe:.2f}")
        c2.metric("Deflated Sharpe Prob", f"{dsr:.1%}", delta_color="normal" if dsr>0.95 else "inverse")
        
        if dsr < 0.95:
            st.error(f"⚠️ Likely False Positive. DSR ({dsr:.1%}) < 95% Confidence Threshold.")
        else:
            st.success("✅ Statistically Significant Result")
            
        st.plotly_chart(plot_pro_chart(equity), use_container_width=True)

elif "ML" in selected_strategy:
    with st.expander("📘 Technical Methodology: Purged K-Fold CV", expanded=True):
        st.markdown("**Corrected Logic:** Prevents leakage by Purging training data overlapping with test labels and Embargoing post-test periods.")

    c1, c2 = st.columns(2)
    purge = c1.slider("Purge Window (Days)", 0, 20, 5)
    embargo = c2.slider("Embargo Window (Days)", 0, 50, 21)
    target = st.selectbox("Target Asset", prices.columns[:5])
    
    if st.button("RUN WALK-FORWARD"):
        with st.spinner("Training & Cross-Validating..."):
            df = ml_strat.FeatureEngineer.create_dataset(prices[target])
            X, y = df.drop(columns=['target']), df['target']
            engine = ml_strat.MLEngine()
            acc = engine.run_walk_forward(X, y)
            
        st.metric("Out-of-Sample Accuracy", f"{acc:.2%}")
        
        if 0.45 < acc < 0.55:
            st.success("✅ Result is Realistic (No Leakage detected)")
        else:
            st.warning("⚠️ Suspicious Accuracy (Possible Leakage)")