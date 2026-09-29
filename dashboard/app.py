import streamlit as st
import pandas as pd
import joblib
import sys
import os
import matplotlib.pyplot as plt

# Connect the src folder to import the Triage Agent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from agent import TriageAgent

st.set_page_config(layout="wide", page_title="Predictive Maintenance Dashboard")
st.title("Self-Healing Fleet: Live Triage & Monitoring")

@st.cache_resource
def load_assets():
    # Load the model generated in Step 3 and the processed dataset
    model = joblib.load("models/xgb_rul_model.pkl")
    df = pd.read_csv("data/processed/train_FD001_processed.csv")
    return model, df

model, df = load_assets()
agent = TriageAgent()

# Sidebar UI
st.sidebar.header("Fleet Telemetry Controls")
engine_list = sorted(df["unit_number"].unique())
engine_id = st.sidebar.selectbox("Select Engine Unit to Monitor", engine_list)

# Filter dataset to simulate streaming for the selected engine
engine_data = df[df["unit_number"] == engine_id].copy()

# Feature extraction for model inference
exclude_cols = ["unit_number", "time_cycles", "RUL"]
feature_cols = [c for c in engine_data.columns if c not in exclude_cols]
X_engine = engine_data[feature_cols]

# Run model predictions
engine_data["Predicted_RUL"] = model.predict(X_engine)
current_cycle = engine_data["time_cycles"].max()
latest_rul = engine_data["Predicted_RUL"].iloc[-1]

# Top level metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current Operating Cycle", int(current_cycle))
col2.metric("Latest Predicted RUL", f"{latest_rul:.1f} cycles")

# Pass model output into the Agent logic
work_order = agent.evaluate_rul(engine_id, current_cycle, latest_rul)

if work_order and work_order["severity"] == "CRITICAL":
    col3.error("Status: CRITICAL - FAULT IMMINENT")
elif work_order and work_order["severity"] == "WARNING":
    col3.warning("Status: WARNING - DEGRADATION DETECTED")
else:
    col3.success("Status: HEALTHY")

# Middle Row: RUL Degradation Chart
st.subheader(f"Engine {int(engine_id)} Degradation Curve")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(engine_data["time_cycles"], engine_data["RUL"], label="True RUL", linestyle="--", color="gray", alpha=0.7)
ax.plot(engine_data["time_cycles"], engine_data["Predicted_RUL"], label="Predicted RUL", color="dodgerblue", linewidth=2)
ax.axhline(30, color="orange", linestyle=":", label="Agent Warning Threshold")
ax.axhline(15, color="red", linestyle=":", label="Agent Critical Threshold")
ax.set_xlabel("Time Cycles")
ax.set_ylabel("Remaining Cycles (RUL)")
ax.legend()
st.pyplot(fig)

# Bottom Row: Agent Work Orders
st.subheader("Automated Triage System")
if work_order:
    st.write("⚠️ **Agent has intercepted a low RUL prediction and automatically generated the following work order:**")
    st.json(work_order)
else:
    st.info("No active work orders. Agent confirms machine is operating safely.")