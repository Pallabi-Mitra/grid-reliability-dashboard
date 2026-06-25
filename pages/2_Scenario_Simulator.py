# ============================================================
# PAGE: SCENARIO SIMULATOR
# Lets the user override forecast temperature and see predicted
# risk update live across all zones. Uses the SAME trained model,
# just feeds it hypothetical temperature inputs instead of the
# latest real (synthetic) day's weather.
# ============================================================

import streamlit as st
import pandas as pd
from shared import (
    load_css, get_latest_predictions, build_map, categorical_cols
)

# --- PAGE CONFIG ---

load_css("styles.css")

# --- LOAD DATA / MODEL / PREDICTIONS ---
assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_latest_predictions()

# --- SIDEBAR: FOOTER ---
st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)
# --- PAGE HEADER ---
st.title("🌨️ Scenario Simulator")
st.caption("Adjust forecast temperature to see how predicted zone risk changes in real time")
st.markdown("---")

# --- TEMPERATURE OVERRIDE SLIDER ---
temp_override = st.slider("Forecast Temperature (°F)", min_value=-10, max_value=100, value=45, step=1)

# --- REBUILD FEATURES WITH OVERRIDDEN TEMPERATURE ---
# Copies the latest real data, but swaps temp-related columns for the
# slider value, then re-runs the SAME model on this hypothetical input.
sim_df = latest_df.copy()
sim_df["temp_avg"] = temp_override
sim_df["temp_min"] = temp_override - 8
sim_df["temp_max"] = temp_override + 8
sim_df["temp_range"] = 16
sim_df["cold_day_flag"] = int(temp_override < 20)
sim_df["hot_day_flag"] = int(temp_override > 85)

sim_encoded = pd.get_dummies(sim_df, columns=categorical_cols)
for col in model_features:
    if col not in sim_encoded.columns:
        sim_encoded[col] = 0
X_sim = sim_encoded[model_features]
sim_df["predicted_impact_ratio"] = model.predict(X_sim)
sim_df["predicted_impacted_mw"] = sim_df["predicted_impact_ratio"] * sim_df["dependable_capacity_mw"]

# --- SIMULATED ZONE SUMMARY ---
sim_zone_summary = sim_df.groupby("operating_region").agg(
    total_capacity_mw=("dependable_capacity_mw", "sum"),
    predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
    num_generators=("asset_id", "count")
).reset_index()
sim_zone_summary["risk_pct"] = (sim_zone_summary["predicted_mw_at_risk"] / sim_zone_summary["total_capacity_mw"]) * 100
sim_zone_summary["risk_level"] = sim_zone_summary["risk_pct"].apply(
    lambda pct: "🔴 RED" if pct > 65 else ("🟡 YELLOW" if pct > 45 else "🟢 GREEN")
)

# --- METRICS WITH DELTA VS REAL (LATEST) DATA ---
s1, s2, s3 = st.columns(3)
s1.metric(
    "Avg Impact Ratio",
    f"{sim_df['predicted_impact_ratio'].mean():.3f}",
    delta=f"{sim_df['predicted_impact_ratio'].mean() - latest_df['predicted_impact_ratio'].mean():.3f}"
)
s2.metric(
    "Total MW at Risk",
    f"{sim_df['predicted_impacted_mw'].sum():.0f} MW",
    delta=f"{sim_df['predicted_impacted_mw'].sum() - latest_df['predicted_impacted_mw'].sum():.0f} MW"
)
s3.metric(
    "Zones at RED",
    f"{(sim_zone_summary['risk_level'] == '🔴 RED').sum()} / 11",
    delta=f"{(sim_zone_summary['risk_level'] == '🔴 RED').sum() - (zone_summary['risk_level'] == '🔴 RED').sum()}"
)

# --- SIMULATED MAP ---
# Reuses build_map() from shared.py, same function as Overview,
# just fed the simulated summary instead of the live one.
st.plotly_chart(build_map(sim_zone_summary, f"Simulated Zone Risk at {temp_override}°F"), width='stretch')
