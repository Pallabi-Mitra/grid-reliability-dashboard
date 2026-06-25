# ============================================================
# PAGE: ZONE GENERATORS
# Lets the user pick a zone and see every generator in it,
# ranked by predicted risk. This is the "scan and rank" view,
# used to decide which generator to inspect further on Risk Drivers.
# ============================================================

import streamlit as st
import pandas as pd
from shared import (
    load_css, get_live_weather_predictions, zone_names, categorical_cols
)

# --- PAGE CONFIG ---

load_css("styles.css")

# --- LOAD DATA / MODEL / PREDICTIONS ---
assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_live_weather_predictions()

# --- SIDEBAR: FOOTER ---
st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

# --- PAGE HEADER ---
st.title("🔍 Zone Generators")
st.caption("Select a zone to see every generator ranked by predicted risk")
st.markdown("---")

# --- ZONE SELECTOR ---
# Note: this selectbox's chosen zone is stored in st.session_state
# below so the Risk Drivers page can read it and default to the
# same zone/generator, avoiding the user having to re-pick.
selected_zone = st.selectbox(
    "Select a zone to inspect:",
    options=sorted(zone_summary["operating_region"].tolist()),
    format_func=lambda z: f"Zone {z} · {zone_names.get(z, '')} · {zone_summary[zone_summary['operating_region']==z]['risk_level'].values[0]}"
)
st.session_state["selected_zone"] = selected_zone

# --- SCORE EVERY GENERATOR IN THE SELECTED ZONE ---
zone_generators = latest_df[latest_df["operating_region"] == selected_zone].copy()
zone_gen_encoded = pd.get_dummies(zone_generators, columns=categorical_cols)
for col in model_features:
    if col not in zone_gen_encoded.columns:
        zone_gen_encoded[col] = 0
zone_generators["predicted_impact_ratio"] = model.predict(zone_gen_encoded[model_features])
zone_generators["predicted_impacted_mw"] = zone_generators["predicted_impact_ratio"] * zone_generators["dependable_capacity_mw"]
zone_generators["risk_level"] = zone_generators["predicted_impact_ratio"].apply(
    lambda x: "🔴 HIGH" if x > 0.65 else ("🟡 MODERATE" if x > 0.45 else "🟢 LOW")
)

# --- ZONE-LEVEL METRICS ---
zs = zone_summary[zone_summary["operating_region"] == selected_zone].iloc[0]
zc1, zc2, zc3 = st.columns(3)
zc1.metric("Zone Risk Level", zs["risk_level"])
zc2.metric("Zone MW at Risk", f"{zs['predicted_mw_at_risk']:.1f} MW")
zc3.metric("Generators in Zone", str(int(zs["num_generators"])))

# --- GENERATOR TABLE, RANKED BY RISK ---
display_cols = ["asset_id", "fuel_category", "broad_asset_category",
                "dependable_capacity_mw", "predicted_impact_ratio",
                "predicted_impacted_mw", "risk_level", "recent_avg_impact"]

st.dataframe(
    zone_generators[display_cols].sort_values("predicted_impact_ratio", ascending=False).round(3),
    use_container_width=True
)