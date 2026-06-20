# ============================================================
# PAGE: OVERVIEW
# Landing page. Shows live weather context, top-level statewide
# metrics, and the NY zone risk map.
# ============================================================

import streamlit as st
import plotly.graph_objects as go
from shared import (
    load_css, get_latest_predictions, build_map, get_live_weather
)

# --- LOAD CSS ---
load_css("styles.css")

# --- LOAD DATA / MODEL / PREDICTIONS ---
assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_latest_predictions()

# --- SIDEBAR: FOOTER ONLY ---
# Identity header now lives once in dashboard.py, rendered before
# the nav. This page only adds the footer disclaimer at the bottom.
st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

# --- PAGE HEADER ---
st.title("💡 Grid Reliability Dashboard")
st.caption("Predicting generator risk before it becomes an outage")
st.markdown("---")

# --- LIVE CONDITIONS WIDGET ---
weather = get_live_weather(42.6526, -73.7562, "Albany, NY")

with st.container(border=True):
    wc1, wc2 = st.columns([3, 1])
    with wc1:
        if weather:
            st.markdown(
                f"**🌐 Live Conditions** · {weather['location']}: "
                f"{weather['temperature']}°{weather['unit']}, {weather['short_forecast']}"
            )
        else:
            st.markdown("**🌐 Live Conditions** · unavailable right now")
    with wc2:
        if weather:
            st.caption(f"Updated {weather['fetched_at']}")

st.markdown("---")

# --- TOP METRICS ROW ---
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Capacity", f"{zone_summary['total_capacity_mw'].sum():.0f} MW")
m2.metric("Total MW at Risk", f"{zone_summary['predicted_mw_at_risk'].sum():.0f} MW")
m3.metric("Zones at RED", f"{(zone_summary['risk_level'] == '🔴 RED').sum()} / 11")
avg_risk_pct = latest_df["predicted_impact_ratio"].mean() * 100
m4.metric("Avg Risk Level", f"{avg_risk_pct:.0f}%")

# --- ZONE RISK MAP ---
st.markdown("---")
st.subheader("NY Zone Risk Map")
st.plotly_chart(build_map(zone_summary, f"Zone Risk · {latest_date}"), use_container_width=True)