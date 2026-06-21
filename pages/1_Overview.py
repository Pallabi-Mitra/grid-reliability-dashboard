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
st.caption("Real-time risk visibility across New York's grid")
st.markdown("---")

# --- LIVE CONDITIONS: NY-WIDE WEATHER + ALERTS ---
from shared import get_all_zone_weather, get_ny_weather_alerts
import pandas as pd

alerts = get_ny_weather_alerts()
zone_weather = get_all_zone_weather()

# Alerts banner, only shows if something is actually active
if alerts:
    st.error(f"⚠️ {len(alerts)} active NOAA weather alert(s) for New York")
    for a in alerts[:5]:  # cap display at 5 to avoid overwhelming the page
        st.markdown(f"**{a['event']}** · {a['area']}")
        st.caption(a['headline'])
else:
    st.success("No active NOAA weather alerts for New York")

st.markdown("---")

# Per-zone weather table
st.subheader("🌐 Live Conditions by Zone")
if zone_weather:
    weather_df = pd.DataFrame(zone_weather)[["zone", "location", "temperature", "unit", "short_forecast"]]
    weather_df.columns = ["Zone", "Location", "Temp", "Unit", "Conditions"]
    st.dataframe(weather_df, use_container_width=True, hide_index=True)
else:
    st.info("Live weather data unavailable right now")

st.markdown("---")
# --- LIVE NY ELECTRICITY DEMAND (EIA, real public data) ---
from shared import get_ny_live_demand


eia_key_present = bool(os.environ.get("EIA_API_KEY"))


demand = get_ny_live_demand()


if demand:
    st.info(f"⚡ Actual NY grid demand right now: **{demand['demand_mw']:,.0f} MW** (EIA public data, {demand['period']})")
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