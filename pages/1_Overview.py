# ============================================================
# PAGE: OVERVIEW
# Landing page. Dark header bar, alert banner, metric cards,
# live weather/alerts/EIA demand, zone risk map.
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime
from shared import (
    load_css, get_live_weather_predictions, build_map,
    get_all_zone_weather, get_ny_weather_alerts, get_ny_live_demand
)

load_css("styles.css")

assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_live_weather_predictions()

st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data · no real NYISO operational data used</div>",
    unsafe_allow_html=True
)

# --- PAGE HEADER BAR ---
now = datetime.now().strftime("%b %d, %Y · %H:%M EST")
st.markdown(f"""
<div class="page-header">
    <div>
        <p class="page-header-title">Grid Reliability Intelligence Platform</p>
        <p class="page-header-subtitle">New York load zones · synthetic demo data</p>
    </div>
    <div class="live-indicator">
        <div class="live-dot"></div>
        <span class="live-text">Live · {now}</span>
    </div>
</div>
""", unsafe_allow_html=True)
# --- CONTENT AREA ---
st.markdown('<div class="content-area">', unsafe_allow_html=True)

# --- ALERTS ---
alerts = get_ny_weather_alerts()
if alerts:
    alert_lines = "".join([
        f"<strong>{a['event']}</strong> · {a['area']}<br>"
        for a in alerts[:3]
    ])
    st.markdown(f"""
    <div class="alert-banner">
        <p class="alert-banner-text">
            ⚠ {len(alerts)} active NOAA weather alert(s) for New York<br>
            {alert_lines}
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- METRICS ---
demand = get_ny_live_demand()
demand_val = f"{demand['demand_mw']:,.0f} MW" if demand else "unavailable"
demand_sub = "EIA live" if demand else "EIA unavailable"

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card">
        <p class="metric-label">Total capacity</p>
        <p class="metric-value">{zone_summary['total_capacity_mw'].sum():,.0f} MW</p>
    </div>
    <div class="metric-card">
        <p class="metric-label">MW at risk</p>
        <p class="metric-value" style="color:#DC2626;">{zone_summary['predicted_mw_at_risk'].sum():,.0f} MW</p>
    </div>
    <div class="metric-card">
        <p class="metric-label">Zones at RED</p>
        <p class="metric-value">{(zone_summary['risk_level'] == '🔴 RED').sum()} <span style="font-size:0.9rem;color:#64748B;font-weight:400;">/ 11</span></p>
    </div>
    <div class="metric-card">
        <p class="metric-label">NY grid demand</p>
        <p class="metric-value">{demand_val}</p>
        <p class="metric-sub">{demand_sub}</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- LIVE CONDITIONS TABLE ---
zone_weather = get_all_zone_weather()
if zone_weather:
    weather_df = pd.DataFrame(zone_weather)[["zone", "location", "temperature", "short_forecast"]]
    weather_df.columns = ["Zone", "Location", "Temp (°F)", "Conditions"]
    st.subheader("Live Conditions by Zone")
    st.dataframe(weather_df, use_container_width=True, hide_index=True)

st.markdown("---")

# --- ZONE RISK MAP ---
st.subheader("NY Zone Risk Map")
st.plotly_chart(build_map(zone_summary, f"Zone risk · {latest_date}"), use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("""
<div class="page-footer">
    <p class="page-footer-text">Synthetic data · no real NYISO operational data used</p>
    <p class="page-footer-text">Grid Reliability Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)