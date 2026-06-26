# ============================================================
# PAGE: OVERVIEW
# ============================================================

import streamlit as st
import pandas as pd
from datetime import datetime
from shared import (
    load_css, get_live_weather_predictions, build_map,
    get_all_zone_weather, get_ny_weather_alerts, get_ny_live_demand
)
from shared import (
    load_css, get_live_weather_predictions, build_map,
    get_all_zone_weather, get_ny_weather_alerts, get_ny_live_demand,
    zone_names
)

load_css("styles.css")

assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_live_weather_predictions()

# --- SIDEBAR ---


# --- PAGE HEADER ---
now = datetime.now().strftime("%b %d, %Y · %H:%M EST")
st.markdown(f"""
<div class="page-header">
    <div>
        <p class="page-header-title">Grid Reliability Intelligence Platform</p>
        <p class="page-header-subtitle">New York load zones</p>
    </div>
    <div class="live-indicator">
        <div class="live-dot"></div>
        <span class="live-text">Live · {now}</span>
    </div>
</div>
<div class="content-area">
""", unsafe_allow_html=True)

# --- ALERTS ---
alerts = get_ny_weather_alerts()
if alerts:
    alert_lines = " · ".join([f"<strong>{a['event']}</strong> · {a['area']}" for a in alerts[:2]])
    st.markdown(f"""
    <div class="alert-banner">
        <p class="alert-banner-text">⚠ {len(alerts)} active NOAA weather alert(s) for New York &nbsp;·&nbsp; {alert_lines}</p>
    </div>
    """, unsafe_allow_html=True)

# --- METRIC CARDS ---
demand = get_ny_live_demand()
demand_val = f"{demand['demand_mw']:,.0f} MW" if demand else "unavailable"
demand_sub = "EIA live" if demand else ""
avg_risk = latest_df["predicted_impact_ratio"].mean() * 100

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
        <p class="metric-value">{(zone_summary['risk_level'] == '🔴 RED').sum()} <span style="font-size:0.85rem;color:#94A3B8;font-weight:400;">/ 11</span></p>
    </div>
    <div class="metric-card">
        <p class="metric-label">Avg risk level</p>
        <p class="metric-value">{avg_risk:.0f}%</p>
    </div>
    <div class="metric-card">
        <p class="metric-label">NY grid demand</p>
        <p class="metric-value">{demand_val}</p>
        <p class="metric-sub">{demand_sub}</p>
    </div>
</div>
""", unsafe_allow_html=True)

# --- TWO COLUMN: WEATHER TABLE + MAP ---
col_left, col_right = st.columns([1, 1.8])

with col_left:
    zone_weather = get_all_zone_weather()
    st.markdown("""
    <p style="font-size:0.72rem;font-weight:600;color:#64748B;text-transform:uppercase;
    letter-spacing:0.07em;margin:0 0 8px;">Live conditions by zone</p>
    """, unsafe_allow_html=True)
    if zone_weather:
        weather_df = pd.DataFrame(zone_weather)[["zone", "location", "temperature", "short_forecast"]]
        weather_df.columns = ["Zone", "Location", "°F", "Conditions"]
        st.dataframe(weather_df, width='stretch', hide_index=True, height=370)
    else:
        st.info("Weather data unavailable")

with col_right:
    st.markdown("""
    <p style="font-size:0.72rem;font-weight:600;color:#64748B;text-transform:uppercase;
    letter-spacing:0.07em;margin:0 0 8px;">NY zone risk map</p>
    """, unsafe_allow_html=True)
    st.plotly_chart(
        build_map(zone_summary, f"Zone risk · {latest_date}"),
        width='stretch'
    )

# --- CLOSE CONTENT DIV ---
st.markdown('</div>', unsafe_allow_html=True)
# --- 7-DAY ZONE RISK FORECAST ---
from shared import get_zone_forecast, load_forecaster_models
import plotly.graph_objects as go

st.markdown("---")
st.markdown("""
<p style="font-size:0.72rem;font-weight:600;color:#64748B;text-transform:uppercase;
letter-spacing:0.07em;margin:0 0 12px;">7-day zone risk forecast</p>
""", unsafe_allow_html=True)

forecast_zone = st.selectbox(
    "Select zone to forecast:",
    options=sorted(zone_summary["operating_region"].tolist()),
    format_func=lambda z: f"Zone {z} · {zone_names.get(z, '')}",
    key="forecast_zone_selector"
)

if st.button("Run 7-Day Forecast", key="run_forecast"):
    with st.spinner("Running quantile forecast..."):
        forecast_df = get_zone_forecast(forecast_zone)

    if forecast_df is not None:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["p95"],
            mode="lines",
            name="High estimate (p95)",
            line=dict(color="#DC2626", dash="dot", width=1.5),
            fill=None
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["p50"],
            mode="lines+markers",
            name="Typical estimate (p50)",
            line=dict(color="#1A2332", width=2),
            fill="tonexty",
            fillcolor="rgba(220,38,38,0.08)"
        ))
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["p05"],
            mode="lines",
            name="Low estimate (p05)",
            line=dict(color="#16A34A", dash="dot", width=1.5),
            fill="tonexty",
            fillcolor="rgba(22,163,74,0.08)"
        ))

        fig.update_layout(
            height=320,
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            margin=dict(l=0, r=0, t=20, b=0),
            yaxis=dict(
                title="Zone risk % (capacity-weighted)",
                range=[0, 1],
                gridcolor="#E2E8F0"
            ),
            xaxis=dict(gridcolor="#E2E8F0"),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            font=dict(color="#1A2332", size=12)
        )

        st.plotly_chart(fig, width='stretch')

        st.markdown("""
        <p style="font-size:0.72rem;font-weight:600;color:#64748B;text-transform:uppercase;
        letter-spacing:0.07em;margin:12px 0 8px;">Daily forecast values</p>
        """, unsafe_allow_html=True)
        display_df = forecast_df.rename(columns={
            "day": "Day", "date": "Date",
            "p05": "Low (p05)", "p50": "Typical (p50)", "p95": "High (p95)"
        })
        st.dataframe(display_df, width='stretch', hide_index=True)

        st.caption(
            "Forecast uses quantile XGBoost (p05/p50/p95). "
            "Simulates time passing by incrementing days_since_last_event forward. "
            "Other features held at today's values. Synthetic data only."
        )
    else:
        st.warning(f"No forecast data available for Zone {forecast_zone}.")
# --- FOOTER ---
st.markdown("""
<div class="page-footer">
    <p class="page-footer-text">Synthetic data · no real operational data used</p>
    <p class="page-footer-text">Grid Reliability Intelligence Platform</p>
</div>
""", unsafe_allow_html=True)
