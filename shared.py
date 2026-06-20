# ============================================================
# SHARED DATA, MODEL, AND UTILITIES
# Imported by every page in pages/ so data and model load once,
# consistently, instead of each page repeating setup code.
# ============================================================

import streamlit as st
import pandas as pd
import requests
from datetime import datetime
from xgboost import XGBRegressor
import plotly.graph_objects as go


# ------------------------------------------------------------
# CSS LOADER
# Loads styles.css into the page. Called once per page, near the top.
# ------------------------------------------------------------
def load_css(filepath):
    with open(filepath) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# DATA AND MODEL LOADING
# @st.cache_data / @st.cache_resource mean these only run once
# per session, even though multiple pages call them.
# ------------------------------------------------------------
@st.cache_data
def load_data():
    assets = pd.read_csv("assets.csv")
    daily = pd.read_csv("daily_records.csv")
    df = daily.merge(assets, on="asset_id", how="left")
    return assets, daily, df

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("model.json")
    return model


# ------------------------------------------------------------
# STATIC LOOKUPS
# Zone risk thresholds, colors, coordinates, and display names.
# Kept in one place so every page colors/labels zones identically.
# ------------------------------------------------------------
categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]

def get_risk_color(pct):
    if pct > 65: return "🔴 RED"
    elif pct > 45: return "🟡 YELLOW"
    else: return "🟢 GREEN"

color_map = {"🔴 RED": "#DC2626", "🟡 YELLOW": "#D97706", "🟢 GREEN": "#16A34A"}

zone_coords = {
    "A": (42.9, -78.9), "B": (43.15, -77.6), "C": (43.05, -76.15),
    "D": (44.0, -75.5), "E": (43.1, -75.2), "F": (42.65, -73.75),
    "G": (41.7, -73.9), "H": (41.2, -73.8), "I": (40.95, -73.85),
    "J": (40.71, -74.0), "K": (40.8, -73.1)
}

zone_names = {
    "A": "West", "B": "Genesee", "C": "Central", "D": "North",
    "E": "Mohawk Valley", "F": "Capital", "G": "Hudson Valley",
    "H": "Millwood", "I": "Dunwoodie", "J": "NYC", "K": "Long Island"
}


# ------------------------------------------------------------
# PREDICTIONS
# Runs the model on the latest date in the dataset and builds
# the zone-level summary table used across every page.
# ------------------------------------------------------------
def get_latest_predictions():
    """Loads data/model, scores the latest date, returns everything pages need."""
    assets, daily, df = load_data()
    model = load_model()
    model_features = model.get_booster().feature_names

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()
    latest_encoded = pd.get_dummies(latest_df, columns=categorical_cols)
    for col in model_features:
        if col not in latest_encoded.columns:
            latest_encoded[col] = 0
    X_latest = latest_encoded[model_features]
    predicted_ratio = model.predict(X_latest)
    latest_df["predicted_impact_ratio"] = predicted_ratio
    latest_df["predicted_impacted_mw"] = predicted_ratio * latest_df["dependable_capacity_mw"]

    zone_summary = latest_df.groupby("operating_region").agg(
        total_capacity_mw=("dependable_capacity_mw", "sum"),
        predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
        num_generators=("asset_id", "count")
    ).reset_index()
    zone_summary["risk_pct"] = (zone_summary["predicted_mw_at_risk"] / zone_summary["total_capacity_mw"]) * 100
    zone_summary["risk_level"] = zone_summary["risk_pct"].apply(get_risk_color)

    return assets, daily, df, model, model_features, latest_date, latest_df, zone_summary


# ------------------------------------------------------------
# ZONE MAP BUILDER
# Shared by Overview (live data) and Scenario Simulator (simulated data).
# Written once here so both pages stay visually identical.
# ------------------------------------------------------------
def build_map(summary, title):
    fig = go.Figure()
    for _, row in summary.iterrows():
        zone = row["operating_region"]
        if zone not in zone_coords:
            continue
        lat, lon = zone_coords[zone]
        color = color_map[row["risk_level"]]
        fig.add_trace(go.Scattergeo(
            lat=[lat], lon=[lon],
            mode="markers+text",
            marker=dict(size=35, color=color, line=dict(width=2, color="white")),
            text=[zone],
            textfont=dict(size=14, color="white", family="Arial Black"),
            hovertext=f"Zone {zone} ({zone_names.get(zone,'')})<br>Risk Level: {row['risk_level']}<br>Risk %: {row['risk_pct']:.1f}%<br>MW at Risk: {row['predicted_mw_at_risk']:.1f}<br>Generators: {row['num_generators']}",
            hoverinfo="text",
            showlegend=False
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#1A2332", size=16)),
        geo=dict(
            scope="usa",
            projection=dict(type="albers usa"),
            center=dict(lat=42.8, lon=-75.5),
            projection_scale=12,
            showland=True,
            landcolor="#EDEFF3",
            showsubunits=True,
            subunitcolor="#C5CAD4",
            bgcolor="#F7F8FA"
        ),
        height=500,
        paper_bgcolor="#F7F8FA",
        font=dict(color="#1A2332", family="IBM Plex Sans"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ------------------------------------------------------------
# LIVE WEATHER (NOAA api.weather.gov)
# REAL, live, public data. Shown only as ambient "Live Conditions"
# context on Overview. Never feeds into the model or predictions,
# kept clearly separate per the project's synthetic-data framing.
# ------------------------------------------------------------
NWS_USER_AGENT = "GridReliabilityDashboard, student-project@example.com"

@st.cache_data(ttl=600)  # cache 10 minutes, NWS doesn't update faster than that
def get_live_weather(lat, lon, location_label):
    """
    Two-step NWS API call:
    1. /points/{lat},{lon} -> resolves to a forecast URL for that location
    2. forecast URL -> actual current/near-term conditions
    Returns None on any failure so the page can show a quiet fallback
    instead of crashing if NWS is down or unreachable.
    """
    headers = {"User-Agent": NWS_USER_AGENT}
    try:
        points_resp = requests.get(
            f"https://api.weather.gov/points/{lat},{lon}",
            headers=headers, timeout=5
        )
        points_resp.raise_for_status()
        forecast_url = points_resp.json()["properties"]["forecast"]

        forecast_resp = requests.get(forecast_url, headers=headers, timeout=5)
        forecast_resp.raise_for_status()
        current_period = forecast_resp.json()["properties"]["periods"][0]

        return {
            "location": location_label,
            "temperature": current_period["temperature"],
            "unit": current_period["temperatureUnit"],
            "short_forecast": current_period["shortForecast"],
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
    except Exception:
        return None