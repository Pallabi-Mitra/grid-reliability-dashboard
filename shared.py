# ============================================================
# SHARED DATA, MODEL, AND UTILITIES
# Imported by every page in pages/ so data and model load once,
# consistently, instead of each page repeating setup code.
#
# LOAD ORDER (called by pages in this sequence):
#   load_data() -> load_model() -> get_latest_predictions()
#   OR
#   load_data() -> load_model() -> get_all_zone_weather()
#                               -> get_live_weather_predictions()
#   Overview also calls: get_ny_weather_alerts(), get_ny_live_demand()
#   Overview forecast calls: load_forecaster_models() -> get_zone_forecast()
# ============================================================

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from xgboost import XGBRegressor
import plotly.graph_objects as go
import os


# ------------------------------------------------------------
# CSS LOADER
# ------------------------------------------------------------
def load_css(filepath):
    with open(filepath) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------
# DATA AND MODEL LOADING
# Cached so CSVs and model.json load once per session.
# ------------------------------------------------------------
@st.cache_data
def load_data():
    assets = pd.read_csv("assets.csv")
    daily = pd.read_csv("daily_records.csv")
    df = daily.merge(assets, on="asset_id", how="left")
    return assets, daily, df


@st.cache_resource
def load_model():
    from xgboost import Booster
    booster = Booster()
    booster.load_model("model.json")
    model = XGBRegressor()
    model._Booster = booster
    model._estimator_type = "regressor"
    return model


# ------------------------------------------------------------
# STATIC LOOKUPS
# ------------------------------------------------------------
categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]


def get_risk_color(pct):
    if pct > 65:
        return "🔴 RED"
    elif pct > 45:
        return "🟡 YELLOW"
    else:
        return "🟢 GREEN"


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
# PREDICTIONS (synthetic data, fixed date from CSV)
# Used by all pages except Overview.
# ------------------------------------------------------------
def get_latest_predictions():
    assets, daily, df = load_data()
    model = load_model()
    model_features = model.get_booster().feature_names

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()
    latest_encoded = pd.get_dummies(latest_df, columns=categorical_cols)
    for col in model_features:
        if col not in latest_encoded.columns:
            latest_encoded[col] = 0

    predicted_ratio = model.predict(latest_encoded[model_features])
    latest_df["predicted_impact_ratio"] = predicted_ratio
    latest_df["predicted_impacted_mw"] = predicted_ratio * latest_df["dependable_capacity_mw"]

    zone_summary = latest_df.groupby("operating_region").agg(
        total_capacity_mw=("dependable_capacity_mw", "sum"),
        predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
        num_generators=("asset_id", "count")
    ).reset_index()
    zone_summary["risk_pct"] = (
        zone_summary["predicted_mw_at_risk"] / zone_summary["total_capacity_mw"]
    ) * 100
    zone_summary["risk_level"] = zone_summary["risk_pct"].apply(get_risk_color)

    return assets, daily, df, model, model_features, latest_date, latest_df, zone_summary


# ------------------------------------------------------------
# ZONE MAP BUILDER
# Used by Overview and Scenario Simulator.
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
            hovertext=(
                f"Zone {zone} ({zone_names.get(zone, '')})<br>"
                f"Risk Level: {row['risk_level']}<br>"
                f"Risk %: {row['risk_pct']:.1f}%<br>"
                f"MW at Risk: {row['predicted_mw_at_risk']:.1f}<br>"
                f"Generators: {row['num_generators']}"
            ),
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
# Real public data. Used by Overview for ambient context and
# by get_live_weather_predictions() to drive model inputs.
# ------------------------------------------------------------
NWS_USER_AGENT = "GridReliabilityDashboard, student-project@example.com"


@st.cache_data(ttl=600)
def get_live_weather(lat, lon, location_label):
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


@st.cache_data(ttl=600)
def get_all_zone_weather():
    results = []
    for zone, (lat, lon) in zone_coords.items():
        label = f"Zone {zone} ({zone_names.get(zone, '')})"
        weather = get_live_weather(lat, lon, label)
        if weather:
            weather["zone"] = zone
            results.append(weather)
    return results


@st.cache_data(ttl=300)
def get_ny_weather_alerts():
    headers = {"User-Agent": NWS_USER_AGENT}
    try:
        resp = requests.get(
            "https://api.weather.gov/alerts/active?area=NY",
            headers=headers, timeout=5
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        alerts = []
        for f in features:
            props = f.get("properties", {})
            alerts.append({
                "event": props.get("event", "Alert"),
                "headline": props.get("headline", ""),
                "severity": props.get("severity", "Unknown"),
                "area": props.get("areaDesc", "")
            })
        return alerts
    except Exception:
        return []


@st.cache_data(ttl=900)
def get_ny_live_demand():
    api_key = os.environ.get("EIA_API_KEY")
    if not api_key:
        return None
    try:
        url = (
            "https://api.eia.gov/v2/electricity/rto/region-data/data/"
            f"?api_key={api_key}"
            "&frequency=hourly"
            "&data[0]=value"
            "&facets[respondent][]=NYIS"
            "&facets[type][]=D"
            "&sort[0][column]=period"
            "&sort[0][direction]=desc"
            "&length=1"
        )
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        rows = resp.json().get("response", {}).get("data", [])
        if not rows:
            return None
        latest = rows[0]
        return {
            "demand_mw": float(latest["value"]),
            "period": latest["period"]
        }
    except Exception:
        return None


# ------------------------------------------------------------
# LIVE-WEATHER-DRIVEN PREDICTIONS
# Overview uses this instead of get_latest_predictions().
# Substitutes real NOAA temperatures into the model's weather
# features, then re-scores. Generator capacity, fuel type, and
# maintenance history stay from the synthetic dataset.
# ------------------------------------------------------------
@st.cache_data(ttl=600)
def get_live_weather_predictions():
    assets, daily, df = load_data()
    model = load_model()
    model_features = model.get_booster().feature_names

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()

    zone_weather_list = get_all_zone_weather()
    zone_temp_map = {}
    for w in zone_weather_list:
        zone_temp_map[w["zone"]] = w["temperature"]

    for zone, temp_f in zone_temp_map.items():
        mask = latest_df["operating_region"] == zone
        latest_df.loc[mask, "temp_avg"] = temp_f
        latest_df.loc[mask, "temp_min"] = temp_f - 8
        latest_df.loc[mask, "temp_max"] = temp_f + 8
        latest_df.loc[mask, "temp_range"] = 16
        latest_df.loc[mask, "cold_day_flag"] = int(temp_f < 20)
        latest_df.loc[mask, "hot_day_flag"] = int(temp_f > 85)

    latest_encoded = pd.get_dummies(latest_df, columns=categorical_cols)
    for col in model_features:
        if col not in latest_encoded.columns:
            latest_encoded[col] = 0

    latest_df["predicted_impact_ratio"] = model.predict(latest_encoded[model_features])
    latest_df["predicted_impacted_mw"] = (
        latest_df["predicted_impact_ratio"] * latest_df["dependable_capacity_mw"]
    )

    zone_summary = latest_df.groupby("operating_region").agg(
        total_capacity_mw=("dependable_capacity_mw", "sum"),
        predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
        num_generators=("asset_id", "count")
    ).reset_index()
    zone_summary["risk_pct"] = (
        zone_summary["predicted_mw_at_risk"] / zone_summary["total_capacity_mw"]
    ) * 100
    zone_summary["risk_level"] = zone_summary["risk_pct"].apply(get_risk_color)

    today = datetime.now().strftime("%Y-%m-%d")
    return assets, daily, df, model, model_features, today, latest_df, zone_summary


# ------------------------------------------------------------
# FORECASTER MODELS (quantile XGBoost)
# Three models predicting p05/p50/p95 of impact_ratio.
# Trained offline in Jupyter, saved as forecaster_p*.json.
# Loaded once per session, cached via st.cache_resource.
# ------------------------------------------------------------
@st.cache_resource
def load_forecaster_models():
    from xgboost import Booster
    models = {}
    for name in ["p05", "p50", "p95"]:
        booster = Booster()
        booster.load_model(f"forecaster_{name}.json")
        m = XGBRegressor()
        m._Booster = booster
        m._estimator_type = "regressor"
        models[name] = m
    return models


@st.cache_data(ttl=600)
def get_zone_forecast(zone: str, days: int = 7):
    """
    Generates a 7-day capacity-weighted risk forecast for a zone.
    Simulates time passing by incrementing days_since_last_event
    forward by 1 each day. All other features held at today's values.
    Returns a DataFrame with day, date, p05, p50, p95 columns.
    """
    assets, daily, df = load_data()
    models = load_forecaster_models()

    model_features = models["p50"].get_booster().feature_names
    latest_date = df["date"].max()
    zone_df = df[df["date"] == latest_date].copy()
    zone_df = zone_df[zone_df["operating_region"] == zone].copy()

    if zone_df.empty:
        return None

    forecast_rows = []
    base_date = datetime.now()

    for day_offset in range(1, days + 1):
        forecast_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        day_df = zone_df.copy()

        if "days_since_last_event" in day_df.columns:
            day_df["days_since_last_event"] = day_df["days_since_last_event"] + day_offset

        day_enc = pd.get_dummies(day_df, columns=categorical_cols)
        for col in model_features:
            if col not in day_enc.columns:
                day_enc[col] = 0

        X = day_enc[model_features]
        capacities = day_df["dependable_capacity_mw"].values
        total_capacity = capacities.sum()

        p05_avg = float((models["p05"].predict(X) * capacities).sum() / total_capacity)
        p50_avg = float((models["p50"].predict(X) * capacities).sum() / total_capacity)
        p95_avg = float((models["p95"].predict(X) * capacities).sum() / total_capacity)

        forecast_rows.append({
            "day": f"Day +{day_offset}",
            "date": forecast_date,
            "p05": round(p05_avg, 3),
            "p50": round(p50_avg, 3),
            "p95": round(p95_avg, 3),
        })

    return pd.DataFrame(forecast_rows)