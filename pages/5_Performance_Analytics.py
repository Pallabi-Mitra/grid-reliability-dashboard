# ============================================================
# PAGE: PERFORMANCE ANALYTICS
# Visual analytics. Each chart leads with a computed finding
# statement backed by a real number from the data.
# Now uses live-weather-driven predictions (same as Overview).
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from shared import (
    load_css, get_live_weather_predictions, categorical_cols, zone_names
)

load_css("styles.css")

assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_live_weather_predictions()

st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

st.title("Performance Analytics")
st.caption("What's driving risk, and where to focus")
st.markdown("---")

latest_encoded = pd.get_dummies(latest_df, columns=categorical_cols)
for col in model_features:
    if col not in latest_encoded.columns:
        latest_encoded[col] = 0
latest_df = latest_df.copy()
latest_df["predicted_impact_ratio"] = model.predict(latest_encoded[model_features])
latest_df["predicted_mw_at_risk"] = latest_df["predicted_impact_ratio"] * latest_df["dependable_capacity_mw"]


# ============================================================
# CHART 1: AVG MW AT RISK BY FUEL TYPE
# ============================================================
st.subheader("Which Fuel Type Carries the Most Risk?")

fuel_summary = latest_df.groupby("fuel_category").agg(
    avg_mw_at_risk=("predicted_mw_at_risk", "mean"),
    total_mw_at_risk=("predicted_mw_at_risk", "sum"),
    generator_count=("asset_id", "count")
).reset_index().sort_values("avg_mw_at_risk", ascending=False)

top_fuel = fuel_summary.iloc[0]
st.markdown(
    f"**Finding:** `{top_fuel['fuel_category']}` generators carry the most risk on average, "
    f"**{top_fuel['avg_mw_at_risk']:.1f} MW at risk per generator**, across {int(top_fuel['generator_count'])} units."
)

fig1 = px.bar(
    fuel_summary, x="fuel_category", y="avg_mw_at_risk",
    color="fuel_category", text="generator_count",
    labels={"avg_mw_at_risk": "Avg MW at Risk", "fuel_category": "Fuel Type"}
)
fig1.update_traces(
    texttemplate="%{text} generators",
    textposition="outside"
)
fig1.update_layout(
    paper_bgcolor="#F7F8FA", plot_bgcolor="#FFFFFF",
    font=dict(color="#1A2332", family="IBM Plex Sans"),
    showlegend=False, height=400
)
st.plotly_chart(fig1, width='stretch')

st.markdown("---")


# ============================================================
# CHART 2: GENERATOR TREND OVER TIME
# ============================================================
st.subheader("Is This Generator Getting Worse Or Better?")

selected_asset_trend = st.selectbox(
    "Select a generator:",
    options=sorted(assets["asset_id"].tolist()),
    key="trend_asset"
)

asset_history = daily[daily["asset_id"] == selected_asset_trend].merge(
    assets, on="asset_id", how="left"
).sort_values("date").tail(30).copy()

asset_history_encoded = pd.get_dummies(asset_history, columns=categorical_cols)
for col in model_features:
    if col not in asset_history_encoded.columns:
        asset_history_encoded[col] = 0
asset_history["predicted_impact_ratio"] = model.predict(asset_history_encoded[model_features])
asset_history["predicted_mw_at_risk"] = (
    asset_history["predicted_impact_ratio"] * asset_history["dependable_capacity_mw"]
)

early_avg = asset_history["predicted_mw_at_risk"].head(5).mean()
recent_avg = asset_history["predicted_mw_at_risk"].tail(5).mean()
trend_delta = recent_avg - early_avg

if trend_delta > (0.03 * asset_history["dependable_capacity_mw"].iloc[0]):
    trend_word, trend_icon = "worsened", "↑"
elif trend_delta < -(0.03 * asset_history["dependable_capacity_mw"].iloc[0]):
    trend_word, trend_icon = "improved", "↓"
else:
    trend_word, trend_icon = "stayed roughly flat", "→"

st.markdown(
    f"**Finding:** {trend_icon} `{selected_asset_trend}`'s predicted risk has **{trend_word}** "
    f"over the last 30 days, moving from {early_avg:.1f} MW to {recent_avg:.1f} MW at risk."
)

fig2 = px.line(
    asset_history, x="date", y="predicted_mw_at_risk",
    markers=True,
    labels={"predicted_mw_at_risk": "Predicted MW at Risk", "date": "Date"}
)
fig2.update_traces(line_color="#3B82F6")
fig2.update_layout(
    paper_bgcolor="#F7F8FA", plot_bgcolor="#FFFFFF",
    font=dict(color="#1A2332", family="IBM Plex Sans"),
    height=350
)
st.plotly_chart(fig2, width='stretch')

st.markdown("---")


# ============================================================
# CHART 3: MW AT RISK BY FUEL TYPE, GROUPED BY TEMPERATURE BAND
# ============================================================
st.subheader("Which Fuel Type Is Most Weather-Sensitive?")

def temp_band(t):
    if t < 32:
        return "Cold (below 32°F)"
    elif t > 85:
        return "Hot (above 85°F)"
    else:
        return "Mild (32 to 85°F)"

latest_df["temp_band"] = latest_df["temp_avg"].apply(temp_band)

weather_summary = latest_df.groupby(["fuel_category", "temp_band"]).agg(
    avg_mw_at_risk=("predicted_mw_at_risk", "mean")
).reset_index()

pivot = weather_summary.pivot(index="fuel_category", columns="temp_band", values="avg_mw_at_risk")
if "Mild (32 to 85°F)" in pivot.columns:
    extremes = pivot.drop(columns=["Mild (32 to 85°F)"], errors="ignore")
    if not extremes.empty and not extremes.isna().all().all():
        sensitivity = (extremes.max(axis=1) - pivot["Mild (32 to 85°F)"]).dropna()
        if not sensitivity.empty:
            most_sensitive_fuel = sensitivity.idxmax()
            st.markdown(
                f"**Finding:** `{most_sensitive_fuel}` shows the largest jump in MW at risk during "
                f"extreme temperatures compared to mild weather."
            )
        else:
            st.markdown("**Finding:** Not enough data across temperature bands today.")
    else:
        st.markdown("**Finding:** Today's temperature range does not include extreme bands.")
else:
    st.markdown("**Finding:** Limited temperature spread in today's data.")

fig3 = px.bar(
    weather_summary, x="fuel_category", y="avg_mw_at_risk",
    color="temp_band", barmode="group",
    category_orders={"temp_band": ["Cold (below 32°F)", "Mild (32 to 85°F)", "Hot (above 85°F)"]},
    labels={
        "avg_mw_at_risk": "Avg MW at Risk",
        "fuel_category": "Fuel Type",
        "temp_band": "Temperature Band"
    }
)
fig3.update_layout(
    paper_bgcolor="#F7F8FA", plot_bgcolor="#FFFFFF",
    font=dict(color="#1A2332", family="IBM Plex Sans"),
    height=400,
    legend_title_text="Temperature Band"
)
st.plotly_chart(fig3, width='stretch')

st.markdown("---")


# ============================================================
# CHART 4: ZONE COMPARISON
# ============================================================
st.subheader("Which Zones Need Attention First?")

ranked_zones = zone_summary.sort_values("risk_pct", ascending=False).copy()
ranked_zones["zone_label"] = ranked_zones["operating_region"].apply(
    lambda z: f"Zone {z} · {zone_names.get(z, '')}"
)
top_zone = ranked_zones.iloc[0]

st.markdown(
    f"**Finding:** Zone `{top_zone['operating_region']}` ({zone_names.get(top_zone['operating_region'], '')}) "
    f"has the highest risk, **{top_zone['predicted_mw_at_risk']:.0f} MW at risk** "
    f"across {int(top_zone['num_generators'])} generators."
)

compare_zones = st.multiselect(
    "Filter to specific zones (optional, defaults to all):",
    options=ranked_zones["operating_region"].tolist(),
    default=ranked_zones["operating_region"].tolist(),
    format_func=lambda z: f"Zone {z} · {zone_names.get(z, '')}"
)
display_zones = (
    ranked_zones[ranked_zones["operating_region"].isin(compare_zones)]
    if compare_zones else ranked_zones
)

fig4 = px.bar(
    display_zones, x="zone_label", y="predicted_mw_at_risk",
    color="risk_pct", color_continuous_scale=["#16A34A", "#D97706", "#DC2626"],
    labels={"predicted_mw_at_risk": "Total MW at Risk", "zone_label": "Zone"},
    text="predicted_mw_at_risk"
)
fig4.update_traces(texttemplate="%{text:.0f} MW", textposition="outside")
fig4.update_layout(
    paper_bgcolor="#F7F8FA", plot_bgcolor="#FFFFFF",
    font=dict(color="#1A2332", family="IBM Plex Sans"),
    coloraxis_showscale=False, height=400
)
st.plotly_chart(fig4, width='stretch')

st.markdown("---")


# ============================================================
# CHART 5: MAINTENANCE RECENCY VS RISK
# ============================================================
st.subheader("Is Overdue Maintenance Driving Risk?")

median_days = latest_df["days_since_last_event"].median()
latest_df["service_status"] = latest_df["days_since_last_event"].apply(
    lambda d: "Recently Serviced" if d <= median_days else "Overdue"
)

service_summary = latest_df.groupby("service_status").agg(
    avg_mw_at_risk=("predicted_mw_at_risk", "mean")
).reset_index()

recent_val = service_summary.loc[
    service_summary["service_status"] == "Recently Serviced", "avg_mw_at_risk"
].values[0]
overdue_val = service_summary.loc[
    service_summary["service_status"] == "Overdue", "avg_mw_at_risk"
].values[0]
risk_increase_pct = ((overdue_val - recent_val) / recent_val) * 100 if recent_val > 0 else 0

st.markdown(
    f"**Finding:** Generators overdue for service show **{risk_increase_pct:.0f}% more MW at risk** on average "
    f"than recently serviced ones ({overdue_val:.1f} MW vs {recent_val:.1f} MW). "
    f"This is consistently SHAP's top global risk driver."
)

fig5 = px.bar(
    service_summary, x="service_status", y="avg_mw_at_risk",
    color="service_status",
    color_discrete_map={"Recently Serviced": "#16A34A", "Overdue": "#DC2626"},
    labels={"avg_mw_at_risk": "Avg MW at Risk", "service_status": ""},
    text="avg_mw_at_risk"
)
fig5.update_traces(texttemplate="%{text:.1f} MW", textposition="outside")
fig5.update_layout(
    paper_bgcolor="#F7F8FA", plot_bgcolor="#FFFFFF",
    font=dict(color="#1A2332", family="IBM Plex Sans"),
    showlegend=False, height=400
)
st.plotly_chart(fig5, width='stretch')