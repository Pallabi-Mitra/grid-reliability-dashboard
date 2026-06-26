# ============================================================
# PAGE: ZONE GENERATORS + RISK DRIVERS
# Two tabs in one page:
#   Tab 1: Generator Rankings - scan and rank all generators
#          in a zone by predicted risk level
#   Tab 2: Risk Drivers - pick a generator and see WHY the
#          model predicted its risk level using feature importance
# ============================================================

import streamlit as st
import pandas as pd
from shared import (
    load_css, get_live_weather_predictions, zone_names, categorical_cols
)

load_css("styles.css")

assets, daily, df, model, model_features, latest_date, latest_df, zone_summary = get_live_weather_predictions()

st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

st.markdown("""
<div style="background:linear-gradient(135deg,#0D1B2A,#1A3A5C);padding:2rem 2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;">
    <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.15em;color:#64B5F6;text-transform:uppercase;margin-bottom:0.4rem;">Grid Reliability Intelligence Platform</div>
    <div style="font-size:1.8rem;font-weight:700;color:#FFFFFF;margin-bottom:0.4rem;">Zone Generators</div>
    <div style="font-size:0.9rem;color:#90A4AE;">Per-generator risk rankings · Feature importance breakdown</div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- ZONE SELECTOR (shared across both tabs) ---
selected_zone = st.selectbox(
    "Select a zone:",
    options=sorted(zone_summary["operating_region"].tolist()),
    format_func=lambda z: f"Zone {z} · {zone_names.get(z, '')} · {zone_summary[zone_summary['operating_region']==z]['risk_level'].values[0]}"
)
st.session_state["selected_zone"] = selected_zone

# --- SCORE ALL GENERATORS IN ZONE (shared across both tabs) ---
zone_generators = latest_df[latest_df["operating_region"] == selected_zone].copy()
zone_gen_encoded = pd.get_dummies(zone_generators, columns=categorical_cols)
for col in model_features:
    if col not in zone_gen_encoded.columns:
        zone_gen_encoded[col] = 0

import xgboost as xgb
zone_generators["predicted_impact_ratio"] = model._Booster.predict(
    xgb.DMatrix(zone_gen_encoded[model_features])
)
zone_generators["predicted_impacted_mw"] = (
    zone_generators["predicted_impact_ratio"] * zone_generators["dependable_capacity_mw"]
)
zone_generators["risk_level"] = zone_generators["predicted_impact_ratio"].apply(
    lambda x: "🔴 HIGH" if x > 0.65 else ("🟡 MODERATE" if x > 0.45 else "🟢 LOW")
)

# --- ZONE METRICS ---
zs = zone_summary[zone_summary["operating_region"] == selected_zone].iloc[0]
zc1, zc2, zc3 = st.columns(3)
zc1.metric("Zone Risk Level", zs["risk_level"])
zc2.metric("Zone MW at Risk", f"{zs['predicted_mw_at_risk']:.1f} MW")
zc3.metric("Generators in Zone", str(int(zs["num_generators"])))

st.markdown("---")

# --- TWO TABS ---
tab1, tab2 = st.tabs(["Generator Rankings", "Risk Drivers"])

# ── TAB 1: GENERATOR RANKINGS ──
with tab1:
    st.markdown("##### All generators ranked by predicted risk")

    display_cols = ["asset_id", "fuel_category", "broad_asset_category",
                    "dependable_capacity_mw", "predicted_impact_ratio",
                    "predicted_impacted_mw", "risk_level", "recent_avg_impact"]
    display_cols = [c for c in display_cols if c in zone_generators.columns]

    st.dataframe(
        zone_generators[display_cols].sort_values("predicted_impact_ratio", ascending=False).round(3),
        width='stretch',
        hide_index=True
    )

# ── TAB 2: RISK DRIVERS ──
with tab2:
    st.markdown("##### Select a generator to see why the model predicted its risk level")

    selected_asset = st.selectbox(
        "Select a generator:",
        options=zone_generators["asset_id"].tolist(),
        format_func=lambda a: (
            f"{a} · "
            f"{zone_generators[zone_generators['asset_id']==a]['fuel_category'].values[0]} · "
            f"{zone_generators[zone_generators['asset_id']==a]['risk_level'].values[0]}"
        )
    )

    asset_row = zone_generators[zone_generators["asset_id"] == selected_asset].iloc[0]
    asset_encoded = zone_gen_encoded[zone_generators["asset_id"] == selected_asset][model_features]

    # Feature importance using XGBoost native gain
    importance_scores = model.get_booster().get_score(importance_type="gain")
    shap_series = (
        pd.Series(importance_scores)
        .reindex(model_features)
        .fillna(0)
        .abs()
        .sort_values(ascending=False)
    )
    top_features = shap_series.head(5)

    def explain_feature(fname):
        val = asset_encoded[fname].values[0] if fname in asset_encoded.columns else 0
        explanations = {
            "recent_avg_impact": f"recent average impact ratio is high ({val:.3f})" if val > 0.4 else f"recent average impact is moderate ({val:.3f})",
            "prev_impact_ratio": f"previous day impact was high ({val:.3f})" if val > 0.4 else f"previous day impact was low ({val:.3f})",
            "cold_day_flag": "today is a cold stress day (temp < 20°F)" if val == 1 else "no cold stress today",
            "hot_day_flag": "today is a heat stress day (temp > 85°F)" if val == 1 else "no heat stress today",
            "high_wind_flag": "high wind conditions detected" if val == 1 else "wind within normal range",
            "temp_avg": f"average temperature is {val:.1f}°F",
            "dependable_capacity_mw": f"generator capacity is {val:.1f} MW",
            "days_since_last_event": f"{int(val)} days since last high-impact event",
            "recent_max_impact": f"recent peak impact ratio was {val:.3f}",
            "prior_high_impact_flag": "prior high-impact event on record" if val == 1 else "no prior high-impact flag",
        }
        return explanations.get(fname, f"{fname.replace('_', ' ')} = {val:.3f}")

    explanation_lines = [f"• {explain_feature(fname)}" for fname in top_features.index]

    risk_emoji = (
        "🔴" if asset_row["risk_level"] == "🔴 HIGH"
        else "🟡" if asset_row["risk_level"] == "🟡 MODERATE"
        else "🟢"
    )

    st.markdown(f"""
<div class="app-card">
<h4>{risk_emoji} {selected_asset} · {asset_row['fuel_category']} · {asset_row['broad_asset_category']}</h4>
<p class="app-card-meta">
Predicted Impact Ratio: <strong>{asset_row['predicted_impact_ratio']:.3f}</strong> ·
Predicted MW at Risk: <strong>{asset_row['predicted_impacted_mw']:.1f} MW</strong>
</p>
<p class="app-card-driver-label">Top 5 risk drivers:</p>
{"".join(f'<p class="app-card-driver-line">' + line + '</p>' for line in explanation_lines)}
</div>
""", unsafe_allow_html=True)
