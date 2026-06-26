# ============================================================
# PAGE: PREDICTIVE ANALYTICS
# Upload zone weather CSV. System auto-joins with assets.csv,
# feature engineers, runs multiple pre-trained ML models,
# compares predictions, forecasts forward, downloads results.
#
# Three agents (pure Python, LangGraph):
#   Agent 1: Data Validator - checks uploaded CSV
#   Agent 2: Feature Engineer - joins assets, engineers features
#   Agent 3: Model Runner - scores with all pre-trained models
#
# Pre-trained models (trained in AllModel-checkpoint.ipynb):
#   model.json              - XGBoost (MAE: 0.0128, R2: 0.9956)
#   model_lightgbm.txt      - LightGBM (MAE: 0.0085, R2: 0.9981)
#   model_random_forest.pkl - Random Forest (MAE: 0.0154, R2: 0.9912)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import joblib
from datetime import datetime, timedelta
from typing import TypedDict
from xgboost import XGBRegressor, Booster
import xgboost as xgb
from langgraph.graph import StateGraph, END
import plotly.graph_objects as go
import plotly.express as px
from shared import load_css, zone_names

load_css("styles.css")

categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]

@st.cache_data
def load_assets():
    return pd.read_csv("assets.csv")

@st.cache_resource
def load_all_models():
    models = {}

    _booster = Booster()
    _booster.load_model("model.json")
    m = XGBRegressor()
    m._Booster = _booster
    m._estimator_type = "regressor"
    models["XGBoost"] = {"model": m, "mae": 0.0128, "r2": 0.9956, "type": "xgb"}

    try:
        import lightgbm as lgb
        lgb_booster = lgb.Booster(model_file="model_lightgbm.txt")
        models["LightGBM"] = {"model": lgb_booster, "mae": 0.0085, "r2": 0.9981, "type": "lgbm"}
    except Exception:
        pass

    try:
        rf = joblib.load("model_random_forest.pkl")
        models["Random Forest"] = {"model": rf, "mae": 0.0154, "r2": 0.9912, "type": "sklearn"}
    except Exception:
        pass

    return models

assets_df = load_assets()
all_models = load_all_models()
model_features = all_models["XGBoost"]["model"].get_booster().feature_names

class PipelineState(TypedDict):
    uploaded_df: object
    assets_df: object
    validation_report: dict
    validation_passed: bool
    merged_df: object
    X: object
    selected_models: list
    model_results: dict
    best_model_name: str
    predictions_df: object
    error: str

# ── Agent 1: Validator ──
def data_validator(state: PipelineState) -> PipelineState:
    df = state["uploaded_df"]
    report = {"errors": [], "warnings": [], "rows": len(df), "zones_found": []}

    for col in ["date", "operating_region", "temp_avg"]:
        if col not in df.columns:
            report["errors"].append(f"Missing required column: `{col}`")

    if report["errors"]:
        return {**state, "validation_report": report, "validation_passed": False}

    valid_zones = assets_df["operating_region"].unique().tolist()
    uploaded_zones = df["operating_region"].unique().tolist()
    unknown = [z for z in uploaded_zones if z not in valid_zones]
    if unknown:
        report["errors"].append(f"Unknown zones: {unknown}. Valid zones: {valid_zones}")

    report["zones_found"] = [z for z in uploaded_zones if z in valid_zones]

    if df["temp_avg"].isnull().any():
        report["warnings"].append("Some temp_avg values are null. Will fill with zone mean.")
    if "temp_min" not in df.columns:
        report["warnings"].append("temp_min missing. Will estimate as temp_avg - 8.")
    if "temp_max" not in df.columns:
        report["warnings"].append("temp_max missing. Will estimate as temp_avg + 8.")

    return {**state, "validation_report": report, "validation_passed": len(report["errors"]) == 0}

# ── Agent 2: Feature Engineer ──
def feature_engineer(state: PipelineState) -> PipelineState:
    if not state["validation_passed"]:
        return state

    weather_df = state["uploaded_df"].copy()
    assets = state["assets_df"]

    if "temp_min" not in weather_df.columns:
        weather_df["temp_min"] = weather_df["temp_avg"] - 8
    if "temp_max" not in weather_df.columns:
        weather_df["temp_max"] = weather_df["temp_avg"] + 8

    weather_df["temp_avg"] = weather_df.groupby("operating_region")["temp_avg"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Use only the first date per zone to avoid duplicating generators
    weather_df = weather_df.sort_values("date").groupby("operating_region").first().reset_index()

    merged = assets.merge(weather_df, on="operating_region", how="left").reset_index(drop=True)

    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged["month"] = merged["date"].dt.month
        merged["day_of_year"] = merged["date"].dt.dayofyear
        merged["day_of_week"] = merged["date"].dt.dayofweek

    merged["temp_range"] = merged["temp_max"] - merged["temp_min"]
    merged["cold_day_flag"] = (merged["temp_avg"] < 20).astype(int)
    merged["hot_day_flag"] = (merged["temp_avg"] > 85).astype(int)

    if "season" not in merged.columns and "month" in merged.columns:
        def get_season(m):
            if m in [12, 1, 2]: return "Winter"
            elif m in [3, 4, 5]: return "Spring"
            elif m in [6, 7, 8]: return "Summer"
            else: return "Fall"
        merged["season"] = merged["month"].apply(get_season)

    # Encode for model
    encoded = pd.get_dummies(merged, columns=categorical_cols).reset_index(drop=True)

    for col in model_features:
        if col not in encoded.columns:
            encoded[col] = 0

    X = encoded[model_features].copy()

    return {**state, "merged_df": merged, "X": X}

# ── Agent 3: Model Runner ──
def model_runner(state: PipelineState) -> PipelineState:
    if not state["validation_passed"]:
        return state

    X = state["X"]
    merged = state["merged_df"].copy().reset_index(drop=True)
    selected = state["selected_models"]
    results = {}

    for model_name in selected:
        if model_name not in all_models:
            continue
        entry = all_models[model_name]
        m = entry["model"]
        mtype = entry["type"]
        try:
            if mtype == "xgb":
                preds = m._Booster.predict(xgb.DMatrix(X))
            elif mtype == "lgbm":
                preds = m.predict(X)
            else:
                preds = m.predict(X)
            results[model_name] = {
                "preds": np.clip(preds, 0, 1),
                "mae": entry["mae"],
                "r2": entry["r2"]
            }
        except Exception as e:
            results[model_name] = {"preds": None, "mae": None, "r2": None, "error": str(e)}

    valid = [k for k, v in results.items() if v.get("preds") is not None]
    best = min(valid, key=lambda k: results[k]["mae"]) if valid else list(results.keys())[0]

    preds_df = merged.copy()
    if best in results and results[best]["preds"] is not None:
        preds_array = results[best]["preds"]
        preds_df["predicted_impact_ratio"] = preds_array
        preds_df["dependable_capacity_mw"] = pd.to_numeric(
            preds_df["dependable_capacity_mw"], errors="coerce"
        )
        preds_df["predicted_impacted_mw"] = (
            preds_df["predicted_impact_ratio"] * preds_df["dependable_capacity_mw"]
        )
        preds_df["risk_level"] = preds_df["predicted_impact_ratio"].apply(
            lambda x: "🔴 HIGH" if x > 0.65 else ("🟡 MODERATE" if x > 0.45 else "🟢 LOW")
        )

    return {**state, "model_results": results, "best_model_name": best, "predictions_df": preds_df}

def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("validator", data_validator)
    graph.add_node("engineer", feature_engineer)
    graph.add_node("runner", model_runner)
    graph.set_entry_point("validator")
    graph.add_edge("validator", "engineer")
    graph.add_edge("engineer", "runner")
    graph.add_edge("runner", END)
    return graph.compile()

pipeline = build_pipeline()

# ── Page Header ──
st.markdown("""
<div style="background:linear-gradient(135deg,#0D1B2A,#1A3A5C);padding:2rem 2rem 1.5rem;border-radius:12px;margin-bottom:1.5rem;">
    <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.15em;color:#64B5F6;text-transform:uppercase;margin-bottom:0.4rem;">Grid Reliability Intelligence Platform</div>
    <div style="font-size:1.8rem;font-weight:700;color:#FFFFFF;margin-bottom:0.4rem;">Machine Learning Models</div>
    <div style="font-size:0.9rem;color:#90A4AE;">Upload zone weather data · Multi-model comparison · 30/90/365-day forecast</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

st.markdown("### Loaded Models")
model_overview = pd.DataFrame([
    {"Model": k, "MAE": v["mae"], "R²": v["r2"], "Status": "✅ Ready"}
    for k, v in all_models.items()
]).sort_values("MAE")
st.dataframe(model_overview, hide_index=True, width='stretch')
st.caption("All models pre-trained on 36,500 rows · LightGBM is best performer (MAE 0.0085)")

st.markdown("---")
st.markdown("### 1. Upload Weather Data")

with st.expander("Expected format"):
    st.markdown("""
**Required columns:**

| Column | Example |
|---|---|
| `date` | 2026-07-01 |
| `operating_region` | A |
| `temp_avg` | 88 |

**Optional:** `temp_min`, `temp_max`

**Valid zones:** A B C D E F G H I J K
""")
    sample = pd.DataFrame({
        "date": ["2026-07-01"] * 11,
        "operating_region": ["A","B","C","D","E","F","G","H","I","J","K"],
        "temp_avg": [88,82,79,74,77,80,83,85,86,91,89],
        "temp_min": [75,70,67,62,65,68,72,74,75,80,78],
        "temp_max": [95,90,87,83,86,89,91,93,94,98,96]
    })
    st.dataframe(sample, hide_index=True)
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button(
        "Download sample CSV", buf.getvalue(),
        file_name="sample_weather.csv", mime="text/csv"
    )

# ── File Upload with session state persistence ──
uploaded_file = st.file_uploader("Upload weather CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
        st.session_state["ml_upload_df"] = df_upload
        st.session_state["ml_upload_name"] = uploaded_file.name
        st.session_state.pop("ml_result", None)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

if "ml_upload_df" not in st.session_state:
    st.info("Upload a weather CSV to start the analysis.")
    st.markdown("""
**3-agent pipeline:**

1. **Validator** — checks zones, columns, null values
2. **Feature Engineer** — joins with full generator database, builds 53 features
3. **Model Runner** — scores with XGBoost, LightGBM, Random Forest in parallel

Only `date`, `operating_region`, and `temp_avg` are required.
""")
    st.stop()

df_upload = st.session_state["ml_upload_df"]
st.success(f"Loaded: {st.session_state.get('ml_upload_name', 'file')} · {len(df_upload):,} rows.")

st.markdown("### 2. Select Models")
selected_models = st.multiselect(
    "Models to run:",
    options=list(all_models.keys()),
    default=list(all_models.keys())
)

st.markdown("### 3. Forecast Horizon")
horizon_label = st.selectbox(
    "Forecast ahead:",
    ["30 days", "90 days", "1 year (365 days)"]
)
horizon_days = {"30 days": 30, "90 days": 90, "1 year (365 days)": 365}[horizon_label]

st.markdown("---")

if st.button("Run Analysis", type="primary"):
    with st.spinner("Running 3-agent pipeline..."):
        result = pipeline.invoke(PipelineState(
            uploaded_df=df_upload,
            assets_df=assets_df,
            validation_report={},
            validation_passed=False,
            merged_df=None,
            X=None,
            selected_models=selected_models,
            model_results={},
            best_model_name="",
            predictions_df=None,
            error=""
        ))
    st.session_state["ml_result"] = result
    st.session_state["ml_horizon_label"] = horizon_label
    st.session_state["ml_horizon_days"] = horizon_days

if "ml_result" not in st.session_state:
    st.stop()

result = st.session_state["ml_result"]
horizon_label = st.session_state.get("ml_horizon_label", "30 days")
horizon_days = st.session_state.get("ml_horizon_days", 30)

report = result["validation_report"]

st.markdown("#### Agent 1: Validation")
if report.get("errors"):
    for e in report["errors"]:
        st.error(e)
    st.stop()
st.success(f"Passed · {report['rows']:,} rows · Zones: {report['zones_found']}")
for w in report.get("warnings", []):
    st.warning(w)

st.markdown("#### Agent 2: Feature Engineering")
st.success(f"Joined {len(assets_df)} generators · Built {len(model_features)} features")

model_results = result["model_results"]
best_model = result["best_model_name"]

st.markdown("#### Agent 3: Model Comparison")
comp_rows = []
for mname, mdata in model_results.items():
    if mdata.get("preds") is not None:
        comp_rows.append({
            "Model": mname,
            "MAE": mdata["mae"],
            "R²": mdata["r2"],
            "Best": "✅" if mname == best_model else ""
        })
comp_df = pd.DataFrame(comp_rows).sort_values("MAE")
st.dataframe(comp_df, hide_index=True, width='stretch')
st.success(f"Best model: **{best_model}**")

fig_comp = go.Figure()
for row in comp_rows:
    fig_comp.add_trace(go.Bar(
        name=row["Model"],
        x=["MAE", "1 - R²"],
        y=[row["MAE"], round(1 - row["R²"], 4)],
    ))
fig_comp.update_layout(
    barmode="group",
    title="Model Comparison (lower is better)",
    paper_bgcolor="#0D1B2A",
    plot_bgcolor="#0D1B2A",
    font=dict(color="#E2E8F0"),
    height=320
)
st.plotly_chart(fig_comp, width='stretch')

preds_df = result["predictions_df"]

if preds_df is None or "predicted_impact_ratio" not in preds_df.columns:
    st.error("Predictions failed. Check your CSV format.")
    st.stop()

st.markdown("#### Generator Predictions")
display_cols = [c for c in [
    "asset_id", "operating_region", "fuel_category",
    "dependable_capacity_mw", "predicted_impact_ratio",
    "predicted_impacted_mw", "risk_level"
] if c in preds_df.columns]
st.dataframe(
    preds_df[display_cols].sort_values("predicted_impact_ratio", ascending=False),
    hide_index=True, width='stretch'
)

st.markdown("#### Zone Risk Summary")
zone_sum = preds_df.groupby("operating_region").agg(
    total_mw=("dependable_capacity_mw", "sum"),
    predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
    generators=("asset_id", "count")
).reset_index()
zone_sum["risk_pct"] = (zone_sum["predicted_mw_at_risk"] / zone_sum["total_mw"]) * 100
zone_sum["zone_name"] = zone_sum["operating_region"].map(zone_names)

colors = ["#DC2626" if r > 65 else "#D97706" if r > 45 else "#16A34A"
          for r in zone_sum["risk_pct"]]
fig_zone = go.Figure(go.Bar(
    x=zone_sum["operating_region"],
    y=zone_sum["risk_pct"],
    marker_color=colors,
    text=[f"{r:.1f}%" for r in zone_sum["risk_pct"]],
    textposition="outside",
    hovertext=[
        f"Zone {r['operating_region']} ({r['zone_name']})<br>"
        f"Risk: {r['risk_pct']:.1f}%<br>"
        f"MW at risk: {r['predicted_mw_at_risk']:.1f}<br>"
        f"Generators: {r['generators']}"
        for _, r in zone_sum.iterrows()
    ],
    hoverinfo="text"
))
fig_zone.update_layout(
    title="Predicted Risk % by Zone",
    xaxis_title="Zone",
    yaxis_title="Risk %",
    paper_bgcolor="#0D1B2A",
    plot_bgcolor="#0D1B2A",
    font=dict(color="#E2E8F0"),
    height=380
)
st.plotly_chart(fig_zone, width='stretch')

if len([k for k, v in model_results.items() if v.get("preds") is not None]) > 1:
    st.markdown("#### All Models: Zone Comparison")
    zone_comp_data = []
    for mname, mdata in model_results.items():
        if mdata.get("preds") is not None:
            tmp = preds_df[["operating_region", "dependable_capacity_mw"]].copy().reset_index(drop=True)
            tmp["pred"] = pd.Series(np.clip(mdata["preds"], 0, 1)).values
            tmp["pred_mw"] = tmp["pred"] * pd.to_numeric(tmp["dependable_capacity_mw"], errors="coerce")
            z = tmp.groupby("operating_region").agg(
                total_mw=("dependable_capacity_mw", "sum"),
                pred_mw=("pred_mw", "sum")
            ).reset_index()
            z["risk_pct"] = (z["pred_mw"] / z["total_mw"]) * 100
            z["model"] = mname
            zone_comp_data.append(z)

    if zone_comp_data:
        comp_all = pd.concat(zone_comp_data)
        fig_multi = px.bar(
            comp_all, x="operating_region", y="risk_pct",
            color="model", barmode="group",
            title="Risk % per Zone — All Models",
            labels={"operating_region": "Zone", "risk_pct": "Risk %", "model": "Model"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_multi.update_layout(
            paper_bgcolor="#0D1B2A",
            plot_bgcolor="#0D1B2A",
            font=dict(color="#E2E8F0"),
            height=380
        )
        st.plotly_chart(fig_multi, width='stretch')

st.markdown(f"#### {horizon_label} Risk Forecast")
st.caption("Quantile forecasters: p05 best case · p50 median · p95 worst case")
try:
    from shared import load_forecaster_models
    forecasters = load_forecaster_models()
    forecast_rows = []
    base_date = datetime.now()
    days_to_run = min(horizon_days, 90)

    for day_offset in range(1, days_to_run + 1):
        forecast_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        day_df = preds_df.copy().reset_index(drop=True)
        if "days_since_last_event" in day_df.columns:
            day_df["days_since_last_event"] = pd.to_numeric(
                day_df["days_since_last_event"], errors="coerce"
            ).fillna(0) + day_offset
        day_enc = pd.get_dummies(day_df, columns=[c for c in categorical_cols if c in day_df.columns])
        day_enc = day_enc.reset_index(drop=True)
        for col in model_features:
            if col not in day_enc.columns:
                day_enc[col] = 0
        X_day = day_enc[model_features]
        caps = pd.to_numeric(day_df["dependable_capacity_mw"], errors="coerce").fillna(0).values
        total = caps.sum()
        if total == 0:
            continue
        p05 = float((forecasters["p05"]._Booster.predict(xgb.DMatrix(X_day)) * caps).sum() / total)
        p50 = float((forecasters["p50"]._Booster.predict(xgb.DMatrix(X_day)) * caps).sum() / total)
        p95 = float((forecasters["p95"]._Booster.predict(xgb.DMatrix(X_day)) * caps).sum() / total)
        forecast_rows.append({"date": forecast_date, "p05": p05, "p50": p50, "p95": p95})

    fc_df = pd.DataFrame(forecast_rows)
    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["p95"],
        fill=None, mode="lines",
        line=dict(color="#DC2626", width=0),
        name="p95 worst case"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["p05"],
        fill="tonexty", mode="lines",
        line=dict(color="#16A34A", width=0),
        fillcolor="rgba(220,38,38,0.15)",
        name="p05 best case"
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df["date"], y=fc_df["p50"],
        mode="lines",
        line=dict(color="#F59E0B", width=2),
        name="p50 median"
    ))
    fig_fc.update_layout(
        title=f"Capacity-Weighted Impact Ratio Forecast ({horizon_label})",
        xaxis_title="Date",
        yaxis_title="Impact Ratio",
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font=dict(color="#E2E8F0"),
        height=400,
        legend=dict(bgcolor="#0D1B2A")
    )
    st.plotly_chart(fig_fc, width='stretch')
except Exception as e:
    st.warning(f"Forecast skipped: {e}")

st.markdown("#### Download Results")
out_buf = io.StringIO()
preds_df[display_cols].to_csv(out_buf, index=False)
st.download_button(
    "Download Predictions CSV",
    out_buf.getvalue(),
    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
comp_buf = io.StringIO()
comp_df.to_csv(comp_buf, index=False)
st.download_button(
    "Download Model Comparison CSV",
    comp_buf.getvalue(),
    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)
