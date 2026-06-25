# ============================================================
# PAGE: ML PIPELINE
# Upload zone weather CSV. System auto-joins with assets.csv,
# feature engineers, runs multiple ML models, compares results,
# forecasts forward, downloads for operations team.
#
# Three agents (pure Python, LangGraph):
#   Agent 1: Data Validator - checks uploaded CSV
#   Agent 2: Feature Engineer - joins assets, engineers features
#   Agent 3: Model Runner - runs all models, ranks by MAE/R2
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import asyncio
from datetime import datetime, timedelta
from typing import TypedDict
from xgboost import XGBRegressor, Booster
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from langgraph.graph import StateGraph, END
import plotly.graph_objects as go
import plotly.express as px
from shared import load_css, zone_names

load_css("styles.css")

# ── Constants ──
categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]
EXCLUDE_COLS = ["date", "asset_id", "impact_ratio", "date_str"]

# ── Load assets once ──
@st.cache_data
def load_assets():
    return pd.read_csv("assets.csv")

@st.cache_resource
def load_base_model():
    _booster = Booster()
    _booster.load_model("model.json")
    m = XGBRegressor()
    m._Booster = _booster
    m._estimator_type = "regressor"
    return m

assets_df = load_assets()
base_model = load_base_model()
model_features = base_model.get_booster().feature_names

# ── Pipeline State ──
class PipelineState(TypedDict):
    uploaded_df: object
    assets_df: object
    validation_report: dict
    validation_passed: bool
    engineered_df: object
    X: object
    y: object
    has_labels: bool
    selected_models: list
    model_results: dict
    best_model_name: str
    predictions_df: object
    error: str

# ── Agent 1: Data Validator ──
def data_validator(state: PipelineState) -> PipelineState:
    df = state["uploaded_df"]
    report = {"errors": [], "warnings": [], "rows": len(df), "zones_found": []}

    required_cols = ["date", "operating_region", "temp_avg"]
    for col in required_cols:
        if col not in df.columns:
            report["errors"].append(f"Missing required column: `{col}`")

    if report["errors"]:
        return {**state, "validation_report": report, "validation_passed": False}

    valid_zones = assets_df["operating_region"].unique().tolist()
    uploaded_zones = df["operating_region"].unique().tolist()
    unknown_zones = [z for z in uploaded_zones if z not in valid_zones]
    if unknown_zones:
        report["errors"].append(f"Unknown zones: {unknown_zones}. Valid zones: {valid_zones}")

    report["zones_found"] = [z for z in uploaded_zones if z in valid_zones]

    if df["temp_avg"].isnull().any():
        report["warnings"].append("Some temp_avg values are null. Will be filled with zone mean.")

    if "temp_min" not in df.columns:
        report["warnings"].append("temp_min not found. Will be estimated as temp_avg - 8.")
    if "temp_max" not in df.columns:
        report["warnings"].append("temp_max not found. Will be estimated as temp_avg + 8.")

    if "impact_ratio" in df.columns:
        report["has_labels"] = True
        report["warnings"].append("impact_ratio found. Model comparison enabled.")
    else:
        report["has_labels"] = False

    passed = len(report["errors"]) == 0
    return {**state, "validation_report": report, "validation_passed": passed,
            "has_labels": report.get("has_labels", False)}

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

    merged = assets.merge(weather_df, on="operating_region", how="left")

    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
        merged["month"] = merged["date"].dt.month
        merged["day_of_year"] = merged["date"].dt.dayofyear
        merged["day_of_week"] = merged["date"].dt.dayofweek

    merged["temp_range"] = merged["temp_max"] - merged["temp_min"]
    merged["cold_day_flag"] = (merged["temp_avg"] < 20).astype(int)
    merged["hot_day_flag"] = (merged["temp_avg"] > 85).astype(int)

    if "season" not in merged.columns:
        def get_season(month):
            if month in [12, 1, 2]: return "Winter"
            elif month in [3, 4, 5]: return "Spring"
            elif month in [6, 7, 8]: return "Summer"
            else: return "Fall"
        if "month" in merged.columns:
            merged["season"] = merged["month"].apply(get_season)

    encoded = pd.get_dummies(merged, columns=categorical_cols)
    for col in model_features:
        if col not in encoded.columns:
            encoded[col] = 0

    y = None
    has_labels = state.get("has_labels", False)
    if has_labels and "impact_ratio" in encoded.columns:
        y = encoded["impact_ratio"]

    return {**state, "engineered_df": merged, "X": encoded[model_features], "y": y}

# ── Agent 3: Model Runner ──
def model_runner(state: PipelineState) -> PipelineState:
    if not state["validation_passed"]:
        return state

    X = state["X"]
    y = state["y"]
    has_labels = state.get("has_labels", False)
    selected = state["selected_models"]
    results = {}

    for model_name in selected:
        try:
            if model_name == "XGBoost (existing)":
                preds = base_model._Booster.predict(xgb.DMatrix(X))
                results[model_name] = {"preds": preds, "mae": None, "r2": None}

            elif model_name == "XGBoost (retrained)" and has_labels:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                m = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 objective="reg:squarederror")
                m.fit(X_tr, y_tr)
                preds = m.predict(X)
                te_preds = m.predict(X_te)
                results[model_name] = {
                    "preds": preds,
                    "mae": round(mean_absolute_error(y_te, te_preds), 4),
                    "r2": round(r2_score(y_te, te_preds), 4)
                }

            elif model_name == "LightGBM" and has_labels:
                import lightgbm as lgb
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                       num_leaves=31, random_state=42, verbose=-1)
                m.fit(X_tr, y_tr)
                preds = m.predict(X)
                te_preds = m.predict(X_te)
                results[model_name] = {
                    "preds": preds,
                    "mae": round(mean_absolute_error(y_te, te_preds), 4),
                    "r2": round(r2_score(y_te, te_preds), 4)
                }

            elif model_name == "Random Forest" and has_labels:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                m = RandomForestRegressor(n_estimators=100, max_depth=10,
                                          random_state=42, n_jobs=-1)
                m.fit(X_tr, y_tr)
                preds = m.predict(X)
                te_preds = m.predict(X_te)
                results[model_name] = {
                    "preds": preds,
                    "mae": round(mean_absolute_error(y_te, te_preds), 4),
                    "r2": round(r2_score(y_te, te_preds), 4)
                }

            elif model_name == "Gradient Boosting" and has_labels:
                X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                m = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                               learning_rate=0.05, random_state=42)
                m.fit(X_tr, y_tr)
                preds = m.predict(X)
                te_preds = m.predict(X_te)
                results[model_name] = {
                    "preds": preds,
                    "mae": round(mean_absolute_error(y_te, te_preds), 4),
                    "r2": round(r2_score(y_te, te_preds), 4)
                }
        except Exception as e:
            results[model_name] = {"preds": None, "mae": None, "r2": None, "error": str(e)}

    labeled_results = {k: v for k, v in results.items() if v.get("mae") is not None}
    if labeled_results:
        best = min(labeled_results, key=lambda k: labeled_results[k]["mae"])
    else:
        best = "XGBoost (existing)"

    eng_df = state["engineered_df"].copy()
    if best in results and results[best]["preds"] is not None:
        eng_df["predicted_impact_ratio"] = np.clip(results[best]["preds"], 0, 1)
        eng_df["predicted_impacted_mw"] = (
            eng_df["predicted_impact_ratio"] * eng_df["dependable_capacity_mw"]
        )
        eng_df["risk_level"] = eng_df["predicted_impact_ratio"].apply(
            lambda x: "🔴 HIGH" if x > 0.65 else ("🟡 MODERATE" if x > 0.45 else "🟢 LOW")
        )

    return {**state, "model_results": results, "best_model_name": best,
            "predictions_df": eng_df}

# ── Build LangGraph Pipeline ──
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

# ── Page UI ──
st.title("⚙️ ML Pipeline")
st.caption("Upload zone weather data. System joins generator data, engineers features, runs ML models, forecasts risk.")
st.markdown("---")

st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

st.markdown("### 1. Upload Weather Data")

with st.expander("What to upload"):
    st.markdown("""
Upload a CSV with weather observations per zone. The system automatically joins with the generator database.

**Required columns:**

| Column | Example |
|---|---|
| `date` | 2026-07-01 |
| `operating_region` | A |
| `temp_avg` | 88 |

**Optional columns:**

| Column | Example |
|---|---|
| `temp_min` | 75 |
| `temp_max` | 95 |
| `impact_ratio` | 0.62 (enables model comparison) |

**Valid zones:** A B C D E F G H I J K
""")
    sample = pd.DataFrame({
        "date": ["2026-07-01"] * 3,
        "operating_region": ["A", "B", "J"],
        "temp_avg": [88, 82, 91],
        "temp_min": [75, 70, 80],
        "temp_max": [95, 90, 98]
    })
    st.dataframe(sample, hide_index=True)
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button("Download sample CSV", buf.getvalue(),
                       file_name="sample_weather.csv", mime="text/csv")

uploaded_file = st.file_uploader("Upload weather CSV", type=["csv"])

if uploaded_file:
    try:
        df_upload = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.success(f"Loaded {len(df_upload):,} rows.")

    st.markdown("### 2. Select Models")
    has_labels_preview = "impact_ratio" in df_upload.columns

    model_options = ["XGBoost (existing)"]
    if has_labels_preview:
        model_options += ["XGBoost (retrained)", "LightGBM", "Random Forest", "Gradient Boosting"]
    else:
        st.info("Add `impact_ratio` column to your CSV to unlock model comparison and retraining.")

    selected_models = st.multiselect(
        "Models to run:",
        options=model_options,
        default=model_options[:3] if len(model_options) >= 3 else model_options
    )

    st.markdown("### 3. Forecast Horizon")
    horizon_label = st.selectbox(
        "Forecast ahead:",
        ["30 days", "90 days", "1 year (365 days)"]
    )
    horizon_days = {"30 days": 30, "90 days": 90, "1 year (365 days)": 365}[horizon_label]

    st.markdown("---")

    if st.button("Run Pipeline", type="primary"):
        with st.spinner("Running 3-agent pipeline..."):
            initial_state = PipelineState(
                uploaded_df=df_upload,
                assets_df=assets_df,
                validation_report={},
                validation_passed=False,
                engineered_df=None,
                X=None,
                y=None,
                has_labels=has_labels_preview,
                selected_models=selected_models,
                model_results={},
                best_model_name="",
                predictions_df=None,
                error=""
            )
            result = pipeline.invoke(initial_state)

        report = result["validation_report"]

        st.markdown("### Agent 1: Validation Report")
        if report.get("errors"):
            for e in report["errors"]:
                st.error(e)
            st.stop()
        else:
            st.success(f"Validation passed. {report['rows']:,} rows. Zones: {report['zones_found']}")
        for w in report.get("warnings", []):
            st.warning(w)

        st.markdown("### Agent 2: Feature Engineering")
        st.success(f"Joined with {len(assets_df)} generators. Built {len(model_features)} features.")

        st.markdown("### Agent 3: Model Results")

        model_results = result["model_results"]
        best_model = result["best_model_name"]

        labeled = {k: v for k, v in model_results.items() if v.get("mae") is not None}
        if labeled:
            comp_rows = []
            for mname, mdata in labeled.items():
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
                    y=[row["MAE"], 1 - row["R²"]],
                ))
            fig_comp.update_layout(
                barmode="group",
                title="Model Comparison (lower is better)",
                paper_bgcolor="#0D1B2A",
                plot_bgcolor="#0D1B2A",
                font=dict(color="#E2E8F0"),
                height=350
            )
            st.plotly_chart(fig_comp, width='stretch')

        preds_df = result["predictions_df"]

        if preds_df is not None and "predicted_impact_ratio" in preds_df.columns:

            st.markdown("### Generator Predictions")
            display_cols = [c for c in [
                "asset_id", "operating_region", "fuel_category",
                "dependable_capacity_mw", "predicted_impact_ratio",
                "predicted_impacted_mw", "risk_level"
            ] if c in preds_df.columns]
            st.dataframe(
                preds_df[display_cols].sort_values("predicted_impact_ratio", ascending=False),
                hide_index=True, width='stretch'
            )

            st.markdown("### Zone Risk Summary")
            zone_sum = preds_df.groupby("operating_region").agg(
                total_mw=("dependable_capacity_mw", "sum"),
                predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
                generators=("asset_id", "count")
            ).reset_index()
            zone_sum["risk_pct"] = (zone_sum["predicted_mw_at_risk"] / zone_sum["total_mw"]) * 100
            zone_sum["zone_name"] = zone_sum["operating_region"].map(zone_names)

            fig_zone = go.Figure()
            colors = ["#DC2626" if r > 65 else "#D97706" if r > 45 else "#16A34A"
                      for r in zone_sum["risk_pct"]]
            fig_zone.add_trace(go.Bar(
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
                height=400
            )
            st.plotly_chart(fig_zone, width='stretch')

            if len(labeled) > 1:
                st.markdown("### Model Prediction Comparison by Zone")
                zone_comp_data = []
                for mname, mdata in model_results.items():
                    if mdata.get("preds") is not None:
                        tmp = preds_df[["operating_region", "dependable_capacity_mw"]].copy()
                        tmp["pred"] = np.clip(mdata["preds"], 0, 1)
                        tmp["pred_mw"] = tmp["pred"] * tmp["dependable_capacity_mw"]
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
                        title="Risk % per Zone by Model",
                        labels={"operating_region": "Zone", "risk_pct": "Risk %", "model": "Model"},
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig_multi.update_layout(
                        paper_bgcolor="#0D1B2A",
                        plot_bgcolor="#0D1B2A",
                        font=dict(color="#E2E8F0"),
                        height=400
                    )
                    st.plotly_chart(fig_multi, width='stretch')

            st.markdown(f"### {horizon_label} Forecast")
            st.caption("Using existing XGBoost quantile forecasters (p05/p50/p95)")

            try:
                from shared import load_forecaster_models
                forecasters = load_forecaster_models()
                forecast_rows = []
                base_date = datetime.now()

                for day_offset in range(1, min(horizon_days, 90) + 1):
                    forecast_date = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                    day_df = preds_df.copy()
                    if "days_since_last_event" in day_df.columns:
                        day_df["days_since_last_event"] += day_offset
                    day_enc = pd.get_dummies(day_df, columns=categorical_cols)
                    for col in model_features:
                        if col not in day_enc.columns:
                            day_enc[col] = 0
                    X_day = day_enc[model_features]
                    caps = day_df["dependable_capacity_mw"].values
                    total = caps.sum()
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
                    name="p95 (worst case)"
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc_df["date"], y=fc_df["p05"],
                    fill="tonexty", mode="lines",
                    line=dict(color="#16A34A", width=0),
                    fillcolor="rgba(220,38,38,0.15)",
                    name="p05 (best case)"
                ))
                fig_fc.add_trace(go.Scatter(
                    x=fc_df["date"], y=fc_df["p50"],
                    mode="lines",
                    line=dict(color="#F59E0B", width=2),
                    name="p50 (median)"
                ))
                fig_fc.update_layout(
                    title=f"Capacity-Weighted Risk Forecast ({horizon_label})",
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

            st.markdown("### Download Results")
            out_buf = io.StringIO()
            preds_df[display_cols].to_csv(out_buf, index=False)
            st.download_button(
                "Download Predictions CSV",
                out_buf.getvalue(),
                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

            if labeled:
                comp_buf = io.StringIO()
                comp_df.to_csv(comp_buf, index=False)
                st.download_button(
                    "Download Model Comparison CSV",
                    comp_buf.getvalue(),
                    file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

else:
    st.info("Upload a weather CSV to start the pipeline.")
    st.markdown("""
**How the 3-agent pipeline works:**

1. **Validator Agent** checks zones, columns, null values
2. **Feature Engineer Agent** joins your weather data with the full generator database automatically
3. **Model Runner Agent** trains and compares XGBoost, LightGBM, Random Forest, Gradient Boosting

You only need `date`, `operating_region`, and `temp_avg`.
""")