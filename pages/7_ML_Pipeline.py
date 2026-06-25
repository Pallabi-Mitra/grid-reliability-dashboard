# ============================================================
# PAGE: ML PIPELINE
# Upload generator degradation CSV, run multiple ML models,
# compare performance, generate predictions, download results.
#
# Handles 3 upload formats:
#   A) Labeled data (has impact_ratio) -> train + compare models
#   B) Raw data (base columns, no labels) -> feature engineer + score
#   C) Pre-engineered (has 53 model features) -> score directly
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta
from xgboost import XGBRegressor, Booster
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from shared import load_css, zone_names

load_css("styles.css")

# ── Constants ──
categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]
EXCLUDE_COLS = ["date", "asset_id", "impact_ratio", "date_str"]

REQUIRED_RAW_COLS = [
    "asset_id", "date", "dependable_capacity_mw", "temp_avg",
    "temp_min", "temp_max", "season", "fuel_category",
    "broad_asset_category", "operating_region"
]

RISK_HORIZON_OPTIONS = {
    "30 days": 30,
    "90 days": 90,
    "1 year (365 days)": 365
}

# ── Load existing trained model ──
@st.cache_resource
def load_base_model():
    _booster = Booster()
    _booster.load_model("model.json")
    m = XGBRegressor()
    m._Booster = _booster
    m._estimator_type = "regressor"
    return m

base_model = load_base_model()
model_features = base_model.get_booster().feature_names

# ── Helpers ──
def get_risk_label(ratio):
    if ratio > 0.65:
        return "🔴 HIGH"
    elif ratio > 0.45:
        return "🟡 MODERATE"
    return "🟢 LOW"

def feature_engineer(df):
    encoded = pd.get_dummies(df, columns=categorical_cols)
    for col in model_features:
        if col not in encoded.columns:
            encoded[col] = 0
    return encoded

def detect_format(df):
    has_label = "impact_ratio" in df.columns
    has_all_features = all(f in df.columns for f in model_features)
    has_raw_cols = all(c in df.columns for c in REQUIRED_RAW_COLS)
    if has_all_features:
        return "pre_engineered", has_label
    elif has_raw_cols:
        return "raw", has_label
    return "unknown", has_label

def score_with_model(model, X):
    if hasattr(model, "_Booster"):
        return model._Booster.predict(xgb.DMatrix(X))
    return model.predict(X)

def build_zone_summary(df):
    summary = df.groupby("operating_region").agg(
        total_capacity_mw=("dependable_capacity_mw", "sum"),
        predicted_mw_at_risk=("predicted_impacted_mw", "sum"),
        num_generators=("asset_id", "count")
    ).reset_index()
    summary["risk_pct"] = (summary["predicted_mw_at_risk"] / summary["total_capacity_mw"]) * 100
    summary["risk_level"] = summary["risk_pct"].apply(
        lambda x: "🔴 RED" if x > 65 else ("🟡 YELLOW" if x > 45 else "🟢 GREEN")
    )
    return summary

def generate_forecast(scored_df, horizon_days):
    rows = []
    base_date = datetime.now()
    for day in range(1, horizon_days + 1):
        forecast_date = (base_date + timedelta(days=day)).strftime("%Y-%m-%d")
        day_df = scored_df.copy()
        if "days_since_last_event" in day_df.columns:
            day_df["days_since_last_event"] += day
        enc = feature_engineer(day_df)
        preds = score_with_model(base_model, enc[model_features])
        zone_avg = day_df.copy()
        zone_avg["pred"] = preds
        zone_avg["pred_mw"] = preds * day_df["dependable_capacity_mw"]
        z = zone_avg.groupby("operating_region").agg(
            total_mw=("dependable_capacity_mw", "sum"),
            pred_mw_at_risk=("pred_mw", "sum")
        ).reset_index()
        z["risk_pct"] = (z["pred_mw_at_risk"] / z["total_mw"]) * 100
        z["date"] = forecast_date
        z["day_offset"] = f"Day +{day}"
        rows.append(z)
        if day in [1, horizon_days // 4, horizon_days // 2, horizon_days]:
            pass
    return pd.concat(rows, ignore_index=True)

# ── Page Layout ──
st.title("⚙️ ML Pipeline")
st.caption("Upload generator degradation data, compare ML models, forecast future risk")
st.markdown("---")

st.sidebar.markdown(
    "<div class='sidebar-footer'>Synthetic data for demo purposes. No real operational data used.</div>",
    unsafe_allow_html=True
)

# ── Upload Section ──
st.markdown("### 1. Upload Generator Data")
st.markdown(
    "Upload a CSV with generator degradation records. "
    "The page auto-detects the format and runs the right pipeline."
)

with st.expander("Expected CSV format"):
    st.markdown("""
**Minimum required columns (raw format):**

| Column | Description |
|---|---|
| `asset_id` | Generator ID e.g. AST-001 |
| `date` | Date string YYYY-MM-DD |
| `dependable_capacity_mw` | Generator capacity in MW |
| `temp_avg` | Average temperature °F |
| `temp_min` | Min temperature °F |
| `temp_max` | Max temperature °F |
| `season` | Summer / Winter / Spring / Fall |
| `fuel_category` | Gas / Hydro / Nuclear / Oil / Wind / Solar |
| `broad_asset_category` | Thermal / Renewable / Nuclear |
| `operating_region` | Zone letter A through K |

**Optional:** Include `impact_ratio` (0.0-1.0) to unlock model comparison and evaluation.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

    st.success(f"Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

    fmt, has_labels = detect_format(df_raw)

    if fmt == "unknown":
        st.error(
            "CSV format not recognized. Missing required columns. "
            "Expand 'Expected CSV format' above to see what's needed."
        )
        missing = [c for c in REQUIRED_RAW_COLS if c not in df_raw.columns]
        st.write("Missing columns:", missing)
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", f"{len(df_raw):,}")
    with col2:
        zones_found = df_raw["operating_region"].nunique() if "operating_region" in df_raw.columns else "?"
        st.metric("Zones", zones_found)
    with col3:
        generators_found = df_raw["asset_id"].nunique() if "asset_id" in df_raw.columns else "?"
        st.metric("Generators", generators_found)

    if fmt == "pre_engineered":
        st.info("Format detected: Pre-engineered (53 model features present). Scoring directly.")
    elif fmt == "raw":
        st.info("Format detected: Raw generator data. Running feature engineering then scoring.")
    if has_labels:
        st.info("Target column `impact_ratio` found. Model comparison available.")

    st.markdown("---")

    # ── Model Selection ──
    st.markdown("### 2. Select Models")

    model_options = {
        "XGBoost (trained model)": "xgb_base",
        "XGBoost (retrained on upload)": "xgb_new",
        "LightGBM": "lgbm",
        "Random Forest": "rf",
        "Gradient Boosting": "gbm",
    }

    if not has_labels:
        available = {"XGBoost (trained model)": "xgb_base"}
        st.warning("No `impact_ratio` column found. Only the existing trained model can score this data. Add `impact_ratio` to compare models.")
    else:
        available = model_options

    selected_model_names = st.multiselect(
        "Choose models to run:",
        options=list(available.keys()),
        default=["XGBoost (trained model)"] if not has_labels else [
            "XGBoost (trained model)", "XGBoost (retrained on upload)", "Random Forest"
        ]
    )

    if not selected_model_names:
        st.warning("Select at least one model.")
        st.stop()

    # ── Forecast Horizon ──
    st.markdown("### 3. Forecast Horizon")
    horizon_label = st.selectbox(
        "How far ahead to forecast:",
        options=list(RISK_HORIZON_OPTIONS.keys()),
        index=0
    )
    horizon_days = RISK_HORIZON_OPTIONS[horizon_label]

    st.markdown("---")

    # ── Run Pipeline ──
    if st.button("Run ML Pipeline", type="primary"):

        with st.spinner("Running pipeline..."):

            # Feature engineer if needed
            if fmt == "raw":
                df_enc = feature_engineer(df_raw)
            else:
                df_enc = df_raw.copy()
                for col in model_features:
                    if col not in df_enc.columns:
                        df_enc[col] = 0

            X = df_enc[model_features]
            y = df_raw["impact_ratio"] if has_labels else None

            results = {}
            trained_models = {}

            for model_name in selected_model_names:
                key = available[model_name]

                if key == "xgb_base":
                    preds = score_with_model(base_model, X)
                    trained_models[model_name] = base_model

                elif key == "xgb_new":
                    if has_labels:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                        m = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05,
                                         subsample=0.8, colsample_bytree=0.8, random_state=42,
                                         objective="reg:squarederror")
                        m.fit(X_tr, y_tr)
                        preds = m.predict(X)
                        trained_models[model_name] = m
                    else:
                        continue

                elif key == "lgbm":
                    try:
                        import lightgbm as lgb
                        if has_labels:
                            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                            m = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05,
                                                   num_leaves=31, random_state=42, verbose=-1)
                            m.fit(X_tr, y_tr)
                            preds = m.predict(X)
                            trained_models[model_name] = m
                        else:
                            continue
                    except ImportError:
                        st.warning("LightGBM not installed. Add `lightgbm` to requirements.txt.")
                        continue

                elif key == "rf":
                    if has_labels:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                        m = RandomForestRegressor(n_estimators=100, max_depth=10,
                                                   random_state=42, n_jobs=-1)
                        m.fit(X_tr, y_tr)
                        preds = m.predict(X)
                        trained_models[model_name] = m
                    else:
                        continue

                elif key == "gbm":
                    if has_labels:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                        m = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                       learning_rate=0.05, random_state=42)
                        m.fit(X_tr, y_tr)
                        preds = m.predict(X)
                        trained_models[model_name] = m
                    else:
                        continue

                results[model_name] = preds

            if not results:
                st.error("No models ran. Check your data format.")
                st.stop()

        # ── Model Comparison ──
        if has_labels and len(results) > 1:
            st.markdown("### Model Comparison")
            comparison_rows = []
            for mname, preds in results.items():
                mae = mean_absolute_error(y, preds)
                r2 = r2_score(y, preds)
                comparison_rows.append({
                    "Model": mname,
                    "MAE": round(mae, 4),
                    "R²": round(r2, 4)
                })
            comp_df = pd.DataFrame(comparison_rows).sort_values("MAE")
            best_model_name = comp_df.iloc[0]["Model"]

            st.dataframe(comp_df, hide_index=True, width='stretch')
            st.success(f"Best model by MAE: **{best_model_name}**")
        else:
            best_model_name = list(results.keys())[0]

        # ── Predictions from Best Model ──
        st.markdown("### Generator Predictions")
        st.caption(f"Using: {best_model_name}")

        best_preds = results[best_model_name]
        df_out = df_raw.copy()
        df_out["predicted_impact_ratio"] = np.clip(best_preds, 0, 1)
        df_out["predicted_impacted_mw"] = df_out["predicted_impact_ratio"] * df_out["dependable_capacity_mw"]
        df_out["risk_level"] = df_out["predicted_impact_ratio"].apply(get_risk_label)

        display_cols = ["asset_id", "operating_region", "fuel_category",
                        "dependable_capacity_mw", "predicted_impact_ratio",
                        "predicted_impacted_mw", "risk_level"]
        display_cols = [c for c in display_cols if c in df_out.columns]

        st.dataframe(
            df_out[display_cols].sort_values("predicted_impact_ratio", ascending=False),
            hide_index=True,
            width='stretch'
        )

        # ── Zone Summary ──
        if "operating_region" in df_out.columns and "dependable_capacity_mw" in df_out.columns:
            st.markdown("### Zone Risk Summary")
            zone_sum = build_zone_summary(df_out)
            st.dataframe(zone_sum, hide_index=True, width='stretch')

        # ── Forecast ──
        st.markdown(f"### {horizon_label} Forecast")
        st.caption("Projecting forward using existing XGBoost model with daily time step")

        if fmt == "raw" and "operating_region" in df_raw.columns:
            with st.spinner(f"Generating {horizon_label} forecast..."):
                forecast_df = generate_forecast(df_raw, min(horizon_days, 30))

            pivot = forecast_df.pivot_table(
                index="date", columns="operating_region", values="risk_pct"
            ).reset_index()
            st.dataframe(pivot, hide_index=True, width='stretch')
        else:
            st.info("Forecast requires raw format with `operating_region` and base columns.")

        # ── Download ──
        st.markdown("### Download Results")

        csv_buffer = io.StringIO()
        df_out.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name=f"ml_pipeline_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

        if has_labels and len(results) > 1:
            comp_buffer = io.StringIO()
            comp_df.to_csv(comp_buffer, index=False)
            st.download_button(
                label="Download Model Comparison CSV",
                data=comp_buffer.getvalue(),
                file_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

else:
    st.info("Upload a CSV file above to get started.")
    st.markdown("### What this page does")
    st.markdown("""
- **Upload** generator degradation data in CSV format
- **Auto-detect** format: raw columns, pre-engineered features, or labeled data
- **Compare models**: XGBoost, LightGBM, Random Forest, Gradient Boosting
- **Forecast** generator risk for 30, 90, or 365 days ahead
- **Download** predictions and model comparison for your operations team
""")
