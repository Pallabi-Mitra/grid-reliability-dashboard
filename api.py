# ============================================================
# FASTAPI SCORING ENDPOINT
# Serves pre-trained ML models as a REST API.
# Mirrors what AWS SageMaker does automatically when you
# deploy a model as an endpoint.
#
# Run locally:
#   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
#
# Test:
#   curl -X POST http://localhost:8000/score \
#     -H "Content-Type: application/json" \
#     -d '{"zones": [{"operating_region": "J", "temp_avg": 102, "temp_min": 89, "temp_max": 112}]}'
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
import logging
import time
import os
from datetime import datetime
from xgboost import XGBRegressor, Booster
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("grid-reliability-api")

app = FastAPI(
    title="Grid Reliability Scoring API",
    description="ML scoring endpoint for generator risk prediction. Mirrors AWS SageMaker endpoint pattern.",
    version="1.0.0"
)

categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]

# ── Load models at startup ──
@app.on_event("startup")
async def load_models():
    global models, assets_df, model_features

    logger.info("Loading pre-trained models...")

    assets = pd.read_csv("assets.csv")
    daily = pd.read_csv("daily_records.csv")
    latest_date = daily["date"].max()
    latest_daily = daily[daily["date"] == latest_date][[
        "asset_id", "dependable_capacity_mw", "days_since_last_event",
        "recent_avg_impact", "prev_impact_ratio", "recent_max_impact",
        "prior_high_impact_flag", "high_wind_flag", "impacted_mw"
    ]].copy()
    assets_df = assets.merge(latest_daily, on="asset_id", how="left")

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
    except Exception as e:
        logger.warning(f"LightGBM not loaded: {e}")

    try:
        rf = joblib.load("model_random_forest.pkl")
        models["Random Forest"] = {"model": rf, "mae": 0.0154, "r2": 0.9912, "type": "sklearn"}
    except Exception as e:
        logger.warning(f"Random Forest not loaded: {e}")

    model_features = models["XGBoost"]["model"].get_booster().feature_names
    logger.info(f"Loaded {len(models)} models: {list(models.keys())}")

# ── Request / Response schemas ──
class ZoneWeather(BaseModel):
    operating_region: str = Field(..., description="Zone letter A through K")
    temp_avg: float = Field(..., description="Average temperature in Fahrenheit")
    temp_min: Optional[float] = Field(None, description="Min temperature. Estimated if not provided.")
    temp_max: Optional[float] = Field(None, description="Max temperature. Estimated if not provided.")
    date: Optional[str] = Field(None, description="Date YYYY-MM-DD. Defaults to today.")

class ScoreRequest(BaseModel):
    zones: List[ZoneWeather] = Field(..., description="List of zone weather observations")
    models: Optional[List[str]] = Field(None, description="Models to run. Defaults to all loaded models.")

class GeneratorPrediction(BaseModel):
    asset_id: str
    operating_region: str
    fuel_category: str
    dependable_capacity_mw: float
    predicted_impact_ratio: float
    predicted_impacted_mw: float
    risk_level: str

class ModelResult(BaseModel):
    model_name: str
    mae: float
    r2: float
    is_best: bool

class ScoreResponse(BaseModel):
    request_id: str
    timestamp: str
    best_model: str
    latency_ms: float
    models_used: List[ModelResult]
    zone_summary: dict
    generators: List[GeneratorPrediction]

# ── Feature engineering ──
def engineer_features(weather_zones: List[ZoneWeather]) -> pd.DataFrame:
    weather_df = pd.DataFrame([{
        "operating_region": z.operating_region,
        "temp_avg": z.temp_avg,
        "temp_min": z.temp_min if z.temp_min is not None else z.temp_avg - 8,
        "temp_max": z.temp_max if z.temp_max is not None else z.temp_avg + 8,
        "date": z.date if z.date is not None else datetime.now().strftime("%Y-%m-%d")
    } for z in weather_zones])

    merged = assets_df.merge(weather_df, on="operating_region", how="left").reset_index(drop=True)

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["month"] = merged["date"].dt.month
    merged["day_of_year"] = merged["date"].dt.dayofyear
    merged["day_of_week"] = merged["date"].dt.dayofweek
    merged["temp_range"] = merged["temp_max"] - merged["temp_min"]
    merged["cold_day_flag"] = (merged["temp_avg"] < 20).astype(int)
    merged["hot_day_flag"] = (merged["temp_avg"] > 85).astype(int)

    def get_season(m):
        if m in [12, 1, 2]: return "Winter"
        elif m in [3, 4, 5]: return "Spring"
        elif m in [6, 7, 8]: return "Summer"
        else: return "Fall"

    if "season" not in merged.columns:
        merged["season"] = merged["month"].apply(get_season)

    temp_stress = np.where(
        merged["temp_avg"] > 85,
        (merged["temp_avg"] - 85) / 30.0,
        np.where(merged["temp_avg"] < 20, (20 - merged["temp_avg"]) / 30.0, 0.0)
    )
    temp_stress = np.clip(temp_stress, 0, 1)
    merged["impacted_mw"] = merged["impacted_mw"] * (1 + temp_stress)
    merged["recent_avg_impact"] = np.clip(merged["recent_avg_impact"] + temp_stress * 0.3, 0, 1)
    merged["prev_impact_ratio"] = np.clip(merged["prev_impact_ratio"] + temp_stress * 0.2, 0, 1)

    encoded = pd.get_dummies(merged, columns=categorical_cols).reset_index(drop=True)
    for col in model_features:
        if col not in encoded.columns:
            encoded[col] = 0

    return merged, encoded[model_features]

# ── Predict with one model ──
def predict(model_entry: dict, X: pd.DataFrame) -> np.ndarray:
    m = model_entry["model"]
    mtype = model_entry["type"]
    if mtype == "xgb":
        return np.clip(m._Booster.predict(xgb.DMatrix(X)), 0, 1)
    elif mtype == "lgbm":
        return np.clip(m.predict(X), 0, 1)
    else:
        return np.clip(m.predict(X), 0, 1)

# ── Endpoints ──
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "generators": len(assets_df),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest):
    start = time.time()
    request_id = f"req_{int(start * 1000)}"

    valid_zones = assets_df["operating_region"].unique().tolist()
    for z in request.zones:
        if z.operating_region not in valid_zones:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown zone: {z.operating_region}. Valid zones: {valid_zones}"
            )

    selected_models = request.models if request.models else list(models.keys())
    selected_models = [m for m in selected_models if m in models]

    if not selected_models:
        raise HTTPException(status_code=400, detail="No valid models selected.")

    merged_df, X = engineer_features(request.zones)

    results = {}
    for model_name in selected_models:
        try:
            preds = predict(models[model_name], X)
            results[model_name] = {
                "preds": preds,
                "mae": models[model_name]["mae"],
                "r2": models[model_name]["r2"]
            }
        except Exception as e:
            logger.error(f"Model {model_name} failed: {e}")

    best_model = min(results, key=lambda k: results[k]["mae"])
    best_preds = results[best_model]["preds"]

    merged_df["predicted_impact_ratio"] = best_preds
    merged_df["dependable_capacity_mw"] = pd.to_numeric(
        merged_df["dependable_capacity_mw"], errors="coerce"
    )
    merged_df["predicted_impacted_mw"] = (
        merged_df["predicted_impact_ratio"] * merged_df["dependable_capacity_mw"]
    )
    merged_df["risk_level"] = merged_df["predicted_impact_ratio"].apply(
        lambda x: "HIGH" if x > 0.65 else ("MODERATE" if x > 0.45 else "LOW")
    )

    zone_summary = {}
    for zone, grp in merged_df.groupby("operating_region"):
        total = grp["dependable_capacity_mw"].sum()
        at_risk = grp["predicted_impacted_mw"].sum()
        risk_pct = (at_risk / total * 100) if total > 0 else 0
        zone_summary[zone] = {
            "risk_pct": round(risk_pct, 2),
            "predicted_mw_at_risk": round(at_risk, 1),
            "total_mw": round(total, 1),
            "risk_level": "HIGH" if risk_pct > 65 else ("MODERATE" if risk_pct > 45 else "LOW"),
            "generators": len(grp)
        }

    generators = []
    for _, row in merged_df.iterrows():
        if pd.isna(row.get("predicted_impact_ratio")):
            continue
        generators.append(GeneratorPrediction(
            asset_id=str(row.get("asset_id", "")),
            operating_region=str(row.get("operating_region", "")),
            fuel_category=str(row.get("fuel_category", "")),
            dependable_capacity_mw=float(row.get("dependable_capacity_mw", 0) or 0),
            predicted_impact_ratio=round(float(row["predicted_impact_ratio"]), 4),
            predicted_impacted_mw=round(float(row.get("predicted_impacted_mw", 0) or 0), 1),
            risk_level=str(row.get("risk_level", "LOW"))
        ))

    latency_ms = round((time.time() - start) * 1000, 2)

    logger.info({
        "request_id": request_id,
        "zones": [z.operating_region for z in request.zones],
        "best_model": best_model,
        "latency_ms": latency_ms,
        "zone_risks": {z: v["risk_level"] for z, v in zone_summary.items()}
    })

    return ScoreResponse(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        best_model=best_model,
        latency_ms=latency_ms,
        models_used=[
            ModelResult(
                model_name=k,
                mae=v["mae"],
                r2=v["r2"],
                is_best=(k == best_model)
            )
            for k, v in results.items()
        ],
        zone_summary=zone_summary,
        generators=generators
    )

@app.get("/models")
def list_models():
    return {
        "models": [
            {
                "name": k,
                "mae": v["mae"],
                "r2": v["r2"],
                "type": v["type"]
            }
            for k, v in models.items()
        ]
    }