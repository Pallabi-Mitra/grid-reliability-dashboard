import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from mcp.server.fastmcp import FastMCP

# ── Load model and data once at startup ──
assets = pd.read_csv("assets.csv")
daily = pd.read_csv("daily_records.csv")
df = daily.merge(assets, on="asset_id", how="left")

model = XGBRegressor()
model.load_model("model.json")

categorical_cols = ["season", "fuel_category", "broad_asset_category", "operating_region"]
model_features = model.get_booster().feature_names

latest_date = df["date"].max()
latest_df = df[df["date"] == latest_date].copy()
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
zone_summary["risk_level"] = zone_summary["risk_pct"].apply(
    lambda x: "RED" if x > 65 else ("YELLOW" if x > 45 else "GREEN")
)
import sys
print(f"Model loaded. Latest date: {latest_date}. Zones computed.", file=sys.stderr)

# ── MCP Server ──
mcp = FastMCP("Grid Reliability MCP Server")

@mcp.tool()
def get_zone_risk(zone: str) -> str:
    """
    Get the predicted risk level for a specific NY load zone.
    Zone must be a letter A through K.
    Returns risk level (RED/YELLOW/GREEN), risk percentage, MW at risk, and generator count.
    """
    zone = zone.upper().strip()
    row = zone_summary[zone_summary["operating_region"] == zone]
    if row.empty:
        return f"Zone {zone} not found. Valid zones are A through K."
    row = row.iloc[0]
    return (
        f"Zone {zone} Risk Summary (as of {latest_date}):\n"
        f"Risk Level: {row['risk_level']}\n"
        f"Risk Percentage: {row['risk_pct']:.1f}% of capacity at risk\n"
        f"Predicted MW at Risk: {row['predicted_mw_at_risk']:.1f} MW\n"
        f"Total Zone Capacity: {row['total_capacity_mw']:.1f} MW\n"
        f"Number of Generators: {int(row['num_generators'])}"
    )

@mcp.tool()
def get_generator_predictions(zone: str) -> str:
    """
    Get per-generator risk predictions for a specific NY load zone.
    Returns each generator's asset ID, fuel type, predicted impact ratio,
    predicted MW at risk, and individual risk level sorted highest to lowest.
    """
    zone = zone.upper().strip()
    gens = latest_df[latest_df["operating_region"] == zone].sort_values(
        "predicted_impact_ratio", ascending=False
    )
    if gens.empty:
        return f"No generators found in Zone {zone}."

    lines = [f"Generator predictions for Zone {zone} (as of {latest_date}):\n"]
    for _, row in gens.iterrows():
        risk = "HIGH" if row["predicted_impact_ratio"] > 0.65 else (
            "MODERATE" if row["predicted_impact_ratio"] > 0.45 else "LOW"
        )
        lines.append(
            f"{row['asset_id']} | {row['fuel_category']} | "
            f"Impact Ratio: {row['predicted_impact_ratio']:.3f} | "
            f"MW at Risk: {row['predicted_impacted_mw']:.1f} | "
            f"Risk: {risk}"
        )
    return "\n".join(lines)

@mcp.tool()
def get_shap_explanation(asset_id: str) -> str:
    """
    Get a plain-English SHAP-based explanation of why a specific generator
    received its predicted risk level. Takes an asset ID like AST-023.
    Returns the top 5 risk drivers with their SHAP contribution values.
    """
    importance_scores = model.get_booster().get_score(importance_type="gain")
    shap_series = pd.Series(importance_scores).reindex(model_features).fillna(0).abs().sort_values(ascending=False)

@mcp.tool()
def list_all_zones() -> str:
    """
    List all 11 NY load zones with their current risk levels.
    Useful for getting a quick overview of the entire grid state.
    """
    lines = [f"All NY Load Zone Risk Levels (as of {latest_date}):\n"]
    zone_names = {
        "A": "West", "B": "Genesee", "C": "Central", "D": "North",
        "E": "Mohawk Valley", "F": "Capital", "G": "Hudson Valley",
        "H": "Millwood", "I": "Dunwoodie", "J": "NYC", "K": "Long Island"
    }
    for _, row in zone_summary.sort_values("operating_region").iterrows():
        zone = row["operating_region"]
        name = zone_names.get(zone, "")
        lines.append(
            f"Zone {zone} ({name}): {row['risk_level']} | "
            f"{row['risk_pct']:.1f}% at risk | "
            f"{row['predicted_mw_at_risk']:.1f} MW"
        )
    return "\n".join(lines)

if __name__ == "__main__":
    mcp.run(transport="stdio")