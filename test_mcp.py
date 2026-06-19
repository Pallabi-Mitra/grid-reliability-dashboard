import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import shap

# ── Load same data as MCP server ──
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

zone_names = {
    "A": "West", "B": "Genesee", "C": "Central", "D": "North",
    "E": "Mohawk Valley", "F": "Capital", "G": "Hudson Valley",
    "H": "Millwood", "I": "Dunwoodie", "J": "NYC", "K": "Long Island"
}

# ── Test Tool 1: list_all_zones ──
print("=" * 60)
print("TOOL 1: list_all_zones")
print("=" * 60)
lines = [f"All NY Load Zone Risk Levels (as of {latest_date}):\n"]
for _, row in zone_summary.sort_values("operating_region").iterrows():
    zone = row["operating_region"]
    name = zone_names.get(zone, "")
    lines.append(
        f"Zone {zone} ({name}): {row['risk_level']} | "
        f"{row['risk_pct']:.1f}% at risk | "
        f"{row['predicted_mw_at_risk']:.1f} MW"
    )
print("\n".join(lines))

# ── Test Tool 2: get_zone_risk ──
print("\n" + "=" * 60)
print("TOOL 2: get_zone_risk(zone='B')")
print("=" * 60)
zone = "B"
row = zone_summary[zone_summary["operating_region"] == zone].iloc[0]
print(
    f"Zone {zone} Risk Summary (as of {latest_date}):\n"
    f"Risk Level: {row['risk_level']}\n"
    f"Risk Percentage: {row['risk_pct']:.1f}% of capacity at risk\n"
    f"Predicted MW at Risk: {row['predicted_mw_at_risk']:.1f} MW\n"
    f"Total Zone Capacity: {row['total_capacity_mw']:.1f} MW\n"
    f"Number of Generators: {int(row['num_generators'])}"
)

# ── Test Tool 3: get_generator_predictions ──
print("\n" + "=" * 60)
print("TOOL 3: get_generator_predictions(zone='B')")
print("=" * 60)
gens = latest_df[latest_df["operating_region"] == zone].sort_values(
    "predicted_impact_ratio", ascending=False
)
for _, r in gens.iterrows():
    risk = "HIGH" if r["predicted_impact_ratio"] > 0.65 else (
        "MODERATE" if r["predicted_impact_ratio"] > 0.45 else "LOW"
    )
    print(
        f"{r['asset_id']} | {r['fuel_category']} | "
        f"Impact Ratio: {r['predicted_impact_ratio']:.3f} | "
        f"MW at Risk: {r['predicted_impacted_mw']:.1f} | "
        f"Risk: {risk}"
    )

# ── Test Tool 4: get_shap_explanation ──
print("\n" + "=" * 60)
print("TOOL 4: get_shap_explanation(asset_id='AST-023')")
print("=" * 60)
asset_id = "AST-023"
asset_rows = latest_df[latest_df["asset_id"] == asset_id]
asset_encoded = pd.get_dummies(asset_rows, columns=categorical_cols)
for col in model_features:
    if col not in asset_encoded.columns:
        asset_encoded[col] = 0

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(asset_encoded[model_features])
shap_series = pd.Series(shap_values[0], index=model_features).abs().sort_values(ascending=False)
top5 = shap_series.head(5)

asset_info = asset_rows.iloc[0]
print(
    f"SHAP Explanation for {asset_id} "
    f"({asset_info['fuel_category']}, Zone {asset_info['operating_region']}):\n"
    f"Predicted Impact Ratio: {asset_info['predicted_impact_ratio']:.3f}\n"
    f"Predicted MW at Risk: {asset_info['predicted_impacted_mw']:.1f} MW\n"
    f"\nTop 5 risk drivers:"
)
for feature, importance in top5.items():
    val = asset_encoded[feature].values[0] if feature in asset_encoded.columns else 0
    print(f"  - {feature.replace('_', ' ')}: importance {importance:.4f}, value {val:.3f}")